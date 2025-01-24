import logging
import time
import os
import sys
import argparse
from typing import OrderedDict
import torch
#import resource, psutil
from functools import reduce
import pickle
import ctypes

from pippy.IR import Pipe, annotate_split_points, PipeSplitWrapper
from pippy.IR import LossWrapper
from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec
from pippy.microbatch import LossReducer
import pippy.fx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe, LossWrapper, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.events import EventsContext
from pippy.microbatch import sum_reducer, TensorChunkSpec
from pippy.visualizer import events_to_json



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from timm.utils import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.data import create_carla_dataset, create_carla_loader, resolve_data_config
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    convert_splitbn_model,
    model_parameters,
)

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

NUM_ITERATION = 100

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

pippy.fx.Tracer.proxy_buffer_attributes = True

USE_TQDM = bool(int(os.getenv('USE_TQDM', '1')))

def log_memory_usage(stage):
    print("1:{}".format(torch.cuda.memory_allocated(0)))

def debug_pickle(obj, name):
    try:
        pickle.dumps(obj)
    except TypeError as e:
        print(f"Error pickling {name}: {e}")

def is_pycapsule(obj):
    PyCapsule_CheckExact = ctypes.pythonapi.PyCapsule_CheckExact
    PyCapsule_CheckExact.argtypes = [ctypes.py_object]
    PyCapsule_CheckExact.restype = ctypes.c_int
    return PyCapsule_CheckExact(obj) == 1

def has_submodules_if_pycapsule(obj):
    if is_pycapsule(obj):
        try:
            # for example, check if 'submodules' or a similar property exists
            return hasattr(obj, "submodules") and obj.submodules
        except AttributeError:
            return False
    return False

class LAVLoss(nn.Module):
    def __init__(self):
        super(LAVLoss, self).__init__()
        self.prob_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.loc_criterion = nn.L1Loss(reduction='none')
        self.ori_criterion = nn.L1Loss(reduction='none')
        self.box_criterion = nn.L1Loss(reduction='none')
        self.spd_criterion = nn.L1Loss(reduction='none')

    def __call__(self, output, target):
        prob = target[:, :, 0:1]
        prob_mean = prob.mean()
        prob_mean = torch.maximum(prob_mean, torch.ones_like(prob_mean) * 1e-7)
        prob_det = torch.sigmoid(output[:, :, 0] * (1 - 2 * target[:, :, 0]))

        det_loss = (prob_det * self.prob_criterion(output[:, :, 0], target[:, :, 0])).mean() / prob_det.mean()
        loc_loss = (prob * self.loc_criterion(output[:, :, 1:3], target[:, :, 1:3])).mean() / prob_mean
        box_loss = (prob * self.box_criterion(output[:, :, 3:5], target[:, :, 3:5])).mean() / prob_mean
        ori_loss = (prob * self.ori_criterion(output[:, :, 5:7], target[:, :, 5:7])).mean() / prob_mean
        spd_loss = (prob * self.ori_criterion(output[:, :, 7:8], target[:, :, 7:8])).mean() / prob_mean

        det_loss = 0.4 * det_loss + 0.2 * loc_loss + 0.2 * box_loss + 0.2 * ori_loss
        return det_loss, spd_loss

class MemFuserLoss(nn.Module):
    def __init__(self):
        super(MemFuserLoss, self).__init__()
        self.traffic = LAVLoss()
        self.waypoints = torch.nn.L1Loss()
        self.cls = nn.CrossEntropyLoss()
        self.stop_cls = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        loss_traffic, loss_velocity = self.traffic(output[0], target[4])
        loss_waypoints = self.waypoints(output[1], target[1])
        loss_traffic_light_state = self.cls(output[2], target[3])
        loss_stop_sign = self.stop_cls(output[3], target[6])
        
        loss = (
            loss_traffic * 0.5
            + loss_waypoints * 0.5
            + loss_velocity * 0.05
            + loss_traffic_light_state * 0.1
            + loss_stop_sign * 0.01
        )
        return loss


class ModelLossWrapper(LossWrapper):
    def __init__(self, module, loss_fn):
        super(ModelLossWrapper, self).__init__(module, loss_fn)

    def forward(self, input, target):
        output = self.module(input)
        loss = self.loss_fn(output, target)
        return output, loss


def loss_reducer_fn(a, b):
    return a + b

def initialize_pipeline(model):
    loss_wrapper = ModelLossWrapper(module=model, loss_fn=MemFuserLoss())
    debug_pickle(loss_wrapper, "loss_wrapper")

    annotate_split_points(model, {
        #'position_encoding': PipeSplitWrapper.SplitPoint.END,
        'encoder': PipeSplitWrapper.SplitPoint.BEGINNING,
        #'decoder': PipeSplitWrapper.SplitPoint.BEGINNING
    })
    output_loss_value_spec = (False, True)
    pipe = Pipe.from_tracing(loss_wrapper, output_loss_value_spec=output_loss_value_spec)
    debug_pickle(pipe, "pipe")

    #start to initialize the pipeline driver
    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}

    output_chunk_spec = LossReducer(0.0, loss_reducer_fn)

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    driver = PipelineDriverFillDrain(
            pipe, args_chunk_spec=args_chunk_spec, kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec, world_size=world_size)
    debug_pickle(driver, "driver")
    return driver

def get_optimizer(args, model, _logger):
    linear_scaled_lr = (
        args.lr * args.batch_size * torch.distributed.get_world_size() / 512.0
    )
    args.lr = linear_scaled_lr
    if args.with_backbone_lr:
        if args.local_rank == 0:
            _logger.info(
                "CNN backbone and transformer blocks using different learning rates!"
            )
        backbone_linear_scaled_lr = (
            args.backbone_lr
            * args.batch_size
            * torch.distributed.get_world_size()
            / 512.0
        )
        backbone_weights = []
        other_weights = []
        for name, weight in model.named_parameters():
            if "backbone" in name and "lidar" not in name:
                backbone_weights.append(weight)
            else:
                other_weights.append(weight)
        if args.local_rank == 0:
            _logger.info(
                "%d weights in the cnn backbone, %d weights in other modules"
                % (len(backbone_weights), len(other_weights))
            )
        optimizer = create_optimizer_v2(
            [
                {"params": other_weights},
                {"params": backbone_weights, "lr": backbone_linear_scaled_lr},
            ],
            **optimizer_kwargs(cfg=args),
        )
    else:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    return optimizer

def train_one_epoch_pipeline(
    epoch,
    pipelineDriver,
    loader,
    optimizer,
    loss_fns,
    args,
    writer,
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=None,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
):
    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_waypoints = AverageMeter()
    losses_traffic = AverageMeter()
    losses_velocity = AverageMeter()
    losses_traffic_light_state = AverageMeter()
    losses_stop_sign = AverageMeter()
    start = time.time()
    pipelineDriver.train()

    end = time.time()
    last_idx = len(loader) - 1
    print(f"Length of loader: {len(loader)}")
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if isinstance(input, (tuple, list)):
            batch_size = input[0].size(0)
        elif isinstance(input, dict):
            batch_size = input[list(input.keys())[0]].size(0)
        else:
            batch_size = input.size(0)
        # CYH: not prefetcher, move to cuda here, so prefetcher needs to be False, i.e. args.-no-prefetcher = True
        if not args.prefetcher:
            if isinstance(input, (tuple, list)):
                input = [x.cuda() for x in input]
            elif isinstance(input, dict):
                for key in input:
                    if isinstance(input[key], list):
                        continue
                    input[key] = input[key].cuda()
            else:
                input = input.cuda()
            if isinstance(target, (tuple, list)):
                target = [x.cuda() for x in target]
            elif isinstance(target, dict):
                for key in target:
                    target[key] = target[key].cuda()
            else:
                target = target.cuda()

        print(f"Input: {input.keys()}")
        output, loss = pipelineDriver(input, target)

        losses_m.update(loss.item(), batch_size)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(
                    pipelineDriver, exclude_head="agc" in args.clip_mode
                ),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(pipelineDriver, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        if model_ema is not None:
            model_ema.update(pipelineDriver)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        batch_time_m.update(end - start)

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])

def main():
    _logger = logging.getLogger("train")
    model = create_model(
        "memfuser_baseline_e1d3",
        pretrained=False,
        drop_rate=0.0,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=0.1,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        scriptable=True,
        checkpoint_path=None,
        freeze_num=-1,
        )

    pipelineDriver = initialize_pipeline(model)
    '''
    #initalize the optimizer for single GPU training
    optimizer = get_optimizer(args, model, _logger)
    #initalize the optimizer for pipeline training
    optimizer_pipe = pipelineDriver.instantiate_optimizer(optimizer)
    
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    lr_scheduler_pipe = pipelineDriver.instantiate_lr_scheduler(
        lr_scheduler, total_iters=num_epochs
    )
    
    
    
    NUM_ITERATIONs = 100

    for i in range(NUM_ITERATIONs):
        train_one_epoch_pipeline(
            i,
            pipelineDriver,
            dataset_train,
            optimizer_pipe,
            MemFuserLoss(),
            args,
            writer=None,
            lr_scheduler=lr_scheduler_pipe,
            saver=None,
            output_dir=None,
            amp_autocast=suppress,
            loss_scaler=None,
            model_ema=None,
            mixup_fn=None,
        )
 '''   
    
import sys

# Increase the recursion limit
sys.setrecursionlimit(3000)

# Your existing code
def run_master(_, args):
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)
    print("Using device:", args.device)

    if args.rank == 0:
        number_of_workers = 2
        all_worker_ranks = list(range(0, number_of_workers))  # include master rank = 0
        chunks = len(all_worker_ranks)
        batch_size = args.batch_size * chunks


        #initalize the dataloader
        dataset_train = create_carla_dataset(
            args.dataset,
            root=args.data_dir,
            towns=args.train_towns,
            weathers=args.train_weathers,
            batch_size=args.batch_size,
            with_lidar=args.with_lidar,
            with_seg=args.with_seg,
            with_depth=args.with_depth,
            multi_view=args.multi_view,
            augment_prob=args.augment_prob,
            temporal_frames=args.temporal_frames
        )
        #print(F"Length of dataset: {dataset_train.route_frames}")
        dataset_eval = create_carla_dataset(
            args.dataset,
            root=args.data_dir,
            towns=args.val_towns,
            weathers=args.val_weathers,
            batch_size=args.batch_size,
            with_lidar=args.with_lidar,
            with_seg=args.with_seg,
            with_depth=args.with_depth,
            multi_view=args.multi_view,
            augment_prob=args.augment_prob,
            temporal_frames=args.temporal_frames,
        )
        
        collate_fn = None
        mixup_fn = None



        class OutputLossWrapper(LossWrapper):
            def __init__(self, module, loss_fn):
                super().__init__(module, loss_fn)

            @torch.autocast(device_type="cuda")
            def forward(self, input, target):
                output = self.module(input)
                loss = self.loss_fn(output, target)
                return {"output": output, "loss": loss}

        log_memory_usage("Before initializing model")

        model = create_model(
            "memfuser_baseline_e1d3",
            pretrained=False,
            drop_rate=0.0,
            drop_connect_rate=None,
            drop_path_rate=0.1,
            drop_block_rate=None,
            global_pool=None,
            bn_tf=False,
            bn_momentum=None,
            bn_eps=None,
            scriptable=True,
            checkpoint_path=None,
            freeze_num=-1,
        )
        debug_pickle(model, "model")

        log_memory_usage("After initializing model")


        data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0
        )
        
        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            train_interpolation = data_config["interpolation"]
        
        loader_train = create_carla_loader(
            dataset_train,
            input_size=data_config["input_size"],
            batch_size=args.batch_size,
            multi_view_input_size=args.multi_view_input_size,
            is_training=True,
            scale=args.scale,
            color_jitter=args.color_jitter,
            interpolation=train_interpolation,
            mean=data_config["mean"],
            std=data_config["std"],
            #num_workers=args.workers,
            #distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            )
        args.prefetcher = not args.no_prefetcher
        annotate_split_points(model, {
            'encoder': PipeSplitWrapper.SplitPoint.BEGINNING,
            #'decoder': PipeSplitWrapper.SplitPoint.BEGINNING
        })

        wrapper = OutputLossWrapper(model, MemFuserLoss())

        pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG)
        pipe.to(args.device)
        debug_pickle(pipe, "pipe")
        

        log_memory_usage("After creating Pipe")

        output_chunk_spec = (TensorChunkSpec(0), loss_reducer_fn)
        pipe_driver: PipelineDriverBase = schedules[args.schedule](pipe, chunks,
                                                                len(all_worker_ranks),
                                                                all_ranks=all_worker_ranks,
                                                                output_chunk_spec=output_chunk_spec,
                                                                _record_mem_dumps=bool(args.record_mem_dumps),
                                                                checkpoint=bool(args.checkpoint))
        #debug_pickle(pipe_driver, "pipe_driver")

        optimizer = pipe_driver.instantiate_optimizer(optim.Adam, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        '''
        lr_scheduler, num_epochs = create_scheduler(args, optimizer)
        lr_scheduler_pipe = pipe_driver.instantiate_lr_scheduler(
            lr_scheduler, total_iters=num_epochs
        )
        '''
        
        
        log_memory_usage("After creating optimizer")
        
        this_file_name = os.path.splitext(os.path.basename(__file__))[0]
        pipe_visualized_filename = f"{this_file_name}_visualized_{args.rank}.json"
        batches_events_contexts = []
        
        for i in range(NUM_ITERATION):
            train_one_epoch_pipeline(
                i,
                pipe_driver,
                loader_train,
                optimizer,
                MemFuserLoss(),
                args,
                writer=None,
                lr_scheduler=None,
                saver=None,
                output_dir=None,
                amp_autocast=None,
                loss_scaler=None,
                model_ema=None,
                mixup_fn=None,
            )

        if args.visualize:
            all_events_contexts: EventsContext = reduce(lambda c1, c2: EventsContext().update(c1).update(c2),
                                                        batches_events_contexts, EventsContext())
            with open(pipe_visualized_filename, "w") as f:
                f.write(events_to_json(all_events_contexts))
            print(f"Saved {pipe_visualized_filename}")
        print('Finished')
        print(f"Communication overload: {pipe_driver.get_communication_overload()}")
        print(f"Data transferred: {pipe_driver.get_data_transferred_mb()} MB")

    else:
        print("This is a worker rank")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 3)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--visualize', type=int, default=0, choices=[0, 1])
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--num_worker_threads', type=int, default=16)
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    #The following arguments are used for the dataloader
    parser.add_argument("--train-towns", type=int, nargs="+", default=[1,2,3,4,5,6,7,10])
    parser.add_argument("--val-towns", type=int, nargs="+", default=[1])
    parser.add_argument("--train-weathers", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19])
    parser.add_argument("--val-weathers", type=int, nargs="+", default=[1])
    parser.add_argument("--with-lidar", action="store_true", default=True)
    parser.add_argument("--with-seg", action="store_true", default=False)
    parser.add_argument("--with-depth", action="store_true", default=False)
    parser.add_argument("--multi-view", action="store_true", default=True)
    parser.add_argument("--multi-view-input-size", default=None, nargs=3, type=int)
    #The following arguments are used for the model
    parser.add_argument(
    "--sched",
    default="cosine",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step")',
    )
    parser.add_argument("--dataset", type=str, default="carla")
    parser.add_argument("--data-dir", type=str, default="./Caraladata/Device1/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )
    parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--lr-cycle-limit",
        type=int,
        default=1,
        metavar="N",
        help="learning rate cycle limit",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=5e-6,
        metavar="LR",
        help="warmup learning rate (default: 0.0001)",
    )
    parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
    )
    parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )

    #data_config resolver args
    parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=True,
    help="disable fast prefetcher",
    )
    # Augmentation & regularization parameters
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="Disable all training augmentation, override other train aug args",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[0.08, 1.0],
        metavar="PCT",
        help="Random resize scale (default: 0.08 1.0)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs="+",
        default=[3.0 / 4.0, 4.0 / 3.0],
        metavar="RATIO",
        help="Random resize aspect ratio (default: 0.75 1.33)",
    )
    parser.add_argument(
        "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
    )
    parser.add_argument(
        "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
    )
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default=None,
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". (default: None)',
    ),
    parser.add_argument(
        "--aug-splits",
        type=int,
        default=0,
        help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
    )
    parser.add_argument(
        "--jsd",
        action="store_true",
        default=False,
        help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
    )
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Random erase prob (default: 0.)",
    )
    parser.add_argument(
        "--remode", type=str, default="const", help='Random erase mode (default: "const")'
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.0, help="Label smoothing (default: 0.0)"
    )
    parser.add_argument(
        "--smoothed_l1", default=False, action='store_true', help="L1 smooth"
    )
    parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
    )

    parser.add_argument(
    "--num-classes",
    type=int,
    default=None,
    metavar="N",
    help="number of label classes (Model default if None)",
    )
    parser.add_argument(
        "--gp",
        default=None,
        type=str,
        metavar="POOL",
        help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        metavar="N",
        help="Image patch size (default: None => model default)",
    )
    parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
    )

    parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override mean pixel value of dataset",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override std deviation of of dataset",
    )
    parser.add_argument(
        "--interpolation",
        default="",
        type=str,
        metavar="NAME",
        help="Image resize interpolation type (overrides model)",
    )

    parser.add_argument("--backbone-lr", type=float, default=1e-3)
    parser.add_argument("--with-backbone-lr", action="store_true", default=False)
    parser.add_argument("--clip-grad", type=float, default=None)
    parser.add_argument("--clip-mode", type=str, default="norm")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment-prob", type=float, default=0.5)
    parser.add_argument("--temporal-frames", type=int, default=1)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--freeze-num", type=int, default=-1)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-connect-rate", type=float, default=None)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--drop-block-rate", type=float, default=None)
    parser.add_argument("--global-pool", type=str, default=None)
    parser.add_argument("--bn-tf", action="store_true", default=False)
    parser.add_argument("--bn-momentum", type=float, default=None)
    parser.add_argument("--bn-eps", type=float, default=None)
    parser.add_argument("--scriptable", action="store_true", default=False)

    
    
    
    args = parser.parse_args()

    from datetime import datetime
    logging_path = '/home/cailab/xtaWorkspace/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_pipeline_driver.log'
    logging.basicConfig(
        level=logging.DEBUG,
        force=True,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )

    print(f"Logging to {logging_path}")
    
    print(torch.cuda.is_available())
    run_pippy(run_master, args)
