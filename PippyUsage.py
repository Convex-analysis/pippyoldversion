import logging
import time
import os
import sys
import argparse
from typing import OrderedDict
import torch
import resource, psutil
from functools import reduce

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
from timm.data import create_carla_dataset, create_carla_loader
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


def initialize_pipeline(model):
    '''
    print(" The following model is not warpped with a loss function ".center(80, "+"))
    pipe = Pipe.from_tracing(model)
    print(pipe)
    print(" The following model is warpped with a loss function ".center(80, "*"))
    '''
    loss_wrapper = ModelLossWrapper(module=model, loss_fn=MemFuserLoss())

    annotate_split_points(model, {
        #'position_encoding': PipeSplitWrapper.SplitPoint.END,
        'encoder': PipeSplitWrapper.SplitPoint.BEGINNING,
        'decoder': PipeSplitWrapper.SplitPoint.BEGINNING
    })
    output_loss_value_spec = (False, True)
    pipe = Pipe.from_tracing(loss_wrapper, output_loss_value_spec=output_loss_value_spec)
    print(pipe)

    #start to initialize the pipeline driver
    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}

    output_chunk_spec = LossReducer(0.0, lambda a, b: a + b)

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1 

    driver = PipelineDriverFillDrain(
            pipe, args_chunk_spec=args_chunk_spec, kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec, world_size=world_size)
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

        with amp_autocast():
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
            temporal_frames=args.temporal_frames,
        )
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

        log_memory_usage("After initializing model")

        annotate_split_points(model, {
            'encoder': PipeSplitWrapper.SplitPoint.BEGINNING,
            'decoder': PipeSplitWrapper.SplitPoint.BEGINNING
        })

        wrapper = OutputLossWrapper(model, MemFuserLoss())

        pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG)
        pipe.to(args.device)

        log_memory_usage("After creating Pipe")

        output_chunk_spec = (TensorChunkSpec(0), sum_reducer)
        pipe_driver: PipelineDriverBase = schedules[args.schedule](pipe, chunks,
                                                                len(all_worker_ranks),
                                                                all_ranks=all_worker_ranks,
                                                                output_chunk_spec=output_chunk_spec,
                                                                _record_mem_dumps=bool(args.record_mem_dumps),
                                                                checkpoint=bool(args.checkpoint))

        optimizer = pipe_driver.instantiate_optimizer(optim.Adam, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        lr_scheduler, num_epochs = create_scheduler(args, optimizer)
        lr_scheduler_pipe = pipe_driver.instantiate_lr_scheduler(
            lr_scheduler, total_iters=num_epochs
        )
        
        log_memory_usage("After creating optimizer")
        
        this_file_name = os.path.splitext(os.path.basename(__file__))[0]
        pipe_visualized_filename = f"{this_file_name}_visualized_{args.rank}.json"
        batches_events_contexts = []
        
        for i in range(NUM_ITERATION):
            train_one_epoch_pipeline(
                i,
                pipe_driver,
                dataset_train,
                optimizer,
                MemFuserLoss(),
                args,
                writer=None,
                lr_scheduler=lr_scheduler_pipe,
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
    args = parser.parse_args()
    
    print(torch.cuda.is_available())
    run_pippy(run_master, args)