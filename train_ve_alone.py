import logging
import time
import os
import sys
import argparse
import csv
import torch
from functools import reduce
import pickle
import ctypes

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

try:
    import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

NUM_ITERATION = 20
USE_TQDM = bool(int(os.getenv('USE_TQDM', '1')))


def log_memory_usage(stage):
    try:
        if JTOP_AVAILABLE:
            with jtop.jtop() as jetson:
                if jetson.ok():
                    memory_used = jetson.memory['RAM']["shared"]  # Memory used in KB
                    memory_used = memory_used / 1024  # Convert to MB
                    print(f"Memory used ({stage}): {memory_used:.2f} MB")
                    return memory_used
        # Fall back to torch CUDA memory if jtop not available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
            print(f"CUDA Memory used ({stage}): {memory_used:.2f} MB")
            return memory_used
    except Exception as e:
        print(f"Error reading memory: {e}")
    return 0


class LAVLoss(torch.nn.Module):
    def __init__(self):
        super(LAVLoss, self).__init__()
        self.prob_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.loc_criterion = torch.nn.L1Loss(reduction='none')
        self.ori_criterion = torch.nn.L1Loss(reduction='none')
        self.box_criterion = torch.nn.L1Loss(reduction='none')
        self.spd_criterion = torch.nn.L1Loss(reduction='none')

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


class MemFuserLoss(torch.nn.Module):
    def __init__(self):
        super(MemFuserLoss, self).__init__()
        self.traffic = LAVLoss()
        self.waypoints = torch.nn.L1Loss()
        self.cls = torch.nn.CrossEntropyLoss()
        self.stop_cls = torch.nn.CrossEntropyLoss()

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


def get_optimizer(args, model, _logger):
    linear_scaled_lr = (
        args.lr * args.batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1) / 512.0
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
            * (torch.cuda.device_count() if torch.cuda.is_available() else 1)
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


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    writer=None,
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
    
    # Set the model to training mode
    model.train()
    
    # Track metrics for this epoch
    memory_measurements = []
    memory_measurements.append(log_memory_usage(f"Start of epoch {epoch}"))
    total_samples_processed = 0

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
            
        # Update total samples count
        total_samples_processed += batch_size
        
        # Move data to CUDA if available
        if torch.cuda.is_available():
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

        # Forward pass
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(input)
            loss = loss_fn(output, target)

        losses_m.update(loss.item(), batch_size)

        # Optimizer step
        optimizer.zero_grad()
        
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head="agc" in args.clip_mode),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        batch_time_m.update(time.time() - end)
        end = time.time()
        
        # Record memory usage after batch
        if batch_idx % 10 == 0:  # Record memory every 10 batches to reduce overhead
            memory_measurements.append(log_memory_usage(f"After batch {batch_idx} in epoch {epoch}"))
            
            # Print progress
            print(f"Epoch: {epoch}, Batch: {batch_idx}/{last_idx}, Loss: {losses_m.val:.4f} ({losses_m.avg:.4f}), Time: {batch_time_m.val:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Calculate average memory usage for this epoch
    avg_memory_usage = sum(memory_measurements) / len(memory_measurements) if memory_measurements else 0
    print(f"Epoch {epoch} average memory usage: {avg_memory_usage:.2f} MB")
    print(f"Epoch {epoch} total samples processed: {total_samples_processed}")
    
    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    # Return metrics
    epoch_metrics = {
        "loss": losses_m.avg,
        "total_samples": total_samples_processed,
        "avg_memory_usage": avg_memory_usage,
        "epoch_time": end - time.time()
    }
    return epoch_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--num_worker_threads', type=int, default=4)

    # Dataset arguments
    parser.add_argument("--train-towns", type=int, nargs="+", default=[1,2,3,4,5,6,7,10])
    parser.add_argument("--val-towns", type=int, nargs="+", default=[1])
    parser.add_argument("--train-weathers", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19])
    parser.add_argument("--val-weathers", type=int, nargs="+", default=[1])
    parser.add_argument("--with-lidar", action="store_true", default=True)
    parser.add_argument("--with-seg", action="store_true", default=False)
    parser.add_argument("--with-depth", action="store_true", default=False)
    parser.add_argument("--multi-view", action="store_true", default=True)
    parser.add_argument("--multi-view-input-size", default=None, nargs=3, type=int)
    parser.add_argument("--dataset", type=str, default="carla")
    parser.add_argument("--data-dir", type=str, default="./Caraladata/Device1/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer/training arguments
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)
    parser.add_argument("--with-backbone-lr", action="store_true", default=False)
    parser.add_argument("--clip-grad", type=float, default=None)
    parser.add_argument("--clip-mode", type=str, default="norm")
    parser.add_argument("--amp", action="store_true", default=False)

    # Scheduler parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "step")',
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
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

    # Data loader parameters
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
    
    # Augmentation parameters
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
        "--color-jitter",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="random",
        help='Training interpolation (random, bilinear, bicubic default: "random")',
    )
    parser.add_argument("--augment-prob", type=float, default=0.5)
    
    # Model parameters
    parser.add_argument("--temporal-frames", type=int, default=1)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--freeze-num", type=int, default=-1)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--drop-block-rate", type=float, default=None)
    parser.add_argument("--drop-connect-rate", type=float, default=None)
    
    # Other parameters
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--global-pool", type=str, default=None)
    parser.add_argument("--bn-tf", action="store_true", default=False)
    parser.add_argument("--bn-momentum", type=float, default=None)
    parser.add_argument("--bn-eps", type=float, default=None)
    parser.add_argument("--input-size", default=None, nargs=3, type=int)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--mean", type=float, nargs="+", default=None)
    parser.add_argument("--std", type=float, nargs="+", default=None)
    parser.add_argument("--interpolation", default="", type=str)
    parser.add_argument("--crop-pct", default=None, type=float)
    parser.add_argument("--scriptable", action="store_true", default=False)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    _logger = logging.getLogger("train")

    # Check CUDA availability
    if torch.cuda.is_available():
        _logger.info(f"Using CUDA with {torch.cuda.device_count()} device(s)")
    else:
        _logger.info("CUDA not available, using CPU")

    # Initialize model
    log_memory_usage("Before initializing model")
    model = create_model(
        "memfuser_baseline_e1d3",
        pretrained=False,
        drop_rate=args.drop_rate,
        drop_connect_rate=args.drop_connect_rate,
        drop_path_rate=args.drop_path_rate,
        drop_block_rate=args.drop_block_rate,
        global_pool=args.global_pool,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.scriptable,
        checkpoint_path=args.checkpoint_path,
        freeze_num=args.freeze_num,
    )
    log_memory_usage("After initializing model")

    # Move model to device
    device = torch.device(args.device)
    model = model.to(device)

    # Setup datasets and data loaders
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

    # Configure data loading
    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]
    
    # Create data loaders
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
        num_workers=args.num_worker_threads,
        pin_memory=args.pin_mem,
    )

    # Initialize loss function
    loss_fn = MemFuserLoss()
    
    # Initialize optimizer
    optimizer = get_optimizer(args, model, _logger)
    
    # Initialize scheduler
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    
    # Initialize AMP scaler for mixed precision training
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    loss_scaler = create_loss_scaler() if args.amp else None
    
    # Set up CSV file for recording metrics
    this_file_name = os.path.splitext(os.path.basename(__file__))[0]
    filecreatetime = time.strftime("%Y%m%d-%H%M%S")
    metrics_file = f"{this_file_name}_metrics_{filecreatetime}.csv"
    
    with open(metrics_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Epoch execution time (s)', 'Loss', 'Total samples', 'Average memory usage (MB)'])
    
    print(f"Starting training for {NUM_ITERATION} epochs")
    start_time = time.time()

    # Training loop
    for epoch in range(NUM_ITERATION):
        epoch_start_time = time.time()
        
        # Run one epoch and get metrics
        metrics = train_one_epoch(
            epoch,
            model,
            loader_train,
            optimizer,
            loss_fn,
            args,
            writer=None,
            lr_scheduler=lr_scheduler,
            saver=None,
            output_dir=None,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            model_ema=None,
            mixup_fn=None,
        )
        
        # Calculate epoch execution time
        epoch_execution_time = time.time() - epoch_start_time
        
        # Write metrics to CSV
        with open(metrics_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                epoch,
                f"{epoch_execution_time:.2f}",
                f"{metrics['loss']:.4f}",
                metrics["total_samples"],
                f"{metrics['avg_memory_usage']:.2f}"
            ])
        
        print(f"Epoch {epoch} execution time: {epoch_execution_time:.2f} seconds")
        print(f"Metrics saved to {metrics_file}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
