import argparse
import os
import torch
import torch.nn as nn
from torchprofile import profile_macs
from timm.models.memfuser import memfuser_baseline
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

def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 3)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)

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

    parser.add_argument('--rpc_timeout', type=int, default=600)
    
    args = parser.parse_args()
    
    return args

def conv_flops(H, W, C_in, C_out, K):
    """
    Compute FLOPs for a convolutional layer.
    FLOPs = 2 * H * W * C_in * C_out * K * K
    """
    return 2 * H * W * C_in * C_out * K * K

def bn_flops(H, W, C_in):
    """
    Compute FLOPs for a batch normalization layer.
    FLOPs = 2 * H * W * C_in
    """
    return 2 * H * W * C_in

def pool_flops(H, W, C, K):
    """
    Compute FLOPs for a pooling layer.
    FLOPs = H * W * C * K * K
    """
    return H * W * C * K * K

def pointwise_flops(N, D, C_out):
    """
    Compute FLOPs for point-wise feature computation.
    FLOPs = N * D * C_out
    """
    return N * D * C_out

def pillarwise_flops(P, M, F):
    """
    Compute FLOPs for pillar-wise feature aggregation.
    FLOPs = P * M * F
    """
    return P * M * F

def patch_embedding_flops(H, W, P, E):
    """
    Compute FLOPs for patch embedding.
    FLOPs = H * W * P^2 * E
    """
    return H * W * P * P * E

def position_embedding_flops(N, E):
    """
    Compute FLOPs for position embedding.
    FLOPs = N * E
    """
    return N * E

def transformer_block_flops(N, E, h):
    """
    Compute FLOPs for a transformer block.
    FLOPs = N * E * (4E + 2N)
    """
    return N * E * (4 * E + 2 * N)

def count_flops(module, input):
    """
    Count FLOPs for a model.
    """
    H, W = input.shape[2], input.shape[3]
    C_in = input.shape[1]
    C_out = module.weight.shape[0]
    K = module.kernel_size[0]
    if isinstance(module, nn.Conv2d):
        return conv_flops(H, W, C_in, C_out, K)
    elif isinstance(module, nn.BatchNorm2d):
        return bn_flops(H, W, C_in)
    elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
        return pool_flops(H, W, C_in, K)
    elif isinstance(module, nn.Linear):
        return pointwise_flops(1, C_in, C_out)
    else:
        return 0

def profile_model(model, input_size):
    """
    Profile the model to count the FLOPs of each module.
    """
    input = torch.randn(*input_size).cuda()
    model = model.cuda()
    total_flops = 0

    def count_flops_hook(module, input, output):
        nonlocal total_flops
        flops = profile_macs(module, input)
        total_flops += flops

    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(count_flops_hook))

    with torch.no_grad():
        model(input)

    for hook in hooks:
        hook.remove()

    return total_flops

args = arg_config()
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
#TODO here need an actual input size
input_size = (1, 3, 224, 224)
total_flops = profile_model(model, input_size)

print(f"Total FLOPs: {total_flops}")


