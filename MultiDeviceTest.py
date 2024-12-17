import os
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import nn
from torchvision import datasets, transforms
from pippy import PipelineDriverFillDrain, PipelineDriver1F1B
#from pippy.microbatch import sum_reducer, TensorChunkSpec
#from pippy.visualizer import events_to_json
#from examples.resnet import ResNet18
import pippy.fx
PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
}

pippy.fx.Tracer.proxy_buffer_attributes = True

USE_TQDM = bool(int(os.getenv('USE_TQDM', '1')))

def run_master(_, args):
    #MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    #print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)
    print("Using device:", args.device)

    number_of_workers = 4
    all_worker_ranks = list(range(1, 1 + number_of_workers))  # exclude master rank = 0
    chunks = len(all_worker_ranks)
    batch_size = args.batch_size * chunks

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10('./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replicate', action='store_true')
    parser.add_argument('--schedule', type=str, default='FillDrain')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    run_master(0, args)