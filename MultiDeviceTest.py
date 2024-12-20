import os
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import nn
from torchvision import datasets, transforms
from pippy import PipelineDriverFillDrain, PipelineDriver1F1B
import argparse
import os
import sys
from functools import reduce
import psutil

import torch
from torch import optim
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms  # type: ignore
from tqdm import tqdm  # type: ignoreimport pippy.fx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe, LossWrapper, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.events import EventsContext
from pippy.microbatch import sum_reducer, TensorChunkSpec
from pippy.visualizer import events_to_json
import pippy.fx
from examples.resnet import ResNet34
PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
}

pippy.fx.Tracer.proxy_buffer_attributes = True

USE_TQDM = bool(int(os.getenv('USE_TQDM', '1')))

def run_master(_, args):
    class OutputLossWrapper(LossWrapper):
            def __init__(self, module, loss_fn):
                super().__init__(module, loss_fn)

            @torch.autocast(device_type="cuda")
            def forward(self, input, target):
                output = self.module(input)
                loss = self.loss_fn(output, target)
                # Here we use a dict with the "loss" keyword so that PiPPy can automatically find the loss field when
                # generating the backward pass
                return {"output": output, "loss": loss}
    model = ResNet34()

    annotate_split_points(model, {
            #'layer1': PipeSplitWrapper.SplitPoint.END,
            'layer2.conv1': PipeSplitWrapper.SplitPoint.END,
            #'layer3': PipeSplitWrapper.SplitPoint.END,
        })

    wrapper = OutputLossWrapper(model, cross_entropy)
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)
    print("Using device:", args.device)
    pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG)
    pipe.to(args.device)
    print(pipe)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replicate', action='store_true')
    parser.add_argument('--schedule', type=str, default='FillDrain')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run_master(0, args)