import logging
import time
import os
import sys
import argparse
import csv
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

from pippy.PipelineDriver import print_blue, print_red, print_green

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
    memory_mb = torch.cuda.memory_allocated(0)/1024/1024
    print("GPU {}:{}".format(stage, memory_mb))
    return memory_mb

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

    # Print out the full model structure to find valid split points
    print("Full model structure (all modules):")
    all_modules = {}
    for name, module in model.named_modules():
        if name:  # Skip the root module
            all_modules[name] = type(module).__name__
            print(f"Module: {name} - {type(module).__name__}")
    
    # Find all decoder-related modules for potential split points
    decoder_modules = {name: module_type for name, module_type in all_modules.items() 
                      if 'decoder' in name.lower()}
    print("\nDecoder-related modules:")
    for name, module_type in decoder_modules.items():
        print(f"  {name}: {module_type}")
    
    # Approach 1: Use only top-level module split points that don't conflict
    # This avoids the issue where we try to split a submodule of an already wrapped module
    annotate_split_points(model, {
            #'encoder': PipeSplitWrapper.SplitPoint.BEGINNING,
            'decoder': PipeSplitWrapper.SplitPoint.BEGINNING
        })
    
    print("Using split points:", split_points)
    annotate_split_points(model, split_points)
    
    # Approach 2 (alternative): Split the model into 5 stages manually by directly accessing submodules
    # This would require more knowledge of the specific model structure
    # We'd need to implement a custom splitting logic here based on the actual model
    
    output_loss_value_spec = (False, True)
    pipe = Pipe.from_tracing(loss_wrapper, output_loss_value_spec=output_loss_value_spec)
    debug_pickle(pipe, "pipe")
    
    # Count the number of stages in the generated pipe to confirm our splits worked
    print(f"Number of stages in the pipe: {len(pipe.split_gm.submodules)}")
    print(pipe)
    
    #start to initialize the pipeline driver
    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}

    output_chunk_spec = {
        'output': (TensorChunkSpec(0), TensorChunkSpec(0), TensorChunkSpec(0), TensorChunkSpec(0), TensorChunkSpec(0)),
        'loss': LossReducer(0.0, loss_reducer_fn)
    }


    world_size = 5
    
    # Define chunks - setting to world_size as a reasonable default
    chunks = 5

    driver = PipelineDriverFillDrain(
            pipe,
            chunks,  # Add the missing chunks parameter
            world_size=world_size,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec)
    debug_pickle(driver, "driver")
    return driver

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
    print(pipelineDriver)
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

if __name__ == "__main__":
    main()