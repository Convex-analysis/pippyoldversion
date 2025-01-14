import logging
import time
import os
import sys
import argparse
from typing import OrderedDict
import torch
#import resource
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

def debug_pickle(obj, name):
    try:
        pickle.dumps(obj)
    except TypeError as e:
        print(f"Error pickling {name}: {e}")



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

class PipeWrapper:
    def __init__(self, pipe):
        self.pipe = pipe

    def to_pycapsule(self):
        # Convert the Pipe object to a PyCapsule
        capsule = ctypes.py_object(self.pipe)
        return capsule

    @staticmethod
    def from_pycapsule(capsule):
        # Convert the PyCapsule back to a Pipe object
        pipe = ctypes.cast(capsule, ctypes.py_object).value
        return PipeWrapper(pipe)

    def serialize(self):
        # Serialize the Pipe object using pickle
        # Convert the PyCapsule to a serializable format
        capsule = self.to_pycapsule()
        capsule_address = ctypes.addressof(ctypes.c_void_p.from_buffer(capsule))
        serialized_pipe = pickle.dumps(capsule_address)
        return serialized_pipe

    @staticmethod
    def deserialize(serialized_pipe):
        # Deserialize the Pipe object using pickle
        # Convert the serialized format back to a PyCapsule
        capsule_address = pickle.loads(serialized_pipe)
        capsule = ctypes.cast(ctypes.c_void_p(capsule_address), ctypes.py_object)
        pipe = ctypes.cast(capsule, ctypes.py_object).value
        return PipeWrapper(pipe)

def initialize_pipeline(model):
    

    annotate_split_points(model, {
        'encoder': PipeSplitWrapper.SplitPoint.BEGINNING,
        'decoder': PipeSplitWrapper.SplitPoint.BEGINNING
    })
    loss_wrapper = ModelLossWrapper(module=model, loss_fn=MemFuserLoss())
    debug_pickle(loss_wrapper, "loss_wrapper")
    output_loss_value_spec = (False, True)
    
    pipe = Pipe.from_tracing(loss_wrapper, output_loss_value_spec=output_loss_value_spec)
    
    print(F"Pipe: {pipe}")


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
    #Take a Memfuser instance, wrap it in a try-except pickle.dumps block, and see if any TypeError is raised.
    try:
        pickle.dumps(model)
    except TypeError as e:
        print(f"Error pickling model: {e}")
    debug_pickle(model, "model")
    initialize_pipeline(model)
   


if __name__ == "__main__":
    main()
