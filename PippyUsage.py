import torch
from pippy.IR import Pipe, annotate_split_points, PipeSplitWrapper
from pippy.IR import LossWrapper
from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec
from pippy.microbatch import LossReducer

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import (
    create_model
)
from timm.utils import *

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
print(" The following model is not warpped with a loss function ".center(80, "+"))
pipe = Pipe.from_tracing(model)
print(pipe)
print(" The following model is warpped with a loss function ".center(80, "*"))
loss_wrapper = ModelLossWrapper(module=model, loss_fn=MemFuserLoss())

annotate_split_points(model, {
    'decoder': PipeSplitWrapper.SplitPoint.BEGINNING
})
output_loss_value_spec = (False, True)
pipe = Pipe.from_tracing(loss_wrapper, output_loss_value_spec=output_loss_value_spec)
print(pipe)

'''
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
'''

