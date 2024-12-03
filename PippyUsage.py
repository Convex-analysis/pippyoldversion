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

class LAVLoss:
    def __init__(self):
        self.prob_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.loc_criterion = nn.L1Loss(reduction='none')
        self.ori_criterion = nn.L1Loss(reduction='none')
        self.box_criterion = nn.L1Loss(reduction='none')
        self.spd_criterion = nn.L1Loss(reduction='none')
        #self.loc_criterion = nn.SmoothL1Loss(reduction='none')
        #self.ori_criterion = nn.SmoothL1Loss(reduction='none')
        #self.box_criterion = nn.SmoothL1Loss(reduction='none')
        #self.spd_criterion = nn.SmoothL1Loss(reduction='none')

    def __call__(self, output, target):
        prob = target[:, : ,0:1]
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


class MVTL1Loss:
    def __init__(self, weight=1, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss()
        self.weight = weight

    def __call__(self, output, target):
        target_1_mask = target[:, :, 0].ge(0.01)
        target_0_mask = target[:, :, 0].le(0.01)
        target_prob_1 = torch.masked_select(target[:, :, 0], target_1_mask)
        output_prob_1 = torch.masked_select(output[:, :, 0], target_1_mask)
        target_prob_0 = torch.masked_select(target[:, :, 0], target_0_mask)
        output_prob_0 = torch.masked_select(output[:, :, 0], target_0_mask)
        if target_prob_1.numel() == 0:
            loss_prob_1 = 0
        else:
            loss_prob_1 = self.loss(output_prob_1, target_prob_1)
        if target_prob_0.numel() == 0:
            loss_prob_0 = 0
        else:
            loss_prob_0 = self.loss(output_prob_0, target_prob_0)
        loss_1 = 0.5 * loss_prob_0 + 0.5 * loss_prob_1

        output_1 = output[target_1_mask][:][:, 1:7]
        target_1 = target[target_1_mask][:][:, 1:7]
        if target_1.numel() == 0:
            loss_2 = 0
        else:
            loss_2 = self.loss(target_1, output_1)

        # speed pred loss
        output_2 = output[target_1_mask][:][:, 7]
        target_2 = target[target_1_mask][:][:, 7]
        if target_2.numel() == 0:
            loss_3 = target_2.sum() # torch.tensor([0.0]).cuda()
        else:
            loss_3 = self.loss(target_2, output_2)
        return 0.5 * loss_1 * self.weight + 0.5 * loss_2, loss_3

cls_loss = nn.CrossEntropyLoss()

train_loss_fns = {
        #"traffic": MVTL1Loss(1.0, l1_loss=l1_loss),
        "traffic": LAVLoss(),
        "waypoints": torch.nn.L1Loss(),
        "cls": cls_loss,
        "stop_cls": cls_loss,
    }

class MemFuserLoss:
    def __init__(self):
        self.loss_fns = {
            "traffic": LAVLoss(),
            "waypoints": torch.nn.L1Loss(),
            "cls": nn.CrossEntropyLoss(),
            "stop_cls": nn.CrossEntropyLoss(),
        }

    def __call__(self, output, target):
        loss_traffic, loss_velocity = self.loss_fns["traffic"](output[0], target[4])
        loss_waypoints = self.loss_fns["waypoints"](output[1], target[1])
        loss_traffic_light_state = self.loss_fns["cls"](output[2], target[3])
        loss_stop_sign = self.loss_fns["stop_cls"](output[3], target[6])
        
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
            super().__init__(module, loss_fn)

        def forward(self, input, target):
            output = self.module(input)
            return output, self.loss_fn(output, target)

  

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

loss_wrapper = ModelLossWrapper(module=model, loss_fn=MemFuserLoss())   

annotate_split_points(model, {
    'decoder': PipeSplitWrapper.SplitPoint.BEGINNING})

pipe = Pipe.from_tracing(loss_wrapper, output_loss_value_spec=True)
print(pipe)

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

