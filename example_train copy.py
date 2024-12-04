# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from typing import Any
import torch.nn as nn

class MyNetworkBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


class MyNetwork(torch.nn.Module):
    def __init__(self, in_dim, layer_dims):
        super().__init__()

        prev_dim = in_dim
        for i, dim in enumerate(layer_dims):
            setattr(self, f"layer{i}", MyNetworkBlock(prev_dim, dim))
            prev_dim = dim

        self.num_layers = len(layer_dims)
        # 10 output classes
        self.output_proj = torch.nn.Linear(layer_dims[-1], 10)

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f"layer{i}")(x)

        return self.output_proj(x)


mn = MyNetwork(512, [512, 1024, 256])

from pippy.IR import Pipe

pipe = Pipe.from_tracing(mn)
print(pipe)
print(pipe.split_gm.submod_0)


from pippy.IR import annotate_split_points, PipeSplitWrapper

annotate_split_points(
    mn,
    {
        "layer0": PipeSplitWrapper.SplitPoint.END,
        "layer1": PipeSplitWrapper.SplitPoint.END,
    },
)

pipe = Pipe.from_tracing(mn)
print(" pipe ".center(80, "*"))
print(pipe)
print(" submod0 ".center(80, "*"))
print(pipe.split_gm.submod_0)
print(" submod1 ".center(80, "*"))
print(pipe.split_gm.submod_1)
print(" submod2 ".center(80, "*"))
print(pipe.split_gm.submod_2)


# To run a distributed training job, we must launch the script in multiple
# different processes. We are using `torchrun` to do so in this example.
# `torchrun` defines two environment variables: `LOCAL_RANK` and `WORLD_SIZE`,
# which represent the index of this process within the set of processes and
# the total number of processes, respectively.
#
# To learn more about `torchrun`, see
# https://pytorch.org/docs/stable/elastic/run.html


# PiPPy relies on the concept of a "driver" process. The driver process
# should be a single process within the RPC group that instantiates the
# PipelineDriver and issues commands on that object. The other processes
# in the RPC group will receive commands from this process and execute
# the pipeline stages

from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec

    # LossWrapper is a convenient base class you can use to compose your model
    # with the desired loss function for the purpose of pipeline parallel training.
    # Since the loss is executed as part of the pipeline, it cannot reside in the
    # training loop, so you must embed it like this
from pippy.IR import LossWrapper

class ModelLossWrapper(LossWrapper):
     def forward(self, x, target):
        return self.loss_fn(self.module(x), target)

    # TODO: mean reduction
class LAVLoss:
    def __init__(self):
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
    
loss_wrapper = ModelLossWrapper(
    module=mn, loss_fn=torch.nn.MSELoss(reduction="sum")
)

    # Instantiate the `Pipe` similarly to before, but with two differences:
    #   1) We pass in the `loss_wrapper` module to include the loss in the
    #      computation
    #   2) We specify `output_loss_value_spec`. This is a data structure
    #      that should mimic the structure of the output of LossWrapper
    #      and has a True value in the position where the loss value will
    #      be. Since LossWrapper returns just the loss, we just pass True
pipe = Pipe.from_tracing(loss_wrapper, output_loss_value_spec=True)
print(pipe)    
   