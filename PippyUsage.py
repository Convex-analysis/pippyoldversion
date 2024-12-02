import torch
from pippy.IR import Pipe, annotate_split_points, PipeSplitWrapper
from pippy.IR import LossWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import (
    create_model
)
from timm.utils import *


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

annotate_split_points(model, {
    'position_encoding': PipeSplitWrapper.SplitPoint.END,
    'decoder': PipeSplitWrapper.SplitPoint.BEGINNING})
pipe = Pipe.from_tracing(model)
print(pipe)
