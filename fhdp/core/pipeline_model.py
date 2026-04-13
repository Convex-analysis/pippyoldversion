"""
Core model split and template registry utilities for pipeline execution.
"""

from typing import Dict, Any, Optional, Tuple, Callable

import torch
import torch.nn as nn
import timm

from .types import PipelineTemplate, TrainingConfig, ResourceClass

ModelSplitFn = Callable[[int], Tuple[nn.Module, nn.Module]]

DEFAULT_TEMPLATE_BATCH_SIZE = 16


def _split_resnet18_2stage_v1(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    stage0 = nn.Sequential(
        model.conv1, model.bn1, model.act1, model.maxpool,
        model.layer1, model.layer2
    )
    stage1 = nn.Sequential(
        model.layer3, model.layer4, model.global_pool, model.fc
    )
    return stage0, stage1


def _split_resnet18_2stage_v2(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    stage0 = nn.Sequential(
        model.conv1, model.bn1, model.act1, model.maxpool,
        model.layer1
    )
    stage1 = nn.Sequential(
        model.layer2, model.layer3, model.layer4, model.global_pool, model.fc
    )
    return stage0, stage1


class _ViTStage0(nn.Module):
    def __init__(self, model: nn.Module, split_idx: int):
        super().__init__()
        self.patch_embed = model.patch_embed
        self.cls_token = getattr(model, "cls_token", None)
        self.pos_embed = getattr(model, "pos_embed", None)
        self.pos_drop = model.pos_drop
        self.blocks = nn.ModuleList(model.blocks[:split_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class _ViTStage1(nn.Module):
    def __init__(self, model: nn.Module, split_idx: int):
        super().__init__()
        self.blocks = nn.ModuleList(model.blocks[split_idx:])
        self.norm = model.norm
        self.pre_logits = getattr(model, "pre_logits", nn.Identity())
        self.head = model.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        return x


def _split_vit_b16_2stage_v1(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    split_idx = 6
    stage0 = _ViTStage0(model, split_idx)
    stage1 = _ViTStage1(model, split_idx)
    return stage0, stage1


def build_vit_b16_split_by_ratio(
    num_classes: int,
    ratio: float
) -> Tuple[int, nn.Module, nn.Module]:
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    total_blocks = len(model.blocks)
    split_idx = int(round(total_blocks * ratio))
    split_idx = max(1, min(total_blocks - 1, split_idx))
    stage0 = _ViTStage0(model, split_idx)
    stage1 = _ViTStage1(model, split_idx)
    return split_idx, stage0, stage1


MODEL_SPLIT_REGISTRY: Dict[str, ModelSplitFn] = {
    "resnet18_2stage_v1": _split_resnet18_2stage_v1,
    "resnet18_2stage_v2": _split_resnet18_2stage_v2,
    "vit_b16_2stage_v1": _split_vit_b16_2stage_v1,
}

PIPELINE_TEMPLATE_REGISTRY: Dict[str, PipelineTemplate] = {
    "resnet18_2stage_v1": PipelineTemplate(
        template_id="resnet18_2stage_v1",
        resource_requirements=[ResourceClass.HIGH, ResourceClass.MEDIUM],
        expected_duration=10.0,
        communication_pattern=[(0, 1)],
        training_config=TrainingConfig(
            epochs=1,
            batch_size=DEFAULT_TEMPLATE_BATCH_SIZE,
            learning_rate=0.01,
            local_data_size=DEFAULT_TEMPLATE_BATCH_SIZE
        ),
        model_fragment_size=0,
        model_partition={
            "model_name": "resnet18",
            "split_key": "resnet18_2stage_v1",
            "stages": ["stage0", "stage1"]
        }
    ),
    "resnet18_2stage_v2": PipelineTemplate(
        template_id="resnet18_2stage_v2",
        resource_requirements=[ResourceClass.HIGH, ResourceClass.MEDIUM],
        expected_duration=10.0,
        communication_pattern=[(0, 1)],
        training_config=TrainingConfig(
            epochs=1,
            batch_size=DEFAULT_TEMPLATE_BATCH_SIZE,
            learning_rate=0.01,
            local_data_size=DEFAULT_TEMPLATE_BATCH_SIZE
        ),
        model_fragment_size=0,
        model_partition={
            "model_name": "resnet18",
            "split_key": "resnet18_2stage_v2",
            "stages": ["stage0", "stage1"]
        }
    ),
    "vit_b16_2stage_v1": PipelineTemplate(
        template_id="vit_b16_2stage_v1",
        resource_requirements=[ResourceClass.HIGH, ResourceClass.MEDIUM],
        expected_duration=20.0,
        communication_pattern=[(0, 1)],
        training_config=TrainingConfig(
            epochs=1,
            batch_size=DEFAULT_TEMPLATE_BATCH_SIZE,
            learning_rate=0.01,
            local_data_size=DEFAULT_TEMPLATE_BATCH_SIZE
        ),
        model_fragment_size=0,
        model_partition={
            "model_name": "vit_base_patch16_224",
            "split_key": "vit_b16_2stage_v1",
            "stages": ["stage0", "stage1"]
        }
    ),
}


def get_pipeline_template(template_id: str, default_template_id: str) -> PipelineTemplate:
    if template_id in PIPELINE_TEMPLATE_REGISTRY:
        return PIPELINE_TEMPLATE_REGISTRY[template_id]
    return PIPELINE_TEMPLATE_REGISTRY[default_template_id]


def serialize_template(template: PipelineTemplate) -> Dict[str, Any]:
    config = template.training_config
    return {
        "template_id": template.template_id,
        "resource_requirements": [r.value for r in template.resource_requirements],
        "expected_duration": template.expected_duration,
        "communication_pattern": template.communication_pattern,
        "training_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "local_data_size": config.local_data_size,
            "model_size": config.model_size,
            "communication_budget": config.communication_budget
        },
        "model_fragment_size": template.model_fragment_size,
        "model_partition": template.model_partition or {}
    }


def resolve_split_key(split_key: Optional[str], default_split_key: str) -> str:
    if split_key in MODEL_SPLIT_REGISTRY:
        return split_key
    return default_split_key


def build_model_split(
    num_classes: int,
    split_key: Optional[str],
    default_split_key: str
) -> Tuple[str, nn.Module, nn.Module]:
    resolved_key = resolve_split_key(split_key, default_split_key)
    split_fn = MODEL_SPLIT_REGISTRY[resolved_key]
    stage0, stage1 = split_fn(num_classes)
    return resolved_key, stage0, stage1


def build_model_split_from_template_payload(
    template_payload: Dict[str, Any],
    default_template_id: str,
    num_classes: int
) -> Tuple[str, str, nn.Module, nn.Module]:
    template_id = template_payload.get("template_id") or default_template_id
    model_partition = template_payload.get("model_partition") or {}
    split_key = model_partition.get("split_key") or template_id
    resolved_key, stage0, stage1 = build_model_split(num_classes, split_key, default_template_id)
    return template_id, resolved_key, stage0, stage1
