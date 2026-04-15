"""
Core model split and template registry utilities for pipeline execution.
"""

from typing import Dict, Any, Optional, Tuple, Callable, List

import itertools
from dataclasses import dataclass
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


def _split_vit_b16_2stage_light(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    """Split ViT at block 4 - lighter first stage for medium resources"""
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    split_idx = 4
    stage0 = _ViTStage0(model, split_idx)
    stage1 = _ViTStage1(model, split_idx)
    return stage0, stage1


def _split_vit_b16_2stage_medium(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    """Split ViT at block 6 - balanced split"""
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    split_idx = 6
    stage0 = _ViTStage0(model, split_idx)
    stage1 = _ViTStage1(model, split_idx)
    return stage0, stage1


def _split_vit_b16_2stage_heavy(num_classes: int) -> Tuple[nn.Module, nn.Module]:
    """Split ViT at block 8 - heavier first stage for high resources"""
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    split_idx = 9
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
    "vit_b16_2stage_light": _split_vit_b16_2stage_light,
    "vit_b16_2stage_medium": _split_vit_b16_2stage_medium,
    "vit_b16_2stage_v1": _split_vit_b16_2stage_heavy,  # Legacy alias (prefer heavier stage0)
}

PIPELINE_TEMPLATE_REGISTRY: Dict[str, PipelineTemplate] = {
    # ResNet18 templates with different resource requirements
    "resnet18_2stage_high_high": PipelineTemplate(
        template_id="resnet18_2stage_high_high",
        resource_requirements=[ResourceClass.HIGH, ResourceClass.HIGH],
        expected_duration=8.0,
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
            "stages": ["stage0", "stage1"],
            "description": "ResNet18 split at layer2, both stages need high resources"
        }
    ),
    "resnet18_2stage_high_medium": PipelineTemplate(
        template_id="resnet18_2stage_high_medium",
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
            "stages": ["stage0", "stage1"],
            "description": "ResNet18 split at layer2, stage0 high, stage1 medium"
        }
    ),
    "resnet18_2stage_medium_high": PipelineTemplate(
        template_id="resnet18_2stage_medium_high",
        resource_requirements=[ResourceClass.MEDIUM, ResourceClass.HIGH],
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
            "stages": ["stage0", "stage1"],
            "description": "ResNet18 split at layer1, stage0 medium, stage1 high"
        }
    ),
    "resnet18_2stage_medium_medium": PipelineTemplate(
        template_id="resnet18_2stage_medium_medium",
        resource_requirements=[ResourceClass.MEDIUM, ResourceClass.MEDIUM],
        expected_duration=12.0,
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
            "stages": ["stage0", "stage1"],
            "description": "ResNet18 split at layer1, both stages medium resources"
        }
    ),
    # Legacy aliases for backward compatibility
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
        resource_requirements=[ResourceClass.MEDIUM, ResourceClass.HIGH],
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
    # ViT templates with different resource requirements
    "vit_b16_2stage_high_high": PipelineTemplate(
        template_id="vit_b16_2stage_high_high",
        resource_requirements=[ResourceClass.HIGH, ResourceClass.HIGH],
        expected_duration=15.0,
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
            "split_key": "vit_b16_2stage_heavy",
            "stages": ["stage0", "stage1"],
            "description": "ViT split at block 8, both stages need high resources",
            "resource_estimates": {
                "stage0": {"memory_gb": 4.5, "compute_score": 0.85},
                "stage1": {"memory_gb": 3.5, "compute_score": 0.75}
            }
        }
    ),
    "vit_b16_2stage_high_medium": PipelineTemplate(
        template_id="vit_b16_2stage_high_medium",
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
            "split_key": "vit_b16_2stage_medium",
            "stages": ["stage0", "stage1"],
            "description": "ViT split at block 6, stage0 high, stage1 medium",
            "resource_estimates": {
                "stage0": {"memory_gb": 3.5, "compute_score": 0.75},
                "stage1": {"memory_gb": 4.5, "compute_score": 0.85}
            }
        }
    ),
    "vit_b16_2stage_medium_high": PipelineTemplate(
        template_id="vit_b16_2stage_medium_high",
        resource_requirements=[ResourceClass.MEDIUM, ResourceClass.HIGH],
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
            "split_key": "vit_b16_2stage_light",
            "stages": ["stage0", "stage1"],
            "description": "ViT split at block 4, stage0 medium, stage1 high",
            "resource_estimates": {
                "stage0": {"memory_gb": 2.5, "compute_score": 0.65},
                "stage1": {"memory_gb": 5.5, "compute_score": 0.95}
            }
        }
    ),
    # Legacy alias
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
            "split_key": "vit_b16_2stage_heavy",
            "stages": ["stage0", "stage1"],
            "description": "ViT split at block 8, stage0 high, stage1 medium",
            "resource_estimates": {
                "stage0": {"memory_gb": 4.5, "compute_score": 0.85},
                "stage1": {"memory_gb": 3.5, "compute_score": 0.75}
            }
        }
    ),
}


def _estimate_stage_resources(stage_param_count: int, total_param_count: int) -> Tuple[float, float, ResourceClass]:
    if total_param_count <= 0:
        return 0.5, 0.1, ResourceClass.LOW
    param_bytes = stage_param_count * 4
    memory_gb = max(0.5, (param_bytes * 6) / (1024 ** 3))
    compute_score = min(1.0, max(0.05, stage_param_count / total_param_count))
    if compute_score >= 0.45 or memory_gb >= 8.0:
        resource_class = ResourceClass.HIGH
    elif compute_score >= 0.25 or memory_gb >= 4.0:
        resource_class = ResourceClass.MEDIUM
    else:
        resource_class = ResourceClass.LOW
    return memory_gb, compute_score, resource_class


def _build_resnet18_blocks(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [
        ("stem", nn.Sequential(model.conv1, model.bn1, model.act1, model.maxpool)),
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("layer4", model.layer4),
        ("head", nn.Sequential(model.global_pool, model.fc)),
    ]


def _build_vit_b16_blocks(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [
        ("embed", nn.Sequential(model.patch_embed, model.pos_drop)),
        ("blocks_0_3", nn.ModuleList(model.blocks[:4])),
        ("blocks_4_7", nn.ModuleList(model.blocks[4:8])),
        ("blocks_8_11", nn.ModuleList(model.blocks[8:])),
        ("head", nn.Sequential(model.norm, model.head)),
    ]


def _block_param_count(block: nn.Module) -> int:
    return sum(p.numel() for p in block.parameters())


def _enumerate_partitions(
    n_blocks: int,
    stages: int,
    shallow_end_index: int,
    max_imbalance: float = 0.2
) -> List[List[Tuple[int, int]]]:
    if stages < 2 or stages > n_blocks:
        return []
    partitions = []
    cut_min = max(1, shallow_end_index + 1)
    for cuts in itertools.combinations(range(1, n_blocks), stages - 1):
        if cuts[0] < cut_min:
            continue
        indices = (0,) + cuts + (n_blocks,)
        ranges = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
        partitions.append(ranges)
    return partitions


def _stage_costs(block_costs: List[int], ranges: List[Tuple[int, int]]) -> List[int]:
    return [sum(block_costs[start:end]) for start, end in ranges]


def _is_balanced(costs: List[int], max_imbalance: float = 0.2) -> bool:
    if not costs or min(costs) == 0:
        return False
    return (max(costs) / min(costs)) <= (1.0 + max_imbalance)


def _infer_default_split_key(model_name: str) -> str:
    if model_name == "resnet18":
        return "resnet18_2stage_v1"
    if model_name == "vit_base_patch16_224":
        return "vit_b16_2stage_v1"
    return "resnet18_2stage_v1"


def _infer_split_key_for_cut(model_name: str, cut_index: int) -> Optional[str]:
    if model_name == "resnet18":
        if cut_index == 2:
            return "resnet18_2stage_v2"
        if cut_index == 3:
            return "resnet18_2stage_v1"
    if model_name == "vit_base_patch16_224":
        if cut_index == 2:
            return "vit_b16_2stage_light"
        if cut_index == 3:
            return "vit_b16_2stage_heavy"
    return None


@dataclass(frozen=True)
class FunctionalModelSpec:
    model_name: str
    blocks_fn: Callable[[nn.Module], List[Tuple[str, nn.Module]]]
    shallow_end: int
    max_stages: int


def generate_functional_block_templates() -> Dict[str, PipelineTemplate]:
    templates: Dict[str, PipelineTemplate] = {}
    model_specs = [
        FunctionalModelSpec(
            model_name="resnet18",
            blocks_fn=_build_resnet18_blocks,
            shallow_end=1,
            max_stages=4,
        ),
        FunctionalModelSpec(
            model_name="vit_base_patch16_224",
            blocks_fn=_build_vit_b16_blocks,
            shallow_end=1,
            max_stages=4,
        )
    ]
    for spec in model_specs:
        model_name = spec.model_name
        model = timm.create_model(model_name, pretrained=False, num_classes=1000)
        blocks = spec.blocks_fn(model)
        block_names = [name for name, _ in blocks]
        block_costs = [_block_param_count(block) for _, block in blocks]
        total_cost = sum(block_costs)
        n_blocks = len(blocks)
        max_stages = min(spec.max_stages, n_blocks)
        for stages in range(2, max_stages + 1):
            partitions = _enumerate_partitions(n_blocks, stages, spec.shallow_end)
            for ranges in partitions:
                costs = _stage_costs(block_costs, ranges)
                if not _is_balanced(costs):
                    continue
                stage_blocks = [block_names[start:end] for start, end in ranges]
                stage_resource = []
                resource_estimates = {}
                for i, stage_cost in enumerate(costs):
                    mem_gb, compute_score, resource_class = _estimate_stage_resources(stage_cost, total_cost)
                    stage_resource.append(resource_class)
                    resource_estimates[f"stage{i}"] = {
                        "memory_gb": round(mem_gb, 2),
                        "compute_score": round(compute_score, 2),
                        "block_range": [ranges[i][0], ranges[i][1]]
                    }
                cut_index = ranges[0][1]
                split_key = None
                if stages == 2:
                    split_key = _infer_split_key_for_cut(model_name, cut_index)
                    if split_key is None:
                        split_key = _infer_default_split_key(model_name)
                template_id = f"{model_name}_func_s{stages}_c" + "-".join(str(r[1]) for r in ranges[:-1])
                templates[template_id] = PipelineTemplate(
                    template_id=template_id,
                    resource_requirements=stage_resource,
                    expected_duration=12.0 + stages * 2.0,
                    communication_pattern=[(i, i + 1) for i in range(stages - 1)],
                    training_config=TrainingConfig(
                        epochs=1,
                        batch_size=DEFAULT_TEMPLATE_BATCH_SIZE,
                        learning_rate=0.01,
                        local_data_size=DEFAULT_TEMPLATE_BATCH_SIZE
                    ),
                    model_fragment_size=0,
                    model_partition={
                        "model_name": model_name,
                        "split_key": split_key,
                        "stages": [f"stage{i}" for i in range(stages)],
                        "block_partitions": stage_blocks,
                        "resource_estimates": resource_estimates,
                        "description": f"Functional block split with {stages} stages"
                    }
                )
    return templates


PIPELINE_TEMPLATE_REGISTRY.update(generate_functional_block_templates())


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
