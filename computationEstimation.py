import torch
import torch.nn as nn
from torchprofile import profile_macs
from timm.models.memfuser import memfuser_baseline

def count_flops(model, input_size):
    flops = {}
    hooks = []

    def hook_fn(module, input, output):
        module_name = str(module.__class__).split("'")[1]
        if module_name not in flops:
            flops[module_name] = 0
        flops[module_name] += profile_macs(module, input)

    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook_fn))

    dummy_input = {
        "rgb_front": torch.randn(1, 3, 224, 224),
        "rgb_left": torch.randn(1, 3, 224, 224),
        "rgb_right": torch.randn(1, 3, 224, 224),
        "rgb_rear": torch.randn(1, 3, 224, 224),
        "rgb_center": torch.randn(1, 3, 224, 224),
        "lidar": torch.randn(1, 40000, 4),
        "num_points": torch.tensor([40000]),
        "velocity": torch.randn(1, 1),
        "target_point": torch.randn(1, 2)
    }
    model(dummy_input)

    for hook in hooks:
        hook.remove()

    return flops

# Example usage:
# model = memfuser_baseline()
# flops = count_flops(model, (3, 224, 224))
# print(f"Model FLOPS: {flops}")
