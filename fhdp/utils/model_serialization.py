"""
Model Serialization Utilities for FHDP System

Provides utilities for serializing and deserializing PyTorch model states
for cross-platform communication (Tensor ↔ List conversion for JSON transport).
"""
import torch
from typing import Dict, Any


def serialize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert torch.Tensor values to lists for JSON serialization.

    Args:
        state_dict: PyTorch model state dict containing tensors

    Returns:
        Dictionary with all Tensor values converted to lists
    """
    return {k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in state_dict.items()}


def deserialize_state_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restore lists back to torch.Tensor after JSON deserialization.

    Args:
        raw: Dictionary with lists (from JSON) representing tensors

    Returns:
        PyTorch-compatible state dict with tensors restored
    """
    return {k: torch.tensor(v) if isinstance(v, list) else v
            for k, v in raw.items()}
