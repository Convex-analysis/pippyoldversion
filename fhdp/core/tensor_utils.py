"""
Tensor transfer utilities for FHDP pipeline training.

Provides tensor packing, unpacking, and transfer optimizations for pipeline communication.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch


PACK_TENSOR_FASTPATH = True
PACK_TENSOR_USE_PINNED = False
PACK_TENSOR_ZERO_COPY = False
PINNED_BUFFER_CACHE_SIZE = 8
PINNED_BUFFER_MAX_BYTES = 256 * 1024 * 1024


def to_tensor(value: Any, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Convert value to tensor with specified device and dtype"""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if device:
            tensor = tensor.to(device, non_blocking=True)
        return tensor
    return torch.as_tensor(value, dtype=dtype).to(device, non_blocking=True)


def pack_tensor_for_send(
    value: Any,
    serialization_manager: Any = None
) -> Any:
    """Pack tensor for sending over network"""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()

        if PACK_TENSOR_FASTPATH and serialization_manager is not None:
            if tensor.device.type == "cpu" and tensor.is_contiguous():
                try:
                    return {
                        "__tensor_zero_copy__": True,
                        "data": serialization_manager.serialize_tensor_zero_copy(tensor),
                        "shape": tuple(tensor.shape),
                        "dtype": str(tensor.dtype)
                    }
                except Exception:
                    pass

            if tensor.device.type == "cuda":
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                try:
                    return {
                        "__tensor_zero_copy__": True,
                        "data": serialization_manager.serialize_tensor_zero_copy(tensor),
                        "shape": tuple(tensor.shape),
                        "dtype": str(tensor.dtype)
                    }
                except Exception:
                    pass

        cpu_tensor = tensor.to("cpu", non_blocking=True)
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()

        if serialization_manager is not None:
            try:
                return {
                    "__tensor_zero_copy__": True,
                    "data": serialization_manager.serialize_tensor_zero_copy(cpu_tensor),
                    "shape": tuple(cpu_tensor.shape),
                    "dtype": str(cpu_tensor.dtype)
                }
            except Exception:
                pass

        return cpu_tensor

    if isinstance(value, dict):
        return {k: pack_tensor_for_send(v, serialization_manager) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        packed = [pack_tensor_for_send(v, serialization_manager) for v in value]
        return tuple(packed) if isinstance(value, tuple) else packed

    return value


def unpack_tensor_payload(
    value: Any,
    serialization_manager: Any = None
) -> Any:
    """Unpack tensor from received network data"""
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if serialization_manager is not None:
            try:
                result = serialization_manager.deserialize_tensor_zero_copy(data)
                if result is not None:
                    return result
            except Exception:
                pass

            try:
                return serialization_manager.pickle_loads(data)
            except Exception:
                return value
        return value

    if isinstance(value, dict):
        if value.get("__tensor_zero_copy__") and "data" in value:
            data = value.get("data")
            if data is None:
                return value
            if serialization_manager is not None:
                try:
                    result = serialization_manager.deserialize_tensor_zero_copy(data)
                    if result is not None:
                        return result
                except Exception:
                    pass

                try:
                    return serialization_manager.pickle_loads(bytes(data))
                except Exception:
                    return data
            return data

        if value.get("__tensor_metadata__") and "data" in value:
            return value["data"]

        if "data" in value and value.get("__tensor_bytes__"):
            return value["data"]

    return value


def decode_tensor_payload(
    value: Any,
    serialization_manager: Any = None
) -> Any:
    """Decode tensor payload from received data"""
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if serialization_manager is not None:
            try:
                result = serialization_manager.deserialize_tensor_zero_copy(data)
                if result is not None:
                    return result
            except Exception:
                pass

            try:
                return serialization_manager.pickle_loads(data)
            except Exception:
                return value
    return value


def copy_tensor_to_cpu(
    tensor: torch.Tensor,
    use_pinned: bool = PACK_TENSOR_USE_PINNED
) -> torch.Tensor:
    """Copy tensor from GPU to CPU with optional pinned memory optimization"""
    if tensor.device.type == "cpu":
        return tensor

    src = tensor.detach()
    if not src.is_contiguous():
        src = src.contiguous()

    if use_pinned and torch.cuda.is_available():
        try:
            shape_tuple = tuple(src.shape)
            elem_size = src.element_size()
            numel = src.numel()
            bytes_needed = numel * elem_size

            if bytes_needed <= PINNED_BUFFER_MAX_BYTES:
                buffer = torch.empty(shape_tuple, dtype=src.dtype, device="cpu", pin_memory=True)
                buffer.copy_(src, non_blocking=True)
                torch.cuda.current_stream().synchronize()
                return buffer
        except Exception:
            pass

    return src.to("cpu", non_blocking=True).contiguous()


def async_transfer_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Asynchronously transfer tensor from GPU to CPU using CUDA stream"""
    if not tensor.is_cuda:
        return tensor

    try:
        with torch.cuda.stream(torch.cuda.current_stream()):
            cpu_tensor = tensor.detach().contiguous().to("cpu", non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return cpu_tensor
    except Exception:
        return tensor.detach().contiguous().to("cpu", non_blocking=True)


def get_pinned_buffer(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    cache: Optional[dict] = None
) -> Optional[torch.Tensor]:
    """Get pinned memory buffer from cache for fast GPU-CPU transfer"""
    if not PACK_TENSOR_USE_PINNED or not torch.cuda.is_available():
        return None

    try:
        shape_tuple = tuple(shape) if shape is not None else ()
        numel = 1
        for dim in shape_tuple:
            numel *= int(dim)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        bytes_needed = numel * elem_size

        if bytes_needed > PINNED_BUFFER_MAX_BYTES:
            return None

        key = (dtype, shape_tuple)
        if cache is not None:
            buffer = cache.pop(key, None)
            if buffer is not None and buffer.numel() == numel and buffer.dtype == dtype:
                cache[key] = buffer
                return buffer

        buffer = torch.empty(shape_tuple, dtype=dtype, device="cpu", pin_memory=True)
        return buffer
    except Exception:
        return None


def estimate_tensor_bytes(value: Any) -> int:
    """Estimate byte size of a tensor or container"""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()

    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)

    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes

    if isinstance(value, dict):
        return sum(estimate_tensor_bytes(v) for v in value.values())

    if isinstance(value, (list, tuple)):
        return sum(estimate_tensor_bytes(v) for v in value)

    return 0