# PiPPy Wireless C10d Utilities

This document describes how to use PiPPy's wireless C10d utilities for optimizing communication between devices in wireless networks, particularly for NVIDIA Jetson devices.

## Overview

PiPPy now includes support for optimizing communication over wireless networks using PyTorch's C10d library. These optimizations include:

1. **Tensor Compression**: Reduces bandwidth requirements by compressing tensors before transmission
2. **Reliable Communication**: Adds retry mechanisms for handling packet loss in wireless networks
3. **Optimized Process Groups**: Creates process groups optimized for Jetson devices
4. **Extended Timeouts**: Configures longer timeouts for wireless communication

## Usage

### Basic Setup

To enable wireless C10d optimizations, set the `use_c10d` parameter to `True` when calling `run_pippy`:

```python
from pippy.utils import run_pippy

run_pippy(
    run_worker_function,
    world_size=4,
    use_c10d=True,
    c10d_timeout_min=30  # Set timeout in minutes
)
```

### Tensor Compression

To enable tensor compression, set the `compress_tensors` parameter to `True` when creating a PipelineDriver:

```python
from pippy import PipelineDriver

driver = PipelineDriver.PipelineDriver1F1B(
    pipe=pipe,
    chunks=8,
    world_size=world_size,
    use_c10d=True,
    compress_tensors=True,
    compression_bits=8,  # Use 8-bit or 16-bit compression
    wireless_retry_count=3  # Number of retries for failed communications
)
```

### Reliable Communication

The wireless utilities include functions for reliable communication with automatic retries:

```python
from pippy.wireless_utils import reliable_broadcast, reliable_all_reduce

# Broadcast with retries
reliable_broadcast(tensor, src=0, max_retries=3)

# All-reduce with retries
reliable_all_reduce(tensor, op=dist.ReduceOp.SUM, max_retries=3)
```

### Optimized Process Groups

Create process groups optimized for Jetson devices:

```python
from pippy.wireless_utils import create_jetson_optimized_groups

dp_groups, pp_groups, dp_ranks, pp_ranks = create_jetson_optimized_groups(
    world_size=4,
    pp_group_size=2,
    dp_group_size=2
)
```

## Example

See the `examples/wireless_c10d_example.py` script for a complete example of using wireless C10d optimizations.

To run the example:

```bash
python examples/wireless_c10d_example.py --world_size=3 --use_c10d --compress_tensors --compression_bits=8
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_c10d` | Enable C10d for wireless communication | `False` |
| `c10d_timeout_min` | Timeout in minutes for C10d operations | `30` |
| `compress_tensors` | Enable tensor compression | `False` |
| `compression_bits` | Bits for compression (8 or 16) | `8` |
| `wireless_retry_count` | Number of retries for wireless communication | `3` |

## Performance Considerations

- **Compression Tradeoffs**: Tensor compression reduces bandwidth requirements but adds computational overhead. For very small tensors, the overhead may outweigh the benefits.
- **Retry Mechanisms**: The retry mechanisms help with packet loss but can introduce latency. Adjust the `wireless_retry_count` based on your network quality.
- **Timeout Settings**: For unstable networks, increase the `c10d_timeout_min` parameter to prevent premature timeouts.

## Debugging

To enable detailed logging for wireless communication:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about compression ratios, retry attempts, and network performance.
