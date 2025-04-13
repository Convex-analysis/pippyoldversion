"""
Utilities for optimizing PiPPy communication over wireless networks using C10d.
Specifically designed for NVIDIA Jetson devices in a distributed setting.
"""

import os
import time
import logging
import datetime
import torch
import torch.distributed as dist
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

def setup_wireless_c10d(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: str,
    timeout_min: int = 30,
    backend: str = None
) -> None:
    """
    Set up C10d communication optimized for wireless networks.

    Args:
        rank: The rank of the current process
        world_size: Total number of processes
        master_addr: IP address of the master node
        master_port: Port to use for communication
        timeout_min: Timeout in minutes for operations
        backend: Communication backend ('nccl' or 'gloo')
    """
    # Set environment variables
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # Choose appropriate backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # For Jetson devices, NCCL might need specific configurations
    if backend == "nccl" and torch.cuda.is_available():
        # Set NCCL parameters for wireless networks
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Adjust based on your network interface
        os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand
        os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable GPU Direct P2P for wireless

    # Initialize process group with a longer timeout for wireless
    timeout_sec = timeout_min * 60
    logger.info(f"Initializing process group with rank {rank}, world_size {world_size}, "
                f"backend {backend}, timeout {timeout_min} minutes")

    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=timeout_sec)
    )

    logger.info(f"Process group initialized for rank {rank}")

def compress_tensor(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compress a tensor to reduce communication bandwidth.

    Args:
        tensor: The tensor to compress
        bits: Number of bits for quantization (8 or 16)

    Returns:
        Tuple of (compressed_tensor, scale)
    """
    if bits not in [8, 16]:
        raise ValueError("Bits must be either 8 or 16")

    # Get tensor type based on bits
    dtype = torch.int8 if bits == 8 else torch.int16

    # Calculate scale factor
    abs_max = tensor.abs().max()
    if abs_max == 0:
        # Handle zero tensors
        return tensor.to(dtype), torch.tensor([1.0], device=tensor.device)

    scale = abs_max / (2**(bits-1) - 1)

    # Quantize
    compressed = (tensor / scale).round().to(dtype)

    return compressed, scale

def decompress_tensor(compressed_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Decompress a tensor that was compressed with compress_tensor.

    Args:
        compressed_tensor: The compressed tensor
        scale: The scale factor used for compression

    Returns:
        The decompressed tensor
    """
    return compressed_tensor.float() * scale

def reliable_broadcast(
    tensor: torch.Tensor,
    src: int,
    group: Optional[dist.ProcessGroup] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> torch.Tensor:
    """
    Perform a reliable broadcast operation with retries for wireless networks.

    Args:
        tensor: Tensor to broadcast
        src: Source rank
        group: Process group
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        The broadcast tensor
    """
    for attempt in range(max_retries):
        try:
            dist.broadcast(tensor, src, group=group)
            return tensor
        except (RuntimeError, dist.DistBackendError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Broadcast failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"Broadcast attempt {attempt+1} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

def reliable_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> torch.Tensor:
    """
    Perform a reliable all_reduce operation with retries for wireless networks.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation
        group: Process group
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        The reduced tensor
    """
    for attempt in range(max_retries):
        try:
            dist.all_reduce(tensor, op=op, group=group)
            return tensor
        except (RuntimeError, dist.DistBackendError) as e:
            if attempt == max_retries - 1:
                logger.error(f"All-reduce failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"All-reduce attempt {attempt+1} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

def measure_network_bandwidth(
    tensor_size_mb: float = 10.0,
    warmup_rounds: int = 2,
    measurement_rounds: int = 5
) -> float:
    """
    Measure the network bandwidth between nodes.

    Args:
        tensor_size_mb: Size of the test tensor in MB
        warmup_rounds: Number of warmup rounds
        measurement_rounds: Number of measurement rounds

    Returns:
        Estimated bandwidth in MB/s
    """
    rank = dist.get_rank()
    # Get world size (used for logging)
    _ = dist.get_world_size()

    # Create test tensor
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
    tensor = torch.randn(num_elements, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

    # Warmup
    for _ in range(warmup_rounds):
        dist.broadcast(tensor, 0)
        dist.barrier()

    # Measure
    bandwidths = []
    for _ in range(measurement_rounds):
        start_time = time.time()
        dist.broadcast(tensor, 0)
        dist.barrier()
        end_time = time.time()

        elapsed = end_time - start_time
        bandwidth = tensor_size_mb / elapsed
        bandwidths.append(bandwidth)

    avg_bandwidth = sum(bandwidths) / len(bandwidths)

    if rank == 0:
        logger.info(f"Estimated network bandwidth: {avg_bandwidth:.2f} MB/s")

    return avg_bandwidth

def create_jetson_optimized_groups(
    world_size: int,
    pp_group_size: int,
    dp_group_size: int
) -> Tuple[List[dist.ProcessGroup], List[dist.ProcessGroup], List[List[int]], List[List[int]]]:
    """
    Create optimized process groups for Jetson devices.

    Args:
        world_size: Total number of processes
        pp_group_size: Pipeline parallelism group size
        dp_group_size: Data parallelism group size

    Returns:
        Tuple of (dp_groups, pp_groups, dp_ranks_per_pp_rank, pp_ranks_per_dp_rank)
    """
    assert world_size == pp_group_size * dp_group_size, "World size must equal pp_group_size * dp_group_size"

    # Create data parallel groups (processes that handle the same pipeline stage)
    dp_groups = []
    dp_ranks_per_pp_rank = []

    for pp_rank in range(pp_group_size):
        ranks = [pp_rank * dp_group_size + i for i in range(dp_group_size)]
        group = dist.new_group(ranks)
        dp_groups.append(group)
        dp_ranks_per_pp_rank.append(ranks)

    # Create pipeline parallel groups (processes in the same pipeline)
    pp_groups = []
    pp_ranks_per_dp_rank = []

    for dp_rank in range(dp_group_size):
        ranks = [dp_rank + pp_rank * dp_group_size for pp_rank in range(pp_group_size)]
        group = dist.new_group(ranks)
        pp_groups.append(group)
        pp_ranks_per_dp_rank.append(ranks)

    return dp_groups, pp_groups, dp_ranks_per_pp_rank, pp_ranks_per_dp_rank
