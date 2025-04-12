"""
Unit tests for the wireless_utils module.
"""

import os
import sys
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add the parent directory to the path so we can import pippy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pippy.wireless_utils import (
    compress_tensor,
    decompress_tensor,
    reliable_broadcast,
    reliable_all_reduce,
    create_jetson_optimized_groups
)

def _init_process_group(rank, world_size, master_addr, master_port):
    """Initialize the process group for testing."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_addr}:{master_port}"
    )

def _test_compress_decompress(rank, world_size, master_addr, master_port):
    """Test tensor compression and decompression."""
    _init_process_group(rank, world_size, master_addr, master_port)
    
    # Create a test tensor
    tensor = torch.randn(100, 100)
    
    # Test 8-bit compression
    compressed_8bit, scale_8bit = compress_tensor(tensor, bits=8)
    decompressed_8bit = decompress_tensor(compressed_8bit, scale_8bit)
    
    # Test 16-bit compression
    compressed_16bit, scale_16bit = compress_tensor(tensor, bits=16)
    decompressed_16bit = decompress_tensor(compressed_16bit, scale_16bit)
    
    # Calculate error
    error_8bit = torch.abs(tensor - decompressed_8bit).mean().item()
    error_16bit = torch.abs(tensor - decompressed_16bit).mean().item()
    
    # Print results
    if rank == 0:
        print(f"8-bit compression error: {error_8bit}")
        print(f"16-bit compression error: {error_16bit}")
        print(f"8-bit compression ratio: {tensor.numel() * 4 / (compressed_8bit.numel() + scale_8bit.numel() * 4)}")
        print(f"16-bit compression ratio: {tensor.numel() * 4 / (compressed_16bit.numel() * 2 + scale_16bit.numel() * 4)}")
        
        # Verify that 16-bit has lower error than 8-bit
        assert error_16bit < error_8bit, "16-bit compression should have lower error than 8-bit"
        
        # Verify that error is within acceptable range
        assert error_8bit < 0.1, "8-bit compression error too high"
        assert error_16bit < 0.01, "16-bit compression error too high"
    
    dist.barrier()
    dist.destroy_process_group()

def _test_reliable_broadcast(rank, world_size, master_addr, master_port):
    """Test reliable broadcast."""
    _init_process_group(rank, world_size, master_addr, master_port)
    
    # Create a test tensor
    if rank == 0:
        tensor = torch.randn(100, 100)
    else:
        tensor = torch.zeros(100, 100)
    
    # Broadcast the tensor
    reliable_broadcast(tensor, src=0, max_retries=3)
    
    # Verify that all ranks have the same tensor
    sum_tensor = tensor.sum().item()
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, torch.tensor([sum_tensor]))
    
    if rank == 0:
        for i in range(1, world_size):
            assert abs(gathered[0].item() - gathered[i].item()) < 1e-5, f"Tensor at rank {i} differs from rank 0"
    
    dist.barrier()
    dist.destroy_process_group()

def _test_reliable_all_reduce(rank, world_size, master_addr, master_port):
    """Test reliable all-reduce."""
    _init_process_group(rank, world_size, master_addr, master_port)
    
    # Create a test tensor
    tensor = torch.ones(100, 100) * rank
    
    # All-reduce the tensor
    reliable_all_reduce(tensor, op=dist.ReduceOp.SUM, max_retries=3)
    
    # Verify that all ranks have the sum
    expected_sum = sum(range(world_size))
    assert abs(tensor.mean().item() - expected_sum) < 1e-5, f"All-reduce result incorrect at rank {rank}"
    
    dist.barrier()
    dist.destroy_process_group()

def _test_create_groups(rank, world_size, master_addr, master_port):
    """Test creating optimized process groups."""
    _init_process_group(rank, world_size, master_addr, master_port)
    
    # Only test if world_size is a composite number
    if world_size >= 4 and world_size % 2 == 0:
        # Find factors
        pp_group_size = 2
        dp_group_size = world_size // 2
        
        # Create groups
        dp_groups, pp_groups, dp_ranks, pp_ranks = create_jetson_optimized_groups(
            world_size=world_size,
            pp_group_size=pp_group_size,
            dp_group_size=dp_group_size
        )
        
        # Verify that each rank is in exactly one dp group and one pp group
        dp_group_idx = -1
        pp_group_idx = -1
        
        for i, ranks in enumerate(dp_ranks):
            if rank in ranks:
                dp_group_idx = i
                break
        
        for i, ranks in enumerate(pp_ranks):
            if rank in ranks:
                pp_group_idx = i
                break
        
        assert dp_group_idx >= 0, f"Rank {rank} not found in any dp group"
        assert pp_group_idx >= 0, f"Rank {rank} not found in any pp group"
        
        # Verify that groups are correctly formed
        assert len(dp_groups) == pp_group_size, "Incorrect number of dp groups"
        assert len(pp_groups) == dp_group_size, "Incorrect number of pp groups"
    
    dist.barrier()
    dist.destroy_process_group()

class TestWirelessUtils(unittest.TestCase):
    """Test cases for wireless_utils module."""
    
    def setUp(self):
        self.world_size = 2
        self.master_addr = "localhost"
        self.master_port = "29500"
    
    def test_compress_decompress(self):
        """Test tensor compression and decompression."""
        mp.spawn(
            _test_compress_decompress,
            args=(self.world_size, self.master_addr, self.master_port),
            nprocs=self.world_size,
            join=True
        )
    
    def test_reliable_broadcast(self):
        """Test reliable broadcast."""
        mp.spawn(
            _test_reliable_broadcast,
            args=(self.world_size, self.master_addr, self.master_port),
            nprocs=self.world_size,
            join=True
        )
    
    def test_reliable_all_reduce(self):
        """Test reliable all-reduce."""
        mp.spawn(
            _test_reliable_all_reduce,
            args=(self.world_size, self.master_addr, self.master_port),
            nprocs=self.world_size,
            join=True
        )
    
    def test_create_groups(self):
        """Test creating optimized process groups."""
        mp.spawn(
            _test_create_groups,
            args=(self.world_size, self.master_addr, self.master_port),
            nprocs=self.world_size,
            join=True
        )

if __name__ == "__main__":
    unittest.main()
