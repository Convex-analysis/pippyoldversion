"""
Example script demonstrating how to use PiPPy with wireless C10d optimizations.
This example runs a simple pipeline-parallel model across multiple Jetson devices.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist

# Add the parent directory to the path so we can import pippy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pippy import PipelineDriver
from pippy.IR import Pipe, pipe_split
from pippy.utils import run_pippy

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            pipe_split(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            pipe_split(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.layers(x)

def run_worker(rank, world_size, args):
    """
    Worker function that runs on each device.
    """
    # Create the model
    model = SimpleModel()
    
    # Create a pipe from the model
    pipe = Pipe.from_tracing(model, torch.randn(args.batch_size, 512))
    
    # Create a pipeline driver with wireless optimizations
    driver = PipelineDriver.PipelineDriver1F1B(
        pipe=pipe,
        chunks=args.chunks,
        world_size=world_size,
        use_c10d=True,
        compress_tensors=args.compress_tensors,
        compression_bits=args.compression_bits,
        wireless_retry_count=args.wireless_retry_count
    )
    
    # Create input data
    input_data = torch.randn(args.batch_size, 512)
    
    # Run the model
    output = driver(input_data)
    
    # Print statistics
    if rank == 0:
        print(f"Communication overhead: {driver.communication_overload}")
        print(f"Data transferred: {driver.data_transferred_mb:.2f} MB")
        print(f"Output shape: {output.shape}")

def main():
    parser = argparse.ArgumentParser(description="PiPPy Wireless C10d Example")
    parser.add_argument("--world_size", type=int, default=3, help="Number of processes/devices")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--chunks", type=int, default=4, help="Number of microbatches")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master_port", type=str, default="29500", help="Master port")
    parser.add_argument("--compress_tensors", action="store_true", help="Enable tensor compression")
    parser.add_argument("--compression_bits", type=int, default=8, choices=[8, 16], help="Bits for compression")
    parser.add_argument("--wireless_retry_count", type=int, default=3, help="Number of retries for wireless communication")
    parser.add_argument("--c10d_timeout_min", type=int, default=30, help="Timeout in minutes for C10d operations")
    parser.add_argument("--use_c10d", action="store_true", help="Use C10d for wireless communication")
    args = parser.parse_args()
    
    # Run the distributed training
    run_pippy(
        run_worker,
        args.world_size,
        args=(args.world_size, args),
        master_addr=args.master_addr,
        master_port=args.master_port,
        use_c10d=args.use_c10d,
        c10d_timeout_min=args.c10d_timeout_min
    )

if __name__ == "__main__":
    main()
