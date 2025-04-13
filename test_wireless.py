"""
Simple test script to verify that the wireless C10d utilities work correctly.
"""

import os
import sys
import torch
import torch.nn as nn
from pippy.IR import Pipe, pipe_split
from pippy import PipelineDriver
from pippy.utils import run_pippy

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            pipe_split(),
            nn.Linear(20, 10),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.layers(x)

def run_worker(ranks, args):
    """
    Worker function that runs on each device.
    """
    print(f"Worker {args.rank} starting")
    
    # Create the model
    model = SimpleModel()
    
    # Create a pipe from the model
    pipe = Pipe.from_tracing(model, torch.randn(2, 10))
    
    # Create a pipeline driver with wireless optimizations
    driver = PipelineDriver.PipelineDriver1F1B(
        pipe=pipe,
        chunks=2,
        world_size=len(ranks),
        use_c10d=True,
        compress_tensors=True,
        compression_bits=8,
        wireless_retry_count=3
    )
    
    # Create input data
    input_data = torch.randn(2, 10)
    
    # Run the model
    output = driver(input_data)
    
    # Print statistics
    if args.rank == 0:
        print(f"Output shape: {output.shape}")
        print(f"Communication overhead: {driver.communication_overload}")
        print(f"Data transferred: {driver.data_transferred_mb:.2f} MB")
        print("Test completed successfully!")

if __name__ == "__main__":
    # Set up arguments
    class Args:
        def __init__(self):
            self.rank = -1
            self.master_addr = "localhost"
            self.master_port = "29500"
            self.pp_group_size = 2
            self.dp_group_size = 1
            self.cuda = 0
    
    args = Args()
    
    # Run the test
    run_pippy(run_worker, args)
