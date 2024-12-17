import os
import torch
import torch.distributed as dist

#Give me a function can test the distributed training on multiple devices
def func():
    #initialize the process group
    dist.init_process_group(backend='mpi')
    #get the rank of the current device
    rank = dist.get_rank()
    #get the world size
    world_size = dist.get_world_size()
    #get the name of the current device
    name = torch.cuda.get_device_name(rank)
    #print the rank, world size and device name
    print(f"Running on rank {rank} of {world_size} on {name}")




def main():
    dist.init_process_group(backend='mpi')
    #get the rank of the current device
    rank = dist.get_rank()
    print(f"Running on rank {rank}")
    
if __name__ == "__main__":
    func()
