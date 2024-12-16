import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend='mpi')
    rank = dist.get_rank()
    print(f"Running on rank {rank}")

if __name__ == "__main__":
    main()
