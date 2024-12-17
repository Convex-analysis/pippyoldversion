import os
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import nn

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

# Define a function to run on the remote worker
def remote_forward(model_rref, x):
    model = model_rref.local_value()
    return model(x)

def run_worker(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='mpi', rank=rank, world_size=world_size)

    # Initialize the RPC framework
    
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        # On the master node, create the model and send it to the worker
        model = MyModel()
        model_rref = rpc.remote("worker1", MyModel)
        x = torch.randn(10, 10)
        result = rpc.rpc_sync("worker1", remote_forward, args=(model_rref, x))
        print(f"Result from worker: {result}")
    else:
        # On the worker node, wait for RPCs
        print("waiting for RPC communication...")

    # Shutdown the RPC framework
    rpc.shutdown()

if __name__ == "__main__":
    world_size = 2
    os.environ['MASTER_ADDR'] = '192.168.2.14'
    os.environ['MASTER_PORT'] = '29500'
    
    # Use torchrun to launch the script
    rank = int(os.environ['RANK'])
    run_worker(rank, world_size)