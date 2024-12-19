import os
import subprocess
import sys

def main(node_rank):
    # Set environment variables
    os.environ["MASTER_ADDR"] = "192.168.2.10"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["NODE_RANK"] = str(node_rank)

    # Print the environment variables
    print(os.environ["MASTER_ADDR"])
    print(os.environ["MASTER_PORT"])

    # Construct the torchrun command
    command = [
        "torchrun",
        "--nproc_per_node=1",
        "--nnodes=" + os.environ["WORLD_SIZE"],
        "--node_rank=" + os.environ["NODE_RANK"],
        "--master_addr=" + os.environ["MASTER_ADDR"],
        "--master_port=" + os.environ["MASTER_PORT"],
        "pippy_resnet.py",
        "--record_mem_dumps=0",
        "--checkpoint=0"
    ]

    # Run the command
    subprocess.run(command)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_torchrun.py <node_rank>")
        sys.exit(1)
    
    node_rank = sys.argv[1]
    main(node_rank)