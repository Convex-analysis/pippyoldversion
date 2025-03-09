import os
import subprocess
import sys


def main(node_rank, max_memory_mb):

    # Set environment variables
    os.environ["MASTER_ADDR"] = "219.216.64.145"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"  # Set this to the number of CPU cores
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
        "pippy_resnet152.py",
        "--record_mem_dumps=0",
        "--checkpoint=0"
        #"--max_memory_mb=" + str(max_memory_mb)
    ]

    # Run the command
    subprocess.run(command)

    # Print the communication overload and data transferred


if __name__ == "__main__":
    # intra_cluster_loop()

    if len(sys.argv) != 2:
        #print("Usage: python run_torchrun.py <node_rank> <max_memory_mb>")
        print("Usage: python run_torchrun.py <node_rank>")
        sys.exit(1)
    
    node_rank = int(sys.argv[1])
    main(node_rank, 6300)
