import os
import subprocess
import sys
import resource

def set_memory_limit(max_memory_mb):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_mb * 1024 * 1024, hard))

def main(node_rank, max_memory_mb):
    # Set memory limit
    set_memory_limit(max_memory_mb)

    # Set environment variables
    os.environ["MASTER_ADDR"] = "192.168.2.14"
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
        #"--max_memory_mb=" + str(max_memory_mb)
    ]

    # Run the command
    subprocess.run(command)

    # Print the communication overload and data transferred
    from pippy.PipelineDriver import PipelineDriverBase
    print(f"Communication overload: {PipelineDriverBase.get_communication_overload()}")
    print(f"Data transferred: {PipelineDriverBase.get_data_transferred_mb()} MB")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        #print("Usage: python run_torchrun.py <node_rank> <max_memory_mb>")
        print("Usage: python run_torchrun.py <node_rank>")
        sys.exit(1)
    
    node_rank = int(sys.argv[1])
    max_memory_mb = 6300
    print(f"Node rank: {node_rank}")
    print(f"Max memory: {max_memory_mb}")
    main(node_rank, max_memory_mb)