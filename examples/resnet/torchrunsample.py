import os
import subprocess
import sys
import torch
import itertools

DEVICE_IP_LIST = [
    "192.168.2.14",
    "192.168.2.13"
]

def intra_cluster_loop():
    #Set master address and port
    os.environ["MASTER_ADDR"] = "192.168.2.14"
    os.environ["MASTER_PORT"] = "29500"
    #set world size
    os.environ["WORLD_SIZE"] = "3"
    #TODO this line will replace after the pipeline timplate generation
    rank_list = [
        [1,2],#node 1's rank list [1,2]
        [2,1]#node 2's rank list [2,1]
    ]
    
    for pipe_template in rank_list:
        #get the current node ip
        current_node_ip = os.popen('hostname -I').read().split()[0]
        if current_node_ip == os.environ["MASTER_ADDR"]:
            #if the current node is the master node, set the node rank to 0
            os.environ["NODE_RANK"] = "0"
        elif current_node_ip == DEVICE_IP_LIST[0]:
            os.environ["NODE_RANK"] = pipe_template[0]
        elif current_node_ip == DEVICE_IP_LIST[1]:
            os.environ["NODE_RANK"] = pipe_template[1]   
            
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
        
        
    
    
    
     
    
    



def main(node_rank, max_memory_mb):

    # Set environment variables
    os.environ["MASTER_ADDR"] = "192.168.2.14"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["OMP_NUM_THREADS"] = "4"  # Set this to the number of CPU cores
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


if __name__ == "__main__":
    intra_cluster_loop()
    '''
    if len(sys.argv) != 2:
        #print("Usage: python run_torchrun.py <node_rank> <max_memory_mb>")
        print("Usage: python run_torchrun.py <node_rank>")
        sys.exit(1)
    
    node_rank = int(sys.argv[1])
    main(node_rank, 6300)
    '''