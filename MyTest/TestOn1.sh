#!/bin/bash
export MASTER_ADDR="master_ip"
export MASTER_PORT=12345
export WORLD_SIZE=2
export RANK=1  # Set to 0 for master, 1 for worker

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="master_ip" --master_port=12345 test.py