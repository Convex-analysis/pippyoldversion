#!/bin/bash
# run_torchrun.sh

export MASTER_ADDR="192.168.2.14"
export MASTER_PORT=29500
export WORLD_SIZE=2

# Set the rank based on the argument passed to the script
export NODE_RANK=$1
echo $MASTER_ADDR
echo $MASTER_PORT

torchrun --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pippy_resnet.py --record_mem_dumps=0 --checkpoint=0

