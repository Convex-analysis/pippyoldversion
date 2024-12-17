#!/bin/bash
export MASTER_ADDR="192.168.2.14"
export MASTER_PORT=29500
export WORLD_SIZE=2

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=$1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT MultiDeviceTest.py --batch_size=32 --schedule=FillDrain --device=cuda
