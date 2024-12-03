#!/bin/bash

torchrun --nproc_per_node=1 \
         --nnodes=2 \
         --node_rank=1  \ 
         --master_addr="192.168.2.11" \
         --master_port=1234 \
         example.py