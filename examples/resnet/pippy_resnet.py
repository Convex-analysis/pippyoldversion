# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import sys
from functools import reduce
import psutil

import torch
from torch import optim
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms  # type: ignore
from tqdm import tqdm  # type: ignore

import pippy.fx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe, LossWrapper, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.events import EventsContext
from pippy.microbatch import sum_reducer, TensorChunkSpec
from pippy.visualizer import events_to_json
from resnet import ResNet50, ResNet34, ResNet101, ResNet152

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

pippy.fx.Tracer.proxy_buffer_attributes = True

USE_TQDM = bool(int(os.getenv('USE_TQDM', '1')))


def run_master(_, args):
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)
    print("Using device:", args.device)
    if args.rank == 0:
        number_of_workers = 2
        #all_worker_ranks = list(range(1, 1 + number_of_workers))  # exclude master rank = 0
        all_worker_ranks = list(range(0, number_of_workers))  # include master rank = 0
        chunks = len(all_worker_ranks)
        batch_size = args.batch_size * chunks

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        valid_data = datasets.CIFAR10('./data', train=False, transform=transform)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=chunks, rank=args.rank)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, num_replicas=chunks, rank=args.rank)

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, pin_memory=True)

        class OutputLossWrapper(LossWrapper):
            def __init__(self, module, loss_fn):
                super().__init__(module, loss_fn)

            @torch.autocast(device_type="cuda")
            def forward(self, input, target):
                output = self.module(input)
                loss = self.loss_fn(output, target)
                # Here we use a dict with the "loss" keyword so that PiPPy can automatically find the loss field when
                # generating the backward pass
                return {"output": output, "loss": loss}

        model = ResNet34()

        annotate_split_points(model, {
            'layer1': PipeSplitWrapper.SplitPoint.END,
            #'layer2': PipeSplitWrapper.SplitPoint.END,
            #'layer2.0': PipeSplitWrapper.SplitPoint.END,
            #'layer3': PipeSplitWrapper.SplitPoint.BEGINNING,
        })

        wrapper = OutputLossWrapper(model, cross_entropy)

        pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG)
        pipe.to(args.device)

        output_chunk_spec = (TensorChunkSpec(0), sum_reducer)
        process = psutil.Process(os.getpid())
        print(f"Memory usage before pipelineDriver: {process.memory_info().rss / 1024 / 1024} MB")
        pipe_driver: PipelineDriverBase = schedules[args.schedule](pipe, chunks,
                                                                len(all_worker_ranks),
                                                                all_ranks=all_worker_ranks,
                                                                output_chunk_spec=output_chunk_spec,
                                                                _record_mem_dumps=bool(args.record_mem_dumps),
                                                                checkpoint=bool(args.checkpoint))

        optimizer = pipe_driver.instantiate_optimizer(optim.Adam, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        print(f"Memory usage after pipelineDriver: {process.memory_info().rss / 1024 / 1024} MB")
        loaders = {
            "train": train_dataloader,
            "valid": valid_dataloader
        }

        

        this_file_name = os.path.splitext(os.path.basename(__file__))[0]
        pipe_visualized_filename = f"{this_file_name}_visualized_{args.rank}.json"
        batches_events_contexts = []

        for epoch in range(args.max_epochs):
            print(f"Epoch: {epoch + 1}")
            
            print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")
            for k, dataloader in loaders.items():
                epoch_correct = 0
                epoch_all = 0
                # Randomly select num_batches_to_process batches data
                num_batches_to_process = 10  # Change this value to the desired number of batches
                indices = torch.randperm(len(dataloader))[:num_batches_to_process]
                for i, (x_batch, y_batch) in enumerate(tqdm(dataloader) if USE_TQDM else dataloader):
                    if i not in indices:
                        continue
                    x_batch = x_batch.to(args.device)
                    y_batch = y_batch.to(args.device)
                    if k == "train":
                        pipe_driver.train()
                        optimizer.zero_grad()
                        outp, _ = pipe_driver(x_batch, y_batch)
                        preds = outp.argmax(-1)
                        correct = (preds == y_batch).sum()
                        all = len(y_batch)
                        epoch_correct += correct.item()
                        epoch_all += all
                        optimizer.step()
                    else:
                        pipe_driver.eval()
                        with torch.no_grad():
                            outp, _ = pipe_driver(x_batch, y_batch)
                            preds = outp.argmax(-1)
                            correct = (preds == y_batch).sum()
                            all = len(y_batch)
                            epoch_correct += correct.item()
                            epoch_all += all

                    if args.visualize:
                        batches_events_contexts.append(pipe_driver.retrieve_events())
                print(f"Loader: {k}. Accuracy: {epoch_correct / epoch_all}")
        if args.visualize:
            all_events_contexts: EventsContext = reduce(lambda c1, c2: EventsContext().update(c1).update(c2),
                                                        batches_events_contexts, EventsContext())
            with open(pipe_visualized_filename, "w") as f:
                f.write(events_to_json(all_events_contexts))
            print(f"Saved {pipe_visualized_filename}")
        print('Finished')

    else:
        print("This is a worker rank")
        print(f"Current Memory usage in this node is : {process.memory_info().rss / 1024 / 1024} MB")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 3)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--visualize', type=int, default=0, choices=[0, 1])
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--num_worker_threads', type=int, default=16)
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    #args.world_size = 2  # "This program requires exactly 4 workers + 1 master"
    print(torch.cuda.is_available())
    run_pippy(run_master, args)
