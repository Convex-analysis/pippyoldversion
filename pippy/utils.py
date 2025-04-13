# Copyright (c) Meta Platforms, Inc. and affiliates
import os
import socket
import logging
from typing import List


# Pinning process to a separate GPU if not yet done by launch script
# Notes:
# 1. Previously this env was added to work around an issue that each RPC process creates an extra CUDA context on device
# 0.  This issue may have been caused by RPC not automatically pinning spawned worker threads to same CUDA device as the
# main thread. So pinning each RPC process to one device would avoid the issue.
# 2. This pinning must be done before `import torch` at which point CUDA context may have been created. Thus, if user
# code has `import torch` before importing PiPPy, this may not work.
# (Update): the issue in #1 seems to be gone as of March 2023. Hence, we are setting the default value of
# `PIPPY_PIN_DEVICE` to 0 now.
if os.getenv("PIPPY_PIN_DEVICE", "0") == "1":
    cuda_devices_str = os.getenv("CUDA_VISIBLE_DEVICES")
    if (
        cuda_devices_str is None  # not set
        or len(cuda_devices_str.split(",")) > 1
    ):  # or set to all devices
        # If launchers like Torchrun sets `LOCAL_RANK`, we would use this information
        local_rank_str = os.getenv("LOCAL_RANK")
        if local_rank_str is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = local_rank_str
            print(
                f"Pinning local process {local_rank_str} to gpu {os.getenv('CUDA_VISIBLE_DEVICES')}"
            )


import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc


PIPPY_VERBOSITY = os.environ.get("PIPPY_VERBOSITY", "OFF")

if PIPPY_VERBOSITY == "DEBUG":
    logging.getLogger().setLevel(logging.DEBUG)
elif PIPPY_VERBOSITY == "INFO":
    logging.getLogger().setLevel(logging.INFO)
elif PIPPY_VERBOSITY == "OFF":
    pass
else:
    print(f"Unsupported PIPPY_VERBOSITY level: {PIPPY_VERBOSITY}")


def get_rank() -> int:
    worker_info = rpc.get_worker_info()
    logging.debug(worker_info)
    return worker_info.id


def get_device() -> torch.device:
    worker_info = rpc.get_worker_info()
    agent = rpc._get_current_rpc_agent()
    dev_map = agent._get_device_map(worker_info)
    logging.debug(dev_map)
    num_devs = len(dev_map)

    if num_devs == 0:
        logging.debug("Empty device mapping, assuming device type to be cpu")
        device = torch.device("cpu")
    elif num_devs != 1:
        raise AssertionError(
            f"Expecting at most one device for RPC worker {worker_info}, "
            f"but got device map of length {num_devs}: {dev_map}"
        )
    else:
        src_dev = next(iter(dev_map))
        dst_dev = dev_map[src_dev]
        if src_dev != dst_dev:
            raise AssertionError(
                f"Expecting at most one device for RPC worker {worker_info}, "
                f"but got {dev_map}"
            )
        device = src_dev

    logging.info(f"Found device {device} for rank {worker_info.id}")
    return device


def get_pp_rank(rank: int, ranks: List[int]) -> int:
    for index, r in enumerate(ranks):
        if rank == r:
            return index
    raise ValueError(f"Rank {rank} not in ranks {ranks}")


def has_efa() -> bool:
    try:
        import subprocess

        return (
            subprocess.run(
                ["fi_info", "-p", "efa", "-t", "FI_EP_RDM"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except FileNotFoundError:
        return False
    except PermissionError:
        return False


def tp_transports():
    return ["shm", "uv"] if has_efa() else None


global _pp_group_barrier
# Defined later in `run_worker` (triggered via `run_pippy`)


# A barrier util for pipeline dimension
def pp_group_barrier():
    _pp_group_barrier()  # type: ignore[name-defined]


def run_pippy(run_func, args, *extra_args):
    if not hasattr(args, "world_size"):
        assert hasattr(args, "pp_group_size")
        args.dp_group_size = (
            args.dp_group_size if hasattr(args, "dp_group_size") else 1
        )
    else:
        if not hasattr(args, "dp_group_size"):
            args.pp_group_size = (
                args.pp_group_size
                if hasattr(args, "pp_group_size")
                else args.world_size
            )
            assert args.world_size % args.pp_group_size == 0
            args.dp_group_size = args.world_size // args.pp_group_size
        elif not hasattr(args, "pp_group_size"):
            args.dp_group_size = (
                args.dp_group_size if hasattr(args, "dp_group_size") else 1
            )
            assert args.world_size % args.dp_group_size == 0
            args.pp_group_size = args.world_size // args.dp_group_size
        else:
            pass
            # TODO: doesn't work for PiPPyTrainingArguments
            # assert args.world_size == args.dp_group_size * args.pp_group_size

    actual_world_size = args.dp_group_size * args.pp_group_size
    print(
        f"[PiPPy] World size: {actual_world_size}, "
        f"DP group size: {args.dp_group_size}, "
        f"PP group size: {args.pp_group_size}"
    )

    if args.rank == -1:
        mp.spawn(
            run_worker,
            args=(run_func, args, *extra_args),
            nprocs=actual_world_size,
            join=True,
        )
    elif args.rank < actual_world_size:
        run_worker(args.rank, run_func, args, *extra_args)
    else:
        print("I'm unused, exiting")


def run_worker(rank, run_func, args, *extra_args):
    # Declare global variables at the beginning of the function
    global dp_pg_per_pp_rank

    args.rank = rank

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    actual_world_size = args.dp_group_size * args.pp_group_size

    # TODO: Move to training args, blocked by: cannot pickle 'TensorPipeRpcBackendOptions' object
    # Exclude IB for metadata transport due to lack of EFA support on AWS
    if hasattr(args, "num_worker_threads"):
        num_worker_threads = args.num_worker_threads
    else:
        num_worker_threads = 512

    if hasattr(args, "rpc_timeout"):
        rpc_timeout = args.rpc_timeout
    else:
        rpc_timeout = 1800

    # Configure RPC options
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=num_worker_threads,
        rpc_timeout=rpc_timeout,
        _transports=tp_transports(),
    )

    if args.cuda:
        n_devs = torch.cuda.device_count()
        if n_devs > 0:
            dev_id = rank % n_devs
            for i in range(actual_world_size):
                options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
            # Does not seem effective for RPC device pinning. TODO
            # options.set_devices([f'cuda:{dev_id}'])
        else:
            args.cuda = 0
            print("Warning: no CUDA device found. Running on CPU instead.")

    args.device = f"cuda:{dev_id}" if args.cuda else "cpu"
    print(
        f"rank = {rank} host/pid/device = "
        f"{socket.gethostname()}/{os.getpid()}/{args.device}"
    )

    # Check if we should use C10d for wireless communication
    use_c10d = hasattr(args, "use_c10d") and args.use_c10d

    if use_c10d:
        try:
            # Import wireless utilities
            from pippy.wireless_utils import setup_wireless_c10d, create_jetson_optimized_groups

            # Get timeout in minutes (default to 30 if not specified)
            timeout_min = args.c10d_timeout_min if hasattr(args, "c10d_timeout_min") else 30

            # Initialize C10d with wireless optimizations
            print(f"Initializing C10d with wireless optimizations (rank {rank})")
            setup_wireless_c10d(
                rank=rank,
                world_size=actual_world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
                timeout_min=timeout_min,
                backend="nccl" if args.cuda else "gloo"
            )

            # Create process groups for wireless communication

            # Create optimized process groups for Jetson devices
            dp_groups, _, dp_ranks_per_pp_rank, _ = create_jetson_optimized_groups(
                world_size=actual_world_size,
                pp_group_size=args.pp_group_size,
                dp_group_size=args.dp_group_size
            )

            # Log the created groups
            print(f"Created {len(dp_groups)} data parallel groups with ranks: {dp_ranks_per_pp_rank}")

            # Store the groups for later use
            dp_pg_per_pp_rank = dp_groups

            # Initialize RPC for control messages
            rpc.init_rpc(
                f"worker{rank}",
                rank=rank,
                world_size=actual_world_size,
                rpc_backend_options=options,
            )

            print(f"Successfully initialized C10d with wireless optimizations (rank {rank})")

        except ImportError as e:
            print(f"Warning: Could not import wireless_utils: {e}. Falling back to standard initialization.")
            # Fall back to standard initialization
            use_c10d = False

    if not use_c10d:
        # Standard initialization
        # Init DDP process group
        backend = "nccl" if args.cuda else "gloo"
        torch.distributed.init_process_group(
            backend=backend, rank=rank, world_size=actual_world_size
        )

        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=actual_world_size,
            rpc_backend_options=options,
        )

        # Create standard process groups

        # Create data parallel ranks
        dp_ranks_per_pp_rank = (
            torch.arange(actual_world_size)
            .reshape(args.pp_group_size, args.dp_group_size)
            .tolist()
        )

        # Create process groups
        dp_pg_per_pp_rank = [
            torch.distributed.new_group(ranks) for ranks in dp_ranks_per_pp_rank
        ]



    # Process groups are already created in the C10d or standard initialization above

    pp_ranks_per_dp_group = [
        [i * args.dp_group_size + rank for i in range(args.pp_group_size)]
        for rank in range(args.dp_group_size)
    ]

    my_pp_ranks = pp_ranks_per_dp_group[rank % args.dp_group_size]

    args.driver_group = torch.distributed.new_group(
        list(range(args.dp_group_size))
    )

    global exclude_master
    exclude_master = (  # type: ignore[name-defined]
        args.exclude_master if hasattr(args, "exclude_master") else 0
    )
    gspmd = (  # type: ignore[name-defined]
        args.gspmd if hasattr(args, "gspmd") else 0
    )

    # A barrier util for pipeline dimension
    global _pp_group_barrier

    # ProcessGroupGloo cannot create group with strided ranks, e.g. [0, 2, 4, 6, ...]
    # Skipping the `pp_group` and `pp_group_barrier` creation here
    # TODO: unskip
    if torch.distributed.get_backend() == "gloo" and args.dp_group_size > 1:

        def _pp_group_barrier():
            logging.warning(
                f"pp_group_barrier() does not support ProcessGroupGloo with strided ranks {my_pp_ranks}. This will be a no-op."
            )

    else:
        pp_group = torch.distributed.new_group(my_pp_ranks)

        def _pp_group_barrier():
            logging.debug(
                f"Running pipeline group barrier on ranks {my_pp_ranks}"
            )
            torch.distributed.barrier(pp_group)

    if rank >= 0 and rank // args.dp_group_size == 0:
        args.driver_index = rank
        args.local_driver_index = os.getenv("LOCAL_RANK", rank)
        run_func(my_pp_ranks, args, *extra_args)
    elif gspmd == 1:
        run_func(my_pp_ranks, args, *extra_args)

    rpc.shutdown()
