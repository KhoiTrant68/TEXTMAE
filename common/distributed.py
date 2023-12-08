import os
import builtins
import datetime

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """
    Get the distributed rank of the current process.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """
    Get the world size for distributed training.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    """
    Check if the current process is the main process.
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Save on master process only.
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def all_reduce_mean(x):
    """
    Perform all-reduce operation and calculate the mean.
    """
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def init_distributed_mode(args):
    """
    Initialize distributed training mode based on input arguments.
    """
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    Disable printing when not in the master process.
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with timestamp
            builtin_print(*args, **kwargs)

    builtins.print = print
