
import os
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.args import train_option

from common.pos_embed import interpolate_pos_embed
from common.distributed import (
    init_distributed_mode,
    get_rank,
    get_world_size,
    is_main_process,
)
from common.config_optimizers import configure_optimizers
from common.model_utils import load_model, save_model

from utils.dataset import get_image_dataset
from utils.score_cal import write_total_score

from loss.scaler import NativeScalerWithGradNormCount
from loss.rd_loss import RateDistortionLoss

from model import MAEC
from engine_finetune import train_one_epoch, val_one_epoch

def main(args):
    init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    # Set up device
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)

    # Benchmark cuda
    cudnn.benchmark = True

    # Preprocess to make total_score for all image of datasets
    if not args.eval:
        assert args.dataset_path is not None and args.total_scores_path is not None
        write_total_score(mode="train", args=args)
        write_total_score(mode="val", args=args)
    else:
        assert args.dataset_path is not None and args.total_scores_path is not None
        write_total_score(mode="test", args=args)
    
    # Data preprocessing and dataset creation
    dataset_train = get_image_dataset(mode="train", args=args)
    dataset_val = get_image_dataset(mode="val", args=args)

    if True: # args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Dataloader setup
    dataloader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    dataloader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Setup model
    model = MAEC.__dict__[args.model]()

    # Load Mae Encoder model
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.fintune, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint["model"]

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        # Interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model)

    # Change device
    model.to(device)

    # If don't use distributed data parallel
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    # Calculate the effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()

        # Setup learning rates(lr and aux_lr) for training process
    if args.lr is None:  # Only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.aux_lr is None:  # Only base_lr is specified
        args.aux_lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)


    if args.distributed:
        # Pass model to ddp
        print("Model is trained with distributed data parallel")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Optimizer
    optimizer, aux_optimizer = configure_optimizers(model, args)

    # Loss function 
    loss_scaler = NativeScalerWithGradNormCount()
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    best_loss = 1e10


    # Load model
    load_model(
        args=args,
        model_without_ddp=model,
        optimizer=optimizer,
        aux_optimizer=aux_optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats = val_one_epoch(
            0, dataloader_val, model, criterion, args
        )
        print(test_stats)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    last_epoch = 0 + args.start_epoch
    optimizer.param_groups[0]["lr"] = args.lr
    for epoch in range(last_epoch, args.epochs, 5):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_stats = train_one_epoch(
            model,
            criterion,
            dataloader_train,
            optimizer,
            aux_optimizer,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
        )
        test_stats = val_one_epoch(epoch, dataloader_val, model, criterion)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            if test_stats['loss'] < best_loss:
                save_model(
                    args=args,
                    epoch=epoch,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    aux_optimizer=aux_optimizer,
                    loss_scaler=loss_scaler,
                )
                best_loss = test_stats['loss']


if __name__ == "__main__":
    args = train_option()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)