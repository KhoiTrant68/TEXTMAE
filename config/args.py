import argparse


def train_option():
    """
    Define training options and configuration using argparse.

    Returns:
        argparse.ArgumentParser: The argument parser for training options.
    """
    # Training configuration
    parser = argparse.ArgumentParser(description="Training script.", add_help=False)

    # Dataset parameters
    parser.add_argument(
        "--dataset_path", default="datasets/kodak", type=str, help="Dataset path"
    )
    parser.add_argument(
        "--total_scores_path", default="total_scores", type=str, help="Total score path"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_dir",
        help="Path to save, empty for no saving",
    )
    parser.add_argument("--log_dir", default="logs_dir", help="Path tensorboard log")

    # Batch and Epochs
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU ",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="maec_base_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="Images input size")
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="Learning rate (absolute lr)",
    )
    parser.add_argument(
        "--aux_lr",
        type=float,
        default=1e-4,
        metavar="AUX_LR",
        help="Auxiliary learning rate (absolute auxiliary lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=None,
        metavar="LR",
        help="Base learning rate: absolute_lr = base_lr * total_batch_size / 256 ",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="Lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=2, metavar="N", help="Epochs to warmup LR"
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-4,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )

    # Finetuning parameters
    parser.add_argument("--finetune", default=None, help="Finetune from checkpoint")
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="Start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )

    # Device parameters
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="URL used to set up distributed training"
    )

    return parser
