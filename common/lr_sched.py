import math 

def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust the learning rate using a half-cycle cosine schedule after warm-up.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to adjust.
        epoch (int): The current epoch.
        args (argparse.Namespace): Command-line arguments.

    Returns:
        lr (float): The adjusted learning rate.
    """
    if epoch < args.warmup_epochs:
        # Warm-up phase: linearly increase the learning rate
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        # Cosine decay phase
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    # Update the learning rate in the optimizer's parameter groups
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            # Scale the learning rate if lr_scale is specified
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    
    return lr
