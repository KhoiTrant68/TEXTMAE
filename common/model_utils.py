from pathlib import Path
import torch
from common.distributed import save_on_master


def load_model(model_without_ddp, optimizer, aux_optimizer, loss_scaler, args):
    """
    Load a model checkpoint if resuming training.

    Args:
        model_without_ddp (nn.Module): Model to load the checkpoint into.
        optimizer (torch.optim.Optimizer): The primary optimizer.
        aux_optimizer (torch.optim.Optimizer): Auxiliary optimizer.
        loss_scaler (torch.cuda.amp.GradScaler): GradScaler for gradient scaling.
        args (argparse.Namespace): Command-line arguments.
    """
    if args.resume:
        # Load the model checkpoint
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cuda", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cuda")

        # Load the model state
        model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)

        # Load optimizer states if available
        if (
            "optimizer" in checkpoint
            and "aux_optimizer" in checkpoint
            and "epoch" in checkpoint
            and not (hasattr(args, "eval") and args.eval)
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")


def save_model(epoch, model, model_without_ddp, optimizer, aux_optimizer, loss_scaler, args):
    """
    Save the model checkpoint.

    Args:
        epoch (int): Current epoch.
        model (nn.Module): Model to save.
        model_without_ddp (nn.Module): Model without ddp to save.
        optimizer (torch.optim.Optimizer): The primary optimizer.
        aux_optimizer (torch.optim.Optimizer): Auxiliary optimizer.
        loss_scaler (torch.cuda.amp.GradScaler): GradScaler for gradient scaling.
        args (argparse.Namespace): Command-line arguments.
    """
    output_dir = Path(args.output_dir)
    if not Path(output_dir).exists():
        Path.mkdir(output_dir)

    if loss_scaler is not None:
        # Save checkpoint with GradScaler state
        checkpoint_paths = [output_dir / ("best_model.pth")]

        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        # Save checkpoint without GradScaler state
        client_state = {"epoch": epoch}
        model.save_checkpoint(
            save_dir=args.output_dir, tag="best_model", client_state=client_state
        )
