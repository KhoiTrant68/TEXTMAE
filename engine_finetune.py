import math
import sys
import datetime
from typing import Iterable

import torch
from timm.data import Mixup
from timm.utils import accuracy

from compressai.models import CompressionModel
from common.distributed import all_reduce_mean
from utils.logger import MetricLogger, SmoothedValue


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = float(0.0)
        self.avg = float(0.0)
        self.sum = float(0.0)
        self.count = float(0.0)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model: CompressionModel,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    aux_optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_scaler,
    clip_max_norm: float = 0,
    log_writer=None,
    args=None,
):
    """
    Train one epoch of the model.

    Args:
        model (CompressionModel ~ nn.Module): The model to train. Ex: MCM
        criterion (torch.nn.Module): The loss function.
        data_loader (DataLoader): The dataloader for training data
        optimizer (torch.optim.Optimizer): The primary optimizer for the model.
        aux_optimizer (torch.optim.Optimizer): The auxiliary optimizer (used for EntropyBottleneck in CompressionModel).
        epoch (int): The current epoch.
        loss_scaler (torch.cuda.amp.GradScaler): The loss scaler.
        clip_max_norm (float): Maximum gradient norm for gradient clipping.
        mixup_fn (Optional[Mixup]): The Mixup function.
        log_writer (Optional[SummaryWriter]): The tensorboard writer.
        args: Additional arguments or configuration

    Returns:
        dict (dict): A dictionary of averaged statistics for various metrics
    """

    # Initialize variables and set the model to training mode
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    # Initialize accumulative iteration.
    accum_iter = args.accum_iter

    # Synchronize with the same device
    device = next(model.parameters()).device

    # Initialize optimizaters for training
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    # Logging directory of training
    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    # Calculate the runtime
    t0 = datetime.datetime.now()

    for data_iter_step, (samples, _, total_scores) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        total_scores = total_scores.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples, total_scores)
            loss = criterion(outputs, samples)

        # Compute the loss values for each iteration.
        loss["loss"] /= accum_iter

        # aux_loss = model.module.aux_loss()
        aux_loss = model.aux_loss()

        aux_loss /= accum_iter

        # loss_scaler(
        #     loss["loss"],
        #     optimizer,
        #     clip_grad=clip_max_norm,
        #     parameters=model.module.parameters(),
        #     create_graph=False,
        #     update_grad=(data_iter_step + 1) % accum_iter == 0,
        # )

        # loss_scaler(
        #     aux_loss,
        #     aux_optimizer,
        #     clip_grad=clip_max_norm,
        #     parameters=model.module.parameters(),
        #     create_graph=False,
        #     update_grad=(data_iter_step + 1) % accum_iter == 0,
        # )

        if (data_iter_step + 1) % accum_iter == 0:
            loss["loss"].backward()
            if clip_max_norm is not None and clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            aux_loss.backward()
            aux_optimizer.step()

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

        torch.cuda.synchronize()

        loss_value = loss["loss"].item()
        L1_loss_value = loss["L1_loss"].item()
        ssim_loss_value = loss["ssim_loss"].item()
        vgg_loss_value = loss["vgg_loss"].item()
        bpp_loss_value = loss["bpp_loss"].item()
        aux_loss_value = aux_loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(L1_loss=L1_loss_value)
        metric_logger.update(ssim_loss=ssim_loss_value)
        metric_logger.update(vgg_loss=vgg_loss_value)
        metric_logger.update(bpp_loss=bpp_loss_value)
        metric_logger.update(aux_loss=aux_loss_value)

        min_lr = min_aux_lr = 10.0
        max_lr = max_aux_lr = 0.0

        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        for group in aux_optimizer.param_groups:
            min_aux_lr = min(min_aux_lr, group["lr"])
            max_aux_lr = max(max_aux_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(aux_lr=max_aux_lr)

        loss_value_reduce = all_reduce_mean(loss_value)
        L1_loss_value_reduce = all_reduce_mean(L1_loss_value)
        ssim_loss_value_reduce = all_reduce_mean(ssim_loss_value)
        vgg_loss_value_reduce = all_reduce_mean(vgg_loss_value)
        bpp_loss_value_reduce = all_reduce_mean(bpp_loss_value)
        aux_loss_value_reduce = all_reduce_mean(aux_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_100x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_100x = int((data_iter_step / len(data_loader) + epoch) * 100)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_100x)
            log_writer.add_scalar("L1_loss", L1_loss_value_reduce, epoch_100x)
            log_writer.add_scalar("ssim_loss", ssim_loss_value_reduce, epoch_100x)
            log_writer.add_scalar("vgg_loss", vgg_loss_value_reduce, epoch_100x)
            log_writer.add_scalar("bpp_loss", bpp_loss_value_reduce, epoch_100x)
            log_writer.add_scalar("aux_loss", aux_loss_value_reduce, epoch_100x)
            log_writer.add_scalar("lr", max_lr, epoch_100x)
            log_writer.add_scalar("aux_lr", max_aux_lr, epoch_100x)

        if data_iter_step % 50 == 0:
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            dt = deltatime.seconds + 1e-6 * deltatime.microseconds
            print(
                f"Train epoch {epoch}: ["
                f"{data_iter_step * len(samples)}/{len(data_loader.dataset)}"
                f" ({100. * data_iter_step / len(data_loader):.0f}%)]"
                f"\tTime: {dt:.2f} |"
                f'\tLoss: {loss["loss"].item():.3f} |'
                f'\tL1 loss: {loss["L1_loss"].item():.3f} |'
                f'\tSSIM loss: {loss["ssim_loss"].item():.3f} |'
                f'\tVgg loss: {loss["vgg_loss"].item():.3f} |'
                f'\tBpp loss: {loss["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
            t0 = t1
        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {
            k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()
        }


@torch.no_grad()
def val_one_epoch(
    epoch: int,
    test_dataloader: Iterable,
    model: CompressionModel,
    criterion: torch.nn.Module,
    args=None,
):
    """Valid the model for one epoch and calculate average losses.

    Args:
        epoch (int): The current epoch number.
        test_dataloader (DataLoader): The data loader for test data.
        model (nn.Module): The neural network model to evaluate.
        criterion: The loss function.
        args: Additional arguments or configuration

    Returns:
        dict: A dictionary of averaged statistics for various metrics.
    """
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    L1_loss = AverageMeter()
    ssim_loss = AverageMeter()
    vgg_loss = AverageMeter()
    aux_loss = AverageMeter()

    # Setup device
    device = next(model.parameters()).device

    # Setup logger
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # Switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for samples, _, total_scores in metric_logger.log_every(
            test_dataloader, 10, header
        ):
            samples = samples.to(device)
            total_scores = total_scores.to(device)
            with torch.cuda.amp.autocast():
                out_net = model(samples, total_scores)
                # Compute output
                out_criterion = criterion(out_net, samples)

                # aux_loss.update(model.module.aux_loss())
                aux_loss.update(model.aux_loss())

                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ssim_loss.update(out_criterion["ssim_loss"])
                vgg_loss.update(out_criterion["vgg_loss"])
                L1_loss.update(out_criterion["L1_loss"])

            metric_logger.update(loss=loss.avg)
            metric_logger.update(L1_loss=L1_loss.avg)
            metric_logger.update(ssim_loss=ssim_loss.avg)
            metric_logger.update(vgg_loss=vgg_loss.avg)
            metric_logger.update(bpp_loss=bpp_loss.avg)
            metric_logger.update(aux_loss=aux_loss.avg)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tDevice {device} |"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tL1 loss: {L1_loss.avg:.3f} |"
        f"\tSSIM loss: {ssim_loss.avg:.3f} |"
        f"\tVgg loss: {vgg_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    # Return the averaged statistics
    return {k: round(meter.global_avg, 2) for k, meter in metric_logger.meters.items()}
