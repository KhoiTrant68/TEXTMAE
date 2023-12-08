import torch


def de_normalize(batch):
    """
    De-normalize a batch of images for regular normalization.

    Args:
        batch (torch.Tensor): Input batch of images.

    Returns:
        batch (torch.Tensor): De-normalized batch of images.
    """

    batch = (batch + 1.0) / 2.0 * 255.0
    return batch


def normalize_batch(batch):
    """
    Normalize a batch of images using ImageNet mean and std.

    Args:
        batch (torch.Tensor): Input batch of images.

    Returns:
        batch (torch.Tensor): Normalized batch of images.
    """
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= mean
    batch = batch / std
    return batch
