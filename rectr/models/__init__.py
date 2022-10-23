""" Defines the different network architectures. """
import torch

from .SRVT import SRVT
from .SNMT import SegNetMultiTask as SNMT
from .RUNet import RUNet


def create_model(config, device, parallel=False, verbose=False):
    """ Creates a model.

    Args:
        config (ModelConfig): The model configuration.
        device (torch.device): Device to use.
        parallel (bool): Whether to use data parallelism. Default: False.

    Returns:
        torch.nn.Module: The model.

    Raises:
        ValueError: If the model name is invalid.
    """
    out_channels = 0
    channel_splits = []
    channel_names = []
    if config.output_depth:
        out_channels += 1
        channel_splits.append(1)
        channel_names.append('depth')
    if config.output_normals:
        out_channels += 3
        channel_splits.append(3)
        channel_names.append('normals')
    if config.output_xyz:
        out_channels += 3
        channel_splits.append(3)
        channel_names.append('xyz')

    # Validate output channels
    if out_channels == 0:
        raise ValueError('No output channels specified.')

    if verbose:
        print("{} out channels with {} splits and {} streams.".format(out_channels, channel_splits, channel_names))

    # Create model
    if config.name == 'SNMT':
        model = SNMT()
    elif config.name == 'RUNet':
        model = RUNet(in_channels=config.input_channels, out_channels=out_channels)
    else:
        model = SRVT(
            in_channels=config.input_channels,
            out_channels=out_channels,
            channel_splits=tuple(channel_splits),
            channel_names=tuple(channel_names),
            use_shortcut=config.use_shortcut,
            mask_encoder=config.mask_encoder,
            mask_decoder=config.mask_decoder,
            backbone=config.backbone,
            use_unet=config.use_unet,
            use_attentive_fusion=config.use_attentive_fusion,
        )

    # Enable data parallelism (if requested) and move to device
    model = torch.nn.DataParallel(model) if parallel else model
    model = model.to(device, non_blocking=True)

    return model
