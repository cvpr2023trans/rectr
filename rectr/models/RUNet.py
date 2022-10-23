# 3rd party.
import torch
import torch.nn as nn

class RUNet(nn.Module):
    """ The Reconstruction UNet (RU-Net) model.
    
    Takes as input a RGB image and outputs a depth map and a normal map.
    Spatial resolution is preserved.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 4.
    """
    def __init__(self, in_channels=3, out_channels=4):
        super(RUNet, self).__init__()
        self._rcn = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=in_channels, out_channels=out_channels,
                                   init_features=32, pretrained=False)

    def forward(self, x):
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            dict: Dictionary containing the output tensors 'depth' and 'normals'.
        """
        x = self._rcn(x)
        out_d = x[:, 0:1, :, :]
        out_n = x[:, 1:, :, :]
        return {'normals': out_n, 'depth': out_d}