""" Defines the Convolution class. """
import torch.nn as nn


class Convolution(nn.Module):
    """ Convolution module.

    A Convolution module consists of a convolution layer, followed by batch normalization and
    a ReLU activation function, i.e. Conv-BN-ReLU. The convolution operation uses SAME padding
    to preserve the spatial dimensions of the input.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        """ Initializer.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of x channels.
            kernel_size (int|tuple): The size of the convolution kernel.
        """
        super().__init__()
        self._layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """ Forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Tensor of shape (batch_size, out_channels, height, width)
        """
        return self._layer(x)
