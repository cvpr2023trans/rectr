import torch.nn as nn


class ResNetDecoder(nn.Module):
    """ The ResNetDecoder class.

    The decoder is made up of three AttentionDecoderBlocks, each of which reduces the 
    channel dimension by a factor of 2 and upsamples the input by a factor of 2.
    Then, the decoder branches into two paths: one for the depth map and one for the
    nomals map. Both paths use two transposed convolution layers followed by activation
    and batch normalization, and then a 1x1 convolution layer to reduce the channel
    dimension to 1 and 3 respectively. The final output has a spatial dimension of
    (32xH, 32xW)

    Args:
        activation (nn.Module): The activation function to use. Default: nn.LeakyReLU.
    """

    def __init__(self, out_channels, activation=nn.LeakyReLU):
        super().__init__()
        self.activation = activation
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            activation(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.BatchNorm2d(256),
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            activation(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.BatchNorm2d(128),
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            activation(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.BatchNorm2d(64),
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            activation(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            activation(),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, indices):
        """ Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 512, H/32, W/32).
            indices (torch.Tensor): Indices tensor for the max-unpooling layer.

        Returns:
            depth (torch.Tensor): Output tensor of shape (N, 1, 32xH, 32xW).
            normals (torch.Tensor): Output tensor of shape (N, 3, 32xH, 32xW).
        """
        x = self.block5(x)
        x = self.block4(x)
        x = self.block3(x)
        x = self.unpool(x, indices)
        x = self.block2(x)
        x = self.block1(x)
        return x