import torch.nn as nn


class StandardDecoder(nn.Module):
    """ The StandardDecoder for upscaling the TransformerEncoder output.

    It takes a Tensor of shape (B, 16, 7, 7) and outputs a Tensor of shape (B, C, 224, 224)
    where C is the number of output channels, using a series of ConvTranspose2d layers. The
    first five layers use BatchNorm2d and GELU activations. The last layer uses a Sigmoid
    activation. The number of output channels can be changed by passing the number of
    channels as an argument.

    Args:
       out_channels (int): The number of channels of the output Tensor. 
    """

    def __init__(self, out_channels) -> None:
        super().__init__()
        self.dc1 = nn.ConvTranspose2d(16, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.at1 = nn.GELU()

        self.dc2 = nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.at2 = nn.GELU()

        self.dc3 = nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.at3 = nn.GELU()

        self.dc4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.at4 = nn.GELU()

        self.dc5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.at5 = nn.GELU()

        self.dc6 = nn.ConvTranspose2d(64, out_channels, 1, stride=1, padding=0)
        self.at6 = nn.Sigmoid()

    def forward(self, x):
        """ Forward pass.
        
        Args:
            x (torch.Tensor): The input Tensor. Shape (B, 16, 7, 7).

        Returns:
            torch.Tensor: The output Tensor. Shape (B, C, 224, 224).
        """
        x = self.at1(self.bn1(self.dc1(x)))
        x = self.at2(self.bn2(self.dc2(x)))
        x = self.at3(self.bn3(self.dc3(x)))
        x = self.at4(self.bn4(self.dc4(x)))
        x = self.at5(self.bn5(self.dc5(x)))
        x = self.at6(self.dc6(x))
        return x

