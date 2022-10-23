import torch
import torch.nn as nn

from torchvision.models import resnet18

from .SelfAttention import SelfAttention


class ResNetEncoder(nn.Module):
    """ The ResNetEncoder class.

    This encoder is a modified ResNet-18 encoder with self-attention layers added.
    It takes a pre-trained ResNet-18 encoder and adds self-attention layers after
    ach of the residual layers. The final average pooling and fully-connected layers
    are not included, and the max-pooling layer is replaced with a new max-pooling
    layer that returns indices. This gives us an encoder that outputs a feature map
    of shape (N, 512, H/32, W/32).

    ResNetEncoder is based on the 'attentioned' ResNet-18 proposed by [1], and
    the self-attention layers are as described in [2].

    [1] Salvi, A., Gavenski, N., Pooch, E., Tasoniero, F., & Barros, R. (2020, July).
        Attention-based 3D object reconstruction from a single image. In 2020
        International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.
        [https://arxiv.org/abs/2008.04738](https://arxiv.org/abs/2008.04738)
    [2] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
        [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, in_channels=3):
        super().__init__()
        resnet = resnet18(pretrained=True)
        # weights_path = '.cache/pretrained/resnet18-f37072fd.pth'
        # resnet.load_state_dict(torch.load(weights_path))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, stride=1, padding=0),
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.block2 = nn.Sequential(resnet.layer1)
        self.block3 = nn.Sequential(resnet.layer2)
        self.block4 = nn.Sequential(resnet.layer3, SelfAttention(256))
        self.block5 = nn.Sequential(resnet.layer4, SelfAttention(512))

    def forward(self, x):
        """ Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W).

        Returns:
            x (torch.Tensor): Output tensor of shape (N, 512, H/32, W/32).
            indices (torch.Tensor): Indices tensor of shape (N, 512, H/32, W/32).
        """
        x = self.block1(x)
        x, indices = self.pool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x, indices
