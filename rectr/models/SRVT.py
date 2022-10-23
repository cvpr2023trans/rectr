""" Defines the RSRVT class. """
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResNetEncoder, ResNetDecoder, SelfAttention, FusedAttention
from .ODViT import ODViT


class SRVT(nn.Module):
    """ The Shape Reconstruction Vision Transformer (SRVT) Network.

    SRVT combines the ViT-Tiny architecture with a VAE to reconstruct the 3D shape from
    a single image. The network is trained end-to-end using several reconstruction losses
    and a segmentation loss.
    
    This network has four parts:

    1. The Object Detection ViT Network (OD-ViT) is a ViT-Tiny model with a patch size
    of 16 and spatial image size of (224, 224) followed by either a transformer or a
    CNN decoder. It takes an image with arbitrary number of channels as input and outputs
    a feature map of size (16, 7, 7) and a segmentation mask of size (2, 224, 224)
    where 2 is the number of classes (background and object).
    2. The Mask Extractor (MaskEx) is a 1x1 convolutional layer that takes the output
    of the OD-ViT and outputs a 1-channel mask of the same size.
    3. The Feature Extractor (FeatEx) is a CNN encoder that takes the concatenated
    input and OD-ViT output (5 channels) as input and outputs a latent space
    representation of size (C, 7, 7) where C is 256 for the 'segnet' backbone and 512
    for the 'resnet' backbone.
    4. The Reconstruction Network (RecNet) is a decoder symmetric to FeatEx that takes
    the latent space representation as input and outputs the 3D model as one or more
    tensors of spatial size (224, 224) and specified number of channels, each representing
    a different output (e.g. depth, normals, xyz coordinates, etc.).
    """

    def __init__(self, in_channels=3, out_channels=4, channel_splits=(1, 3), 
                 channel_names=('depth', 'normals'), use_shortcut=True,
                 mask_encoder='vit_tiny_patch16_224', mask_decoder='standard',
                backbone='resnet', use_unet=False, use_attentive_fusion=False):
        """ Initialize the RSRVT network.

        Args:
            in_channels (int): The number of input channels. Default: 3.
            out_channels (int): The number of output channels. Default: 4.
            channel_splits (tuple): The number of channels for individual outputs. Sum of values should be equal to out_channels. Default: (1, 3) for depth and normals. 
            channel_names (tuple): The names of individual outputs. Default: ('depth', 'normals').
            use_shortcut (bool): Whether to use the shortcut connection. Defaults to True.
            mask_encoder (str): The name of the ViT model to use in the OD-ViT encoder. Default: 'vit_tiny_patch16_224'.
            mask_decoder (str): The type of decoder to use. Either 'transformer' or 'standard'. Default: 'standard'.
            backbone (str): The backbone to use in the autoencoder. Either 'resnet' or 'segnet'. Default: 'resnet'.
        """
        super().__init__()
        # Following variables are only for the segnet backbone
        self._kernel = 3
        self._pool_size = (2, 2)
        self._blocks = (2, 2, 3, 3, 3)
        self._filters = (32, 64, 128, 256, 256)

        # Check that sum of channel_splits is equal to out_channels
        if sum(channel_splits) != out_channels:
            raise ValueError(f'Channel splits must sum to {out_channels}')

        # Check that channel_splits and channel_names are of same length
        if len(channel_splits) != len(channel_names):
            raise ValueError(f'Channel splits and channel names must be of same length')

        self.out_channels = out_channels
        self.channel_splits = channel_splits
        self.channel_names = channel_names
        self.use_unet = use_unet
        self.use_attentive_fusion = use_attentive_fusion

        # Check that backbone is valid
        backbone = backbone.lower()
        if backbone not in ['resnet', 'segnet']:
            raise ValueError(f'Invalid backbone: {backbone}. Must be either "resnet" or "segnet"')
        self.backbone = backbone

        # Number of channels in the OD-ViT decoder output (2 for segmentation mask)
        intermed_channels = 2 if mask_decoder == 'transformer' or backbone == 'segnet' else 3
        if use_unet:
            self._odvit = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                         in_channels=in_channels, out_channels=intermed_channels,
                                         init_features=32, pretrained=False)
        else:
            self._odvit = ODViT(in_channels, intermed_channels, encoder=mask_encoder, decoder=mask_decoder)
        self._maskex = nn.Conv2d(intermed_channels, 1, kernel_size=1, stride=1, padding=0)

        # Create the reconstruction network
        self._use_shortcut = use_shortcut
        if backbone == 'resnet':
            self._featex = ResNetEncoder(in_channels=in_channels + intermed_channels)
            self._recnet = ResNetDecoder(out_channels=out_channels)
            self._shortcut = nn.Conv2d(16, 512, kernel_size=1, stride=1, padding=0)  # upsample to 512 channels
        elif backbone == 'segnet':
            self._featex = self._create_encoder(in_channels=in_channels + intermed_channels, act=nn.ReLU)
            self._recnet = self._decoder_fc(out_channels=out_channels, act=nn.ReLU)
            self._shortcut = nn.Conv2d(16, 256, kernel_size=1, stride=1, padding=0)  # upsample to 256 channels

        if use_attentive_fusion:
            self.feats_fusion = FusedAttention(in_channels=512 if backbone == 'resnet' else 256, heads=8)

    @staticmethod
    def conv_bn_act(in_channels, filts, k, activation):
        """ A block involving conv2D, batch normalization and activation
        function.

        Args:
            in_channels (int): Input channels.
            filts (int): # of filters.
            k (int): Conv kernel size.
            act (nn.Module): Activation function.

        Returns:
            nn.Sequential: Conv-BN-Act block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filts, kernel_size=(k, k), padding='same'),
            nn.BatchNorm2d(num_features=filts),
            activation(),
        )

    def _create_encoder(self, in_channels, act):
        """ Builds the encoder.

        Args:
            in_channels (int): Input channels.
            act (nn.Module): Activation function.

        Returns:
            nn.ModuleList: Encoder.
        """
        in_channels_ = in_channels
        modules = []
        for i, (num_blocks, num_filts) in enumerate(zip(self._blocks, self._filters)):
            block = []
            for bi in range(num_blocks):
                block.append(self.conv_bn_act(in_channels_, num_filts, self._kernel, act))
                in_channels_ = num_filts

            block.append(nn.MaxPool2d(self._pool_size, return_indices=True))
            modules.append(nn.Sequential(*block))

        return nn.ModuleList(modules)

    def _decoder_fc(self, out_channels, act):
        """ Decoder using fully convolutional layers (symmetric to `_encoder`).

        Args:
            out_channels (int): # of channels of output tensor.
            act (nn.Module): Activation function.

        Returns:
            nn.ModuleList: Decoder.
        """
        blocks = self._blocks[::-1]
        filters = self._filters[::-1] + (out_channels,)
        stages = len(blocks)

        iters = 1
        modules = []
        in_channels_ = filters[0]
        for si in range(stages):
            modules.append(nn.MaxUnpool2d(self._pool_size))
            block = []
            for bi in range(blocks[si]):
                num_filts = filters[(si, si + 1)[bi == blocks[si] - 1]]
                last_layer = (si == stages - 1 and bi == blocks[si] - 1)
                k = (self._kernel, 1)[last_layer]
                block.append(self.conv_bn_act(in_channels_, num_filts, k, activation=(act, nn.Identity)[last_layer]))
                in_channels_ = num_filts
                iters += 1

            block.append(nn.Identity())
            modules.append(nn.Sequential(*block))

        return nn.ModuleList(modules)

    def forward(self, x):
        """ Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (N, 1, H, W).
            torch.Tensor: Output tensor of shape (N, 3, H, W).
        """
        if self.use_unet:
            ds16 = None
            s = self._odvit(x)  # (N, 2, 224, 224)
        else:
            ds16, s = self._odvit(x)  # OD-ViT -> (N, 16, 7, 7), (N, 2, 224, 224)
        x = torch.cat((x, s), dim=1)  # Concatenate the input and the OD-ViT output

        if self.backbone == 'resnet':
            encoding, encoder_indices = self._featex(x)  # Forward pass through the ResNet encoder
            if self._use_shortcut and ds16 is not None:  # Shortcut connection from OD-ViT to ResNetDecoder (if enabled)
                if self.use_attentive_fusion:
                    encoding = self.feats_fusion(q=self._shortcut(ds16), k=encoding, v=encoding)
                else:
                    encoding = encoding + self._shortcut(ds16)
            out = self._recnet(encoding, encoder_indices)  # Forward pass through the ResNet decoder
        else:
            encoder_indices = []  # Forwards pass through the SegNet encoder
            for encoder_block in self._featex:
                x, indices = encoder_block(x)
                encoder_indices.append(indices)
            encoder_indices = encoder_indices[::-1]
            encoding = x

            if self._use_shortcut and ds16 is not None:  # Shortcut connection from OD-ViT to SegNetDecoder (if enabled)
                if self.use_attentive_fusion:
                    encoding = self.feats_fusion(q=self._shortcut(ds16), k=encoding, v=encoding)
                else:
                    encoding = encoding + self._shortcut(ds16)

            out = encoding  # Forward pass through the SegNet decoder
            for i in range(len(encoder_indices)):
                out = self._recnet[i * 2](out, encoder_indices[i])
                out = self._recnet[i * 2 + 1](out)

        mask = self._maskex(s)  # get the mask from the OD-ViT decoder output

        channels = torch.split(out, self.channel_splits, dim=1) + (mask,)
        channel_names = self.channel_names + ('mask',)
        return {name: channel for name, channel in zip(channel_names, channels)}