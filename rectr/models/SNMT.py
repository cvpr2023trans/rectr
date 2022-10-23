# 3rd party.
import torch.nn as nn


def activation(act='relu'):
    """ Create activation function.
    Args:
        act (str): Activation function name.
    Returns:
        nn.Module: Activation function.
    """
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leaky_relu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'linear':
        return nn.Identity()
    else:
        raise Exception('Unknown activation function: {}'.format(act))


class SegNetMultiTask(nn.Module):
    """ Architecture based on SegNet [1] with original encoder and up to two
    decoders producing normal maps (DNM) and depth maps (DDM). The DNM and DDM
    are symmetric to the encoder since they produce the tensors of the same
    spatial size as the input.
    [1] V Badrinarayanan et. al. SegNet: A Deep Convolutional Encoder-Decoder
    Architecture for Image Segmentation. TPAMI 2017.
    Args:
        input_shape (tuple of int): Input image size of shape (H, W, C).
        kernel (int): Conv kernel size.
        pool_size (tuple): Pooling window size.
        blocks (tuple): # of conv-BN-relu blocks at each stage.
        filters (tuple): # of conv filters used at each stage.
    """

    def __init__(self, input_shape=(224, 224, 3), kernel=3, pool_size=(2, 2),
                 blocks=(2, 2, 3, 3, 3), filters=(32, 64, 128, 256, 256), act='relu'):
        super(SegNetMultiTask, self).__init__()
        self._input_shape = input_shape
        self._kernel = kernel
        self._pool_size = pool_size
        self._blocks = blocks
        self._filters = filters

        if len(blocks) != len(filters):
            raise Exception('"blocks" and "filters" must have the same number '
                            'of items, found {} != {}'.
                            format(len(blocks), len(filters)))

        self.encoder = self._encoder(in_channels=input_shape[-1], act=act)
        self.decoder_n = self._decoder_fc(3, act=act)
        self.decoder_d = self._decoder_fc(1, act=act)

    @staticmethod
    def conv_bn_act(in_channels, filts, k, act):
        """ A block involving conv2D, batch normalization and activation
        function.
        Args:
            in_channels (int): Input channels.
            filts (int): # of filters.
            k (int): Conv kernel size.
            act (str): Activation function.
        Returns:
            nn.Sequential: Conv-BN-Act block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filts, kernel_size=(k, k), padding='same'),
            nn.BatchNorm2d(num_features=filts),
            activation(act),
        )

    def _encoder(self, in_channels, act):
        """ Builds the encoder.
        Args:
            in_channels (int): Input channels.
            act (str): Activation function.
        Returns:
            nn.ModuleList: Encoder.
        """
        in_channels_ = in_channels
        modules = []
        for num_blocks, num_filts in zip(self._blocks, self._filters):
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
            act (str): Activation function.
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
                block.append(self.conv_bn_act(in_channels_, num_filts, k, act=(act, 'linear')[last_layer]))
                in_channels_ = num_filts
                iters += 1

            block.append(activation('linear'))
            modules.append(nn.Sequential(*block))

        return nn.ModuleList(modules)

    def freeze_encoder(self, freeze=True):
        """ Freezes the encoder parameters.
        Args:
            freeze (bool): Whether to freeze or unfreeze the encoder. Default: True.
        """
        for module in self.encoder:
            for param in module.parameters():
                param.requires_grad = not freeze

    def freeze_normals(self, freeze=True):
        """ Freezes the normals decoder parameters.
        Args:
            freeze (bool): Whether to freeze or unfreeze the decoder. Default: True.
        """
        for module in self.decoder_n:
            for param in module.parameters():
                param.requires_grad = not freeze

    def freeze_depth(self, freeze=True):
        """ Freezes the depth decoder parameters.
        Args:
            freeze (bool): Whether to freeze or unfreeze the decoder. Default: True.
        """
        for module in self.decoder_d:
            for param in module.parameters():
                param.requires_grad = not freeze

    def forward(self, x):
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output of the normals decoder.
            torch.Tensor: Output of the depth decoder.
        """
        encoder_indices = []
        for encoder_block in self.encoder:
            x, indices = encoder_block(x)
            encoder_indices.append(indices)
        encoder_indices = encoder_indices[::-1]

        out_n = x
        out_d = x
        for i in range(len(encoder_indices)):
            out_n = self.decoder_n[i * 2](out_n, encoder_indices[i])
            out_n = self.decoder_n[i * 2 + 1](out_n)

            out_d = self.decoder_d[i * 2](out_d, encoder_indices[i])
            out_d = self.decoder_d[i * 2 + 1](out_d)

        return {'normals': out_n, 'depth': out_d}