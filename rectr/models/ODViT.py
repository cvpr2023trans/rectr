""" Defines the Object Detection ViT network. """

import torch
import torch.nn as nn


from .blocks import ViTEncoder, StandardDecoder, TransformerDecoder


class ODViT(nn.Module):
    """ The Object Detection ViT network.

    This network is made up of a ViT encoder and either a transformer decoder or a standard decoder.
    """
    def __init__(self, in_channels, out_channels, encoder='vit_tiny_patch16_224', decoder='transformer'):
        """ Initialize the OD-ViT network.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels. Default: 1.
            encoder (str, optional): The name of the ViT encoder. Defaults to 'vit_tiny_patch16_224'.
            decoder (str): The type of decoder to use. Either 'transformer' or 'standard'.
        """
        super().__init__()
        # Check that the decoder is valid
        decoder = decoder.lower()
        if decoder not in ['transformer', 'standard']:
            raise Exception('Unknown decoder: {}'.format(decoder))

        if in_channels != 3:
            self._encoder = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0),
                ViTEncoder(name=encoder),
            )
        else:
            self._encoder = ViTEncoder(name=encoder)

        if decoder == 'transformer':
            self._decoder = TransformerDecoder(n_cls=out_channels)
        else:
            self._decoder = StandardDecoder(out_channels=out_channels)

    def forward(self, x):
        """ Forward pass.

        Args:
            x (torch.Tensor): The input Tensor. Shape (N, in_channels, 224, 224).

        Returns:
            Either a tuple of arbitrary length or a single Tensor. In the case of a
            tuple, it contains torch tensors of shape (N, out_channels_i, 224, 224)
            where out_channels_i is the number of output channels for the i-th
            output, and sum of all out_channels_i is equal to out_channels.

            In the case of a single Tensor, it is of shape (N, out_channels, 224, 224).            
        """
        head, patch_embeddings = self._encoder(x)  # (N, 2, 14, 14), (N, 196, 192)
        if isinstance(self._decoder, TransformerDecoder):
            x = self._decoder(patch_embeddings)  # (N, out_channels, 224, 224)
        else:
            x = self._decoder(head)  # (N, out_channels, 224, 224)
        return head, x
