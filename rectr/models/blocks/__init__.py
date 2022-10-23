""" Defines the blocks which are used to build different architectures. """
import torch.nn as nn

from .Attention import Attention
from .Convolution import Convolution
from .FeedForward import FeedForward
from .MultiTaskModule import MultiTaskModule
from .ResNetEncoder import ResNetEncoder
from .ResNetDecoder import ResNetDecoder
from .SelfAttention import SelfAttention, FusedAttention
from .StandardDecoder import StandardDecoder
from .TransformerDecoder import TransformerDecoder
from .TransformerDecoderLayer import TransformerDecoderLayer
from .ViTEncoder import ViTEncoder
