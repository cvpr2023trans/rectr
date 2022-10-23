""" Defines the SelfAttention class. """
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """ SelfAttention module.

    This module implements the self-attention mechanism described by Vaswani et al. [1].

    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
    information processing systems. 2017.
    [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
    """

    def __init__(self, in_channels):
        """ Initializer.

        Args:
            in_channels (int): The number of input channels.
        """
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """ Forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Tensor of shape (batch_size, in_channels, height, width)
        """
        batch_size, channels, width, height = x.size()

        # Step 1: s_{ij} = f(x)^T·g(x)
        f = self.query(x).view(batch_size, -1, width * height)  # f(x) = B x C x N ( N = W x H )
        g = self.key(x).view(batch_size, -1, width * height)  # g(x) = B x C x N
        s = torch.bmm(f.permute(0, 2, 1), g)  # s_{ij}

        # Step 2: Compute the softmax function β_{j,i} over s_{ij}, which indicates the network
        # attention to the i-th location when synthesizing the j-th region
        attention_map = self.softmax(s)  # β_{j,i} = (B x N x N)

        # Step 3: With the attention map β and the values h(z), now we can compute the self-attention
        # feature maps a = (a_1, a_2, ..., a_N ) ∈ R^{C×N} as v(\sum_{i=1}^N β_{j,i}·h(x))
        h = self.value(x).view(batch_size, -1, width * height)  # h(x) = B x C x N
        a = torch.bmm(h, attention_map.permute(0, 2, 1))  # a = β_{j,i}·h(z_i)
        a = a.view(batch_size, channels, width, height)

        # Step 4: After computing the self-attention feature map a, we perform a normalization
        # operation to compute the final x as y = γ * a + x
        y = self.gamma * a + x
        return y


class FusedAttention(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(FusedAttention, self).__init__()
        self.query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // heads, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // heads, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """ Forward pass.

        Args:
            q (torch.Tensor): Query. Tensor of shape (batch_size, in_channels, height, width)
            k (torch.Tensor): Key. Tensor of shape (batch_size, in_channels, height, width)
            v (torch.Tensor): Value. Tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Tensor of shape (batch_size, in_channels, height, width)
        """
        batch_size, channels, width, height = q.size()

        # Step 1: s_{ij} = f(k)^T·g(q)
        f = self.query(q).view(batch_size, -1, width * height)  # f(q) = B x C x N ( N = W x H )
        g = self.key(k).view(batch_size, -1, width * height)  # g(k) = B x C x N
        s = torch.bmm(f.permute(0, 2, 1), g)  # s_{ij}

        # Step 2: Compute the softmax function β_{j,i} over s_{ij}, which indicates the network
        # attention to the i-th location when synthesizing the j-th region
        attention_map = self.softmax(s)  # β_{j,i} = (B x N x N)

        # Step 3: With the attention map β and the values h(z), now we can compute the self-attention
        # feature maps a = (a_1, a_2, ..., a_N ) ∈ R^{C×N} as v(\sum_{i=1}^N β_{j,i}·h(v))
        h = self.value(v).view(batch_size, -1, width * height)  # h(v) = B x C x N
        a = torch.bmm(h, attention_map.permute(0, 2, 1))  # a = β_{j,i}·h(z_i)
        a = a.view(batch_size, channels, width, height)

        # Step 4: After computing the self-attention feature map a, we perform a normalization
        # operation to compute the final attention as y = γ * a + v
        y = self.gamma * a + v
        return y