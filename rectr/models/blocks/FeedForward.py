""" Implements a simple feed-forward network for use in the transformer decoder. """

import torch.nn as nn


class FeedForward(nn.Module):
    """ Feed-forward network (FFN) with two demse layers and an activation in between.

    This is calculated as: :math:`FFN(x) = activation(xW_1 + b_1)W_2 + b_2`.

    In the original paper [1], the authors used a ReLU activation between the two dense layers,
    however [2] later introduced a GELU activation which became the standard activation function
    in the Transformer-based models like GPT-3 [3] and BERT [4].

    We use the GELU activation by default, but you can also use a ReLU activation by setting the
    `activation` argument to `nn.ReLU`.
    
    References:
        [1] "Attention is All You Need", https://arxiv.org/abs/1706.03762
        [2] "Gaussian Error Linear Units (GELUs)", https://arxiv.org/abs/1606.08415
        [3] "Language Models are Unsupervised Multitask Learners", https://arxiv.org/abs/2005.14165
        [4] "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", https://arxiv.org/abs/1810.04805
    """
    def __init__(self, dim, hidden_dim, dropout, out_dim=None, activation=nn.GELU):
        """ Initializes a new FeedForward instance.

        Args:
            dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            dropout (float): The dropout probability.
            out_dim (int, optional): The output dimension. If not specified, the output dimension
                                     will be the same as the input dimension.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.GELU.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, dim if out_dim is None else out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        """ Performs a forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
