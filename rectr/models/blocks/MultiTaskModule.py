""" Defines the MultiTaskModule class. """
import torch
from torch import nn as nn


class MultiTaskModule(nn.Module):
    """ Multi-task module.

    This module is a wrapper for a multi-task network. It is used to
    combine multiple tasks into a single network. Each task has its own
    loss function, and the total loss is computed as a weighted sum of
    the individual task losses.

    If the weights are not provided, the weights of individual tasks are
    automatically learned by using the method proposed in [1].

    [1] Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using
    uncertainty to weigh losses for scene geometry and semantics." Proceedings
    of the IEEE conference on computer vision and pattern recognition. 2018.
    [https://arxiv.org/abs/1705.07115](https://arxiv.org/abs/1705.07115)
    """

    def __init__(self, tasks, losses, weights=None):
        """ Initializer.

        Args:
            tasks (list): A list of tasks.
            losses (list): A list of loss functions for each task. It must have
                           the same length as the tasks list.
            weights (list|None): A list of weights for each task. It must have the
                                 same length as the tasks list. If None, the weights
                                 are automatically learned. Default: None.
        """
        super(MultiTaskModule, self).__init__()
        if len(tasks) != len(losses):
            raise RuntimeError("The number of tasks and loss functions must match.")

        if weights is not None and len(tasks) != len(weights):
            raise RuntimeError("The number of tasks and weights must match.")

        self.tasks = nn.ModuleList(tasks)
        self.sigma = nn.Parameter(torch.ones(len(tasks), requires_grad=True))
        self.losses = losses
        self.weights = weights

    def forward(self, x, targets=None):
        """ Forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, in_channels, height, width)
            targets (list): A list of GT values for each task, or None. If None,
                            no loss is computed.

        Returns:
            tuple: The x Tensor and the total loss.
        """
        if targets is not None and len(self.tasks) != len(targets):
            raise RuntimeError("The number of tasks and target must match.")

        # Pass input through each task to get predictions
        outputs = [f(x) for f in self.tasks]

        loss = None
        if targets is not None:
            # Compute the loss for each task
            loss = [loss_fn(output, target)
                    for target, output, loss_fn in zip(targets, outputs, self.losses)]

            # Compute the weighted sum of the individual task losses
            if self.weights is None:
                # Use the method proposed in [1]
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loss = 0.5 * torch.Tensor(loss).to(device) / self.sigma ** 2
                loss = loss.sum() + torch.log(self.sigma.prod())
            else:
                loss = sum([loss_i * weight for loss_i, weight in zip(loss, self.weights)])

        return outputs, loss
