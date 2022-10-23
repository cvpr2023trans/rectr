import math
import operator
import warnings

import torch
import torch.nn as nn

from torchmetrics.functional import image_gradients


def apply_mask(x, mask):
    """ Remove the background pixels from the tensor.

    Args:
        x (torch.Tensor): Tensor to apply the mask to. Shape (B, C, H, W).
        mask (torch.Tensor): A boolean mask of the foreground. Shape (N, 1, H, W)

    Returns:
        torch.Tensor: Tensor with the background pixels removed. Shape (Z, C), Z <= N*H*W
    """
    C = x.shape[1]
    x_ = x.permute((0, 2, 3, 1)).reshape((-1, C))  # Shape: (N*H*W, C)
    mask_ = mask.permute((0, 2, 3, 1)).reshape((-1, 1))  # Shape: (N*H*W, 1)
    if C > 1:
        mask_ = mask_.expand((-1, C))  # Shape: (N*H*W, C)

    # Zero the background pixels
    x_[x_ == 0.0] = 1e-8
    x_[mask_] = 0.0

    # Delete the zero rows
    nonzero = x_.abs().sum(dim=-1).bool()
    x_ = x_[nonzero, :]

    x_[x_ == 1e-8] = 0.0
    return x_


class EarlyStopping:
    def __init__(self, monitor, min_delta, patience, mode, verbose=False):
        """ Initialize the early stopping callback.

        Args:
            monitor: Quantity to be monitored.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement, i.e.
                       an absolute change of less than min_delta, will count as no improvement.
            patience: Number of epochs with no improvement after which training will be stopped.
            mode: One of {"auto", "min", "max"}. In min mode, training will stop when the quantity
                  monitored has stopped decreasing; in "max" mode it will stop when the quantity
                  monitored has stopped increasing; in "auto" mode, the direction is automatically
                  inferred from the name of the monitored quantity.
            verbose: If set to True, prints a message when the early stopping callback is triggered.
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.monitor_op = operator.lt if mode == 'min' else operator.gt
        self.verbose = verbose
        self.best = None
        self.wait = 0
        self.stop_training = False
        self.stopped_epoch = 0

    def on_train_begin(self):
        self.wait = 0
        self.best = None
        self.stop_training = False
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(f"Early stopping requires {self.monitor} to be available for monitoring, "
                          f"skipping.", RuntimeWarning)
            return

        if self.best is None:
            self.best = current
            self.wait = 0
        elif self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"Epoch {epoch}: early stopping.")


class WeighedLoss(nn.Module):
    """ Creates a criterion that sums multiple losses with different weights.

    Args:
        losses (dict): Dictionary of the losses.
        weights (dict): Dictionary of the weights.
    """

    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

        if len(self.losses) != len(self.weights):
            raise ValueError("The number of losses and weights must be equal.")

        if not isinstance(self.losses, dict) or not isinstance(self.weights, dict):
            raise ValueError("The losses and weights must be dictionaries.")

        if set(self.losses.keys()) != set(self.weights.keys()):
            raise ValueError("The losses and weights must have the same keys.")


    def __call__(self, x, y, mask=None):
        """ Computes the loss.

        Args:
            x (dict): Dictionary of the predicted values.
            y (dict): Dictionary of the ground truth values.
            mask (dict): Dictionary of the foreground masks. Default: None.

        Returns:
            nn.Tensor: The computed loss.
        """
        loss = 0.0
        for k, loss_fn in self.losses.items():
            w = self.weights[k]
            if w > 0.0:
                try:
                    x_ = x[k] if isinstance(x, dict) else x
                    y_ = y[k] if isinstance(y, dict) else y
                    mask_ = mask[k] if isinstance(mask, dict) else mask
                    loss += w * loss_fn(x_, y_, mask_)
                except KeyError:
                    loss += 0.0

        return loss


class SilhouetteLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x, y, mask=None):
        """ Compute the silhouette loss.

        Counts the number of foreground pixels predicted as background (i.e. 1)
        and the number of background pixels predicted as foreground (i.e. 0).
        The loss is the sum of average of these quantities over all pixels.

        Args:
            pred: Predicted silhouette. Shape: (B, 1, H, W).
            target: Target silhouette. Must be a binary mask with background = 0 and foreground = 1. Shape: (B, 1, H, W).
            mask: This argument is ignored. It is only present for compatibility with the other losses.

        Returns:
            Silhouette loss.
        """
        s = nn.Sigmoid()
        criterion = nn.BCELoss(reduction='mean')
        loss = criterion(s(x), s(y))
        return torch.reshape(loss, (1,))


class EdgeAwareLoss(nn.Module):
    """ Creates a criterion that measures the error between image edges in the input :math:`x` and target :math:`y`."""

    def __init__(self, use_mask) -> None:
        super().__init__()
        self.use_mask = use_mask

    def __call__(self, x, y, mask=None):
        """ Computes the loss.

        Args:
            x (torch.Tensor): Predicted normal maps, shape (B, 3, H, W).
            y (torch.Tensor): GT normal maps, shape (B, 3, H, W).
            mask (torch.Tensor): Mask for the loss, shape (B, 1, H, W).

        Returns:
            torch.Tensor: Loss value, shape (1,).
        """
        # Compute gradients
        x_dy, x_dx = image_gradients(x)
        y_dy, y_dx = image_gradients(y)

        # Compute MSE loss without reduction
        criterion_edge = nn.MSELoss(reduction="none")
        loss_dx = criterion_edge(x_dx, y_dx)
        loss_dy = criterion_edge(x_dy, y_dy)

        if mask is not None and self.use_mask:
            # Apply mask and reduce
            loss_dx = apply_mask(loss_dx, mask).mean()
            loss_dy = apply_mask(loss_dy, mask).mean()

             # Only keep foreground pixels
            x, y = apply_mask(x, mask), apply_mask(y, mask)
        else:
            loss_dx = loss_dx.mean()
            loss_dy = loss_dy.mean()
        
        # Compute the edge error.
        loss = (loss_dx + loss_dy) / 2.0
        return torch.reshape(loss, (1,))


class AngularLoss(nn.Module):
    """ Creates a criterion that measures the normalized angular distance between each vector in the
    input :math:`x` and target :math:`y`.

    The unreduced (i.e. with `reduction` set to `'none'`) loss can be described as:

    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N \\}^\\top, \\quad l_n = cos^{-1}(\\mathbb{S}(x_n, y_n)) \\times 1 / \\pi

    where :math:`N` is the batch size and :math:`\\mathbb{S}` is the cosine similarity defined as:

    .. math::
        \\mathbb{S} = \\frac{a \\cdot b}{max(||a||_2 \\cdot ||b||_2, \\epsilon)}

    where :math:`\\epsilon` is a small constant to avoid division by zero. If `reduction` is not
    `'none'` (default `'mean'`), then:

    .. math::
        \\ell(x, y) =
        \\begin{cases}
            \\text{mean}(L), &  \\text{if reduction} = \\text{'mean';} \\newline
            \\text{sum}(L),  &  \\text{if reduction} = \\text{'sum'.}
        \\end{cases}

    :math:`x` and :math:`y` are tensors of shape :math:`(3, \\dots)` with a total of :math:`N` elements
    each.

    The division by :math:`N` can be avoided if one sets :math:`reduction = 'sum'`.

    Args:
        eps (float): Small value to avoid division by zero. Default: 1e-8.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """

    def __init__(self, use_mask, eps=1e-8, reduction="mean"):
        """ Initializes the AngularLoss class. """
        super(AngularLoss, self).__init__()
        self.criteria = torch.nn.CosineSimilarity(dim=1, eps=eps)
        self.eps = eps
        self.use_mask = use_mask

        if reduction == "mean":
            self.reduce = torch.mean
        elif reduction == "sum":
            self.reduce = torch.sum
        else:
            self.reduce = torch.nn.Identity()

    def __call__(self, x, y, mask=None):
        """ Computes the loss.

        Args:
            x (torch.Tensor): Predicted Tensor, shape (N, 3, ...).
            y (torch.Tensor): GT Tensor, shape (N, 3, ...).

        Returns:
            loss: Tensor of shape (1,).
        """
        if x.shape != y.shape and x.shape[1] == 3:
            raise ValueError("Prediction and GT must have the same shape.")

        s = self.criteria(x, y)
        s = torch.clamp(s, -0.999, 0.999)  # avoid -1 and 1 in acos

        # Mask the loss if needed
        if mask is not None and self.use_mask:
            s = apply_mask(s, mask)

        loss = self.reduce(torch.acos(s) / math.pi)
        return torch.reshape(loss, (1,))


class LengthLoss(nn.Module):
    """ Creates a criterion that measures the mean squared error between the predicted normal vectors and a unit vector."""

    def __init__(self, use_mask) -> None:
        super().__init__()
        self.use_mask = use_mask

    def __call__(self, x, y, mask=None):
        """ Computes the loss.

        Args:
            x (torch.Tensor): Predicted Tensor, shape (B, C, H, W).
            y (torch.Tensor): GT Tensor, shape (B, C, H, W).
            mask (torch.Tensor): Mask for the loss, shape (B, 1, H, W).

        Returns:
            torch.Tensor: Loss value, shape (1,).
        """
        # Compute the length error.
        criterion = nn.MSELoss(reduction='none')
        x_magnitude = torch.norm(x, dim=1, keepdim=True)
        loss = criterion(x_magnitude, torch.ones_like(x_magnitude))

        # Apply mask (if required) and reduce
        if mask is not None and self.use_mask:
            loss = apply_mask(loss, mask).mean()
        else:
            loss = loss.mean()

        return torch.reshape(loss, (1,))


class NormalLoss(WeighedLoss):
    """ Criterion for surface normals.

    Computes masked per-pixel angular error loss function defined as:

    .. math::
        \mathcal{L}_ang = 1/N * \sum_{i}{\cos^{-1}(n_i * \hat{n}_i /
                         || n_i ||*|| \hat{n}_i ||) / \pi},

        \mathcal{L}_len = 1/N * \sum_{i}{(|| \hat{n}_i ||^2 - 1)^2},

    where ngt_i, np_i is i-th GT and predicted normal vector respectively

    Args:
        kappa (float): Weight of the angular loss.
        tau (float): Weight of the length loss.
        eta (float): Weight of the edge loss.
        use_mask (bool): Whether to use a mask for the loss. Default: False.
    """

    def __init__(self, kappa, tau, eta, use_mask=False):
        WeighedLoss.__init__(self,
                             losses={
                                'angular': AngularLoss(use_mask),
                                'length': LengthLoss(use_mask),
                                'edge': EdgeAwareLoss(use_mask),
                             },
                             weights={
                                'angular': kappa,
                                'length': tau,
                                'edge': eta,
                             })


class MaskedL1Loss(nn.Module):
    """ Creates a criterion that measures the element-wise absolute difference between
    input :math:`x` and target :math:`y` with a mask."""

    def __init__(self, use_mask) -> None:
        super().__init__()
        self.use_mask = use_mask

    def __call__(self, x, y, mask):
        """ Computes the loss.

        Args:
            x (torch.Tensor): Predicted Tensor, shape (B, C, H, W).
            y (torch.Tensor): GT Tensor, shape (B, C, H, W).
            mask (torch.Tensor): Mask for the loss, shape (B, 1, H, W).

        Returns:
            torch.Tensor: Loss value, shape (1,).
        """
        # Compute L1 loss without reduction
        criterion = nn.L1Loss(reduction="none")
        loss = criterion(x, y)

        # Apply mask (if required) and reduce
        if mask is not None and self.use_mask:
            loss = apply_mask(loss, mask).mean()
        else:
            loss = loss.mean()

        return torch.reshape(loss, (1,))


class MaskedMSELoss(nn.Module):
    """ Creates a criterion that measures the element-wise squared difference between
    input :math:`x` and target :math:`y` with a mask."""

    def __init__(self, use_mask) -> None:
        super().__init__()
        self.use_mask = use_mask

    def __call__(self, x, y, mask):
        """ Computes the loss.

        Args:
            x (torch.Tensor): Predicted Tensor, shape (B, C, H, W).
            y (torch.Tensor): GT Tensor, shape (B, C, H, W).
            mask (torch.Tensor): Mask for the loss, shape (B, 1, H, W).

        Returns:
            torch.Tensor: Loss value, shape (1,).
        """
        # Compute MSE loss without reduction
        criterion = nn.MSELoss(reduction="none")
        loss = criterion(x, y)

        # Apply mask (if required) and reduce
        if mask is not None and self.use_mask:
            loss = apply_mask(loss, mask).mean()
        else:
            loss = loss.mean()

        return torch.reshape(loss, (1,))


class DepthLoss(WeighedLoss):
    """ Criterion for depth maps.

    Computes masked per-pixel absolute depth error defined as

    .. math::
        loss = 1/N * \sum_{i}{|dgt_i - dp_i|}

    
    Args:
        mu (float): Weight of the depth loss.
        eta (float): Weight of the edge loss.
        use_mask (bool): Whether to use a mask for the loss. Default: False.
    """

    def __init__(self, mu, eta, use_mask=False):
        """ Creates a DepthLoss instance. """
        WeighedLoss.__init__(self,
                             losses={
                                 'depth': MaskedL1Loss(use_mask),
                                 'edge': EdgeAwareLoss(use_mask),
                             },
                             weights={
                                 'depth': mu,
                                 'edge': eta,
                             })


class XYZLoss(WeighedLoss):
    def __init__(self, mu, eta, use_mask=False):
        """ Creates a XYZLoss instance. """
        WeighedLoss.__init__(self,
                             losses={
                                 'xyz': MaskedMSELoss(use_mask),
                                 'edge': EdgeAwareLoss(use_mask),
                             },
                             weights={
                                 'xyz': mu,
                                 'edge': eta,
                             })

    def __call__(self, x, y, mask=None):
        s = nn.Sigmoid()
        return super().__call__(s(x), s(y), mask)