""" Defines the evaluation metrics for the model. """
import math

import torch
from torchmetrics import MeanAbsoluteError as MAE


def m_n(x, y):
    """ Computes the normal error metric.

    The normal error metric is defined as the mean of the angular distance between the predicted
    and ground truth normals. It calculates the absolute difference between the magnitude of the
    predicted and ground truth normals at each pixel, and then reports thei mean.

    Args:
        x (torch.Tensor): the predicted normal map. Shape (N, 3).
        y (torch.Tensor): the GT normal map. Shape (N, 3).

    Return:
        float: the mean angular error in prediction. This is an angle in degrees, and a lower
               value indicates a better prediction.
        float: the percentage of normals with smaller angular error than 10 degrees.
        float: the percentage of normals with smaller angular error than 20 degrees.
        float: the percentage of normals with smaller angular error than 30 degrees.
    """
    # Compute the angular distance between the predicted and ground truth normals
    criterion = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    similarity = criterion(x, y)
    angles = torch.acos(similarity) * 180 / math.pi

    # Compute the mean angular error
    mae = torch.mean(angles).item()

    # Compute the percentage of pixels with an error less than 10, 20 and 30 degrees
    try:
        p10 = torch.sum(angles < 10).item() / angles.numel() * 100
    except ZeroDivisionError:
        p10 = 0.0
    
    try:
        p20 = torch.sum(angles < 20).item() / angles.numel() * 100
    except ZeroDivisionError:
        p20 = 0.0

    try:
        p30 = torch.sum(angles < 30).item() / angles.numel() * 100
    except ZeroDivisionError:
        p30 = 0.0

    return mae, p10, p20, p30


def m_d(x, y):
    """ Computes the depth error metric.

    The depth error metric is defined as the Frechet Inception Distance (FID) between the
    predicted and ground truth depth maps. The FID is a distance metric between two distributions
    of images. It is computed by comparing the activations of the last convolutional layer of a
    pretrained Inception network for the predicted and ground truth depth maps.

    Args:
        x (torch.Tensor): the predicted depth map. Shape (N, 1).
        target (torch.Tensor): the GT depth map. Shape (N, 1).

    Return:
        float: the mean absolute error in prediction. This is a depth value in meters, and a lower
                value indicates a better prediction.
    """
    fid = MAE().to(x.device)
    mae = fid(x, y).item()
    return mae * 100