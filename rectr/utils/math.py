""" The math utils module. """
import numpy as np
import torch


def rescale(x, min_val=0, max_val=1, dtype=None):
    """" Rescale values to range [min_val, max_val].

    Takes a numpy array or torch.Tensor and rescales it to the range [min_val, max_val].
    When the input is a numpy array, the 'dtype' argument can be used to set the data
    type of the output array.

    If no dtype is specified, the output array will have a np.float32 dtype.

    min_val must always be less than max_val. If min_val is greater than max_val,
    the function will swap the values. If min_val is equal to max_val, the function
    will return an array or Tensor with all values equal to min_val.

    Args:
        x (torch.Tensor|numpy.array): Array or Tensor to be rescaled.
        min_val (number): Minimum value.
        max_val (number): Maximum value.
        dtype (str|Type|None): Data type of the output array. Default: None.

    Returns:
        Rescaled array or Tensor.
    """
    if dtype is None:
        dtype = np.float32

    if min_val > max_val:
        min_val, max_val = max_val, min_val

    if min_val == max_val:
        if isinstance(x, torch.Tensor):
            return torch.full_like(x, min_val, dtype=dtype)

        else:
            return np.full_like(x, min_val, dtype=dtype)

    # Rescale to [min_val, max_val]
    diff = np.nan_to_num(x.max() - x.min())
    if diff == 0:
        diff = 1e-8
    x = (x - x.min()) * (max_val - min_val) / diff + min_val

    # Convert to desired dtype
    if isinstance(x, np.ndarray):
        x = x.astype(dtype)

    return x
