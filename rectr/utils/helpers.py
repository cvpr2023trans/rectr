# 3rd party
import numpy as np


def get_mask(img, fgrd=True):
    """ Returns the binary mask where foregournd corresponds to non-zero
    pixels (at least one channel has to be non-zero). `fgrd` flag controls
    whether foreground pixels are True or False.

    Args:
        img (np.array of float): (H, W, 3)-tensor, (H, W)-image of 3D normals.
        fgrd (bool): Whether to return foreground mask.

    Returns:
        np.array of bool: (H, W)-matrix, True - foreground, False - background.
    """

    mask = np.sum(np.abs(img), axis=2).astype(np.bool)

    if not fgrd:
        mask = (1 - mask).astype(np.bool)

    return mask
