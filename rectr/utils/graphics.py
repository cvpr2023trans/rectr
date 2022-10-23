import os

import numpy as np
from matplotlib import pyplot as plt
import PIL.Image as Image

from utils.math import rescale


def apply_color_map(gray, cmap='jet'):
    """ Apply a random color map to a grayscale image.

    The seed value can be used to get repeatable results. Set to None to
    generate a random seed.

    Args:
        gray (numpy.array): Grayscale image, shape (H, W).
        cmap (str): Color map to use. Default: 'jet'.

    Returns:
        numpy.array: RGB image, shape (H, W, 3).
    """
    return (plt.get_cmap(cmap)(gray)[:, :, :3] * 255).astype(np.uint8)


def load_image(path, dtype='float32', keep_alpha=False):
    """ Loads an image.

    The image is loaded as a numpy array and converted to the specified
    `dtype`. Only the 'uint8' and 'float32' dtypes are supported. Pixel
    values are in the range [0, 255] for 'uint8' and [0, 1] for 'float32'.

    For RGB images, the alpha channel is only kept if `keep_alpha` is True.
    By default, the alpha channel is removed. 'keep_alpha' is ignored for
    grayscale images.

    Args:
        path (str): Path of the image to load.
        dtype (str): Data type of the image. Only 'uint8' and 'float32' are
                     supported. Default: 'float32'.
        keep_alpha (bool): Whether to keep alpha channel. Only applies for
                           RGB images and alpha is assumed to be 4th channel.
                           Default: False.

    Returns:
        numpy.ndarray: Loaded image, shape (H, W, 3). C is the number of
                       channels (1 for grayscale, 3 for RGB, 4 for RGBA).
    """
    im = np.asarray(Image.open(path))

    # Check dtype of the image.
    if im.dtype not in (np.uint8, np.float32):
        raise Exception('Loaded image {p} has unsupported dtype {dt}. '
                        'load_image() only supports one of (uint8, float32).'.
                        format(p=path, dt=im.dtype))

    # Check dtype argument.
    if dtype not in ('uint8', 'float32'):
        raise Exception('Supported values for "dtype" are ("uint8, float32"), '
                        'got {dt}'.format(dt=dtype))

    # Keep or remove alpha channel.
    if im.ndim == 3 and im.shape[2] == 4 and not keep_alpha:
        im = im[..., :3]

    # Convert data type.
    if dtype == 'float32' and (im.dtype == np.uint8 or im.max() > 1):
        im = im.astype(np.float32) / 255.0
    elif dtype == 'uint8' and (im.dtype == np.float32 or im.max() < 1):
        im = np.round(im * 255.0).astype(np.uint8)

    # Check image shape.
    if len(im.shape) == 2:  # Grayscale image.
        im = im[..., np.newaxis]  # (H, W, 1)

    return im


def save_image(image, path):
    """ Save image to a file

    Args:
        image (numpy.ndarray): The image to save.
        path (str): Path to save the image to.
    """
    plt.imsave(path, image)


def show_image(image, title="Image"):
    """ Show image in a window

    Args:
        image (numpy.ndarray): The image to show.
        title (str): Title of the window.
    """
    plt.imshow(image)
    plt.title(title)
    plt.show()
    plt.waitforbuttonpress(0)
    plt.close()


def create_error_image(image, outputs, targets):
    """ Create an image visualizing the prediction errors.

    Args:
        image (numpy.ndarray): RGB image, shape (H, W, 3).
        outputs (dict): Dictionary containing the outputs of the model.
        targets (dict): Dictionary containing the targets of the model.

    Returns:
        numpy.ndarray: An image visualizing the prediction errors, shape (2H, 3W, 3).
    """

    rows = []

    image = rescale(image, 0, 255, dtype='uint8')
    if 'mask' in outputs and 'mask' in targets:
        mask_pd = outputs['mask'][:, :, 0]  # (H, W)
        mask_gt = targets['mask'][:, :, 0]  # (H, W)

        # Rescale to [0, 255]
        mask_pd = rescale(mask_pd, 0, 255, dtype='uint8')
        mask_gt = rescale(mask_gt, 0, 255, dtype='uint8')

        # Apply color maps for visualization
        mask_pd = apply_color_map(mask_pd, cmap='Blues')  # (H, W, 3)
        mask_gt = apply_color_map(mask_gt, cmap='Blues')  # (H, W, 3)

        rows.append(np.hstack([image, mask_gt, mask_pd]))
    else:
        rows.append(np.hstack([image, image, image]))

    if 'depth' in outputs and 'depth' in targets:
        depth_pd = outputs['depth'][:, :, 0]  # (H, W)
        depth_gt = targets['depth'][:, :, 0]  # (H, W)

        # Compute depth error (pixel-wise absolute difference)
        depth_er = np.abs(depth_pd - depth_gt)  # (H, W)

        # Rescale to [0, 255]
        depth_pd = rescale(depth_pd, 0, 255, dtype='uint8')
        depth_gt = rescale(depth_gt, 0, 255, dtype='uint8')
        depth_er = rescale(depth_er, 0, 255, dtype='uint8')

        # Apply color map for visualization
        depth_pd = apply_color_map(depth_pd, cmap='jet')  # (H, W, 3)
        depth_gt = apply_color_map(depth_gt, cmap='jet')  # (H, W, 3)
        depth_er = apply_color_map(depth_er, cmap='Purples')  # (H, W, 3)

        rows.append(np.hstack([depth_gt, depth_pd, depth_er]))

    if 'normals' in outputs and 'normals' in targets:
        norms_pd = outputs['normals']  # (H, W, 3)
        norms_gt = targets['normals']  # (H, W, 3)

        # Compute normals error (pixel-wise angular difference)
        dot = np.sum(norms_pd * norms_gt, axis=2, keepdims=True)
        norms_pd_ = np.linalg.norm(norms_pd, axis=2, keepdims=True)
        norms_gt_ = np.linalg.norm(norms_gt, axis=2, keepdims=True)
        cos_sim = (dot / (norms_pd_ * norms_gt_ + 1e-8))[:, :, 0] # (H, W)
        norms_er = np.arccos(cos_sim) * 180.0 / np.pi
        norms_er = np.clip(norms_er, 0, 60)  # clip errors above 60 degrees (max error for visualization)
        norms_er = np.nan_to_num(norms_er, nan=60.0)  # replace NaNs with 60 degrees

        # Convert normal maps to BGR (for visualization)
        norms_pd = norms_pd[:, :, ::-1]
        norms_gt = norms_gt[:, :, ::-1]

        # Rescale to [0, 255]
        norms_pd = rescale(norms_pd, 0, 255, dtype='uint8')
        norms_gt = rescale(norms_gt, 0, 255, dtype='uint8')
        norms_er = rescale(norms_er, 0, 255, dtype='uint8')

        # Apply color map for visualization
        norms_er = apply_color_map(norms_er, cmap='Reds')  # (H, W, 3)

        rows.append(np.hstack([norms_gt, norms_pd, norms_er]))

    if 'xyz' in outputs and 'xyz' in targets:
        xyz_pd = outputs['xyz']  # (H, W, 3)
        xyz_gt = targets['xyz']  # (H, W, 3)

        # Compute xyz error (pixel-wise absolute difference)
        xyz_er = np.abs(xyz_pd - xyz_gt)  # (H, W, 3)
        xyz_er = np.sum(xyz_er, axis=2)  # (H, W)

        # Rescale to [0, 255]
        xyz_pd = rescale(xyz_pd, 0, 255, dtype='uint8')
        xyz_gt = rescale(xyz_gt, 0, 255, dtype='uint8')
        xyz_er = rescale(xyz_er, 0, 255, dtype='uint8')

        # Apply color map for visualization
        xyz_er = apply_color_map(xyz_er, cmap='Greens')  # (H, W, 3)

        rows.append(np.hstack([xyz_gt, xyz_pd, xyz_er]))

    return np.vstack(rows)


def create_view(frame):
    """ Show current frame of the RGB-D dataset as images.

    The input frame is expected to be a 4-tuple with the following elements:
    - image (numpy.ndarray): RGB image, shape (H, W, 3).
    - depth (numpy.ndarray): Depth image, shape (H, W).
    - normals (numpy.ndarray): Normals image, shape (H, W, 3).
    - mask (numpy.ndarray): Boolean array with foreground pixels set to True and background to False, shape (H, W).

    Args:
        frame (tuple): 4-tuple containing (image, depth, norms, mask).
    """
    image, depth, norms, mask = frame

    # Rescale all inputs to [0, 255] and convert to uint8
    image = rescale(image, 0, 255, dtype=np.uint8)
    depth = rescale(depth, 0, 255, dtype=np.uint8)
    norms = rescale(norms, 0, 255, dtype=np.uint8)

    # Convert depth maps from (H, W) to (H, W, 3)
    depth = apply_color_map(depth)

    # Mask background pixels
    image[mask] = 0
    depth[mask] = 0
    norms[mask] = 0

    # Create composite image
    fig = np.hstack((image, depth, norms))
    return fig


def prepare_tensors_for_viewing(image, outputs, targets):
    """ Prepare tensors for viewing as images.

    Args:
        image (torch.Tensor): RGB image, shape (N, 3, H, W).
        outputs (dict): Dictionary containing the following keys:
            - depth (torch.Tensor): Predicted depth map, shape (N, 1, H, W).
            - normals (torch.Tensor): Predicted normals, shape (N, 3, H, W).
            - mask (torch.Tensor): Predicted mask, shape (N, 1, H, W).
        targets (dict): Dictionary containing the following keys:
            - depth (torch.Tensor): Ground truth depth map, shape (N, 1, H, W).
            - normals (torch.Tensor): Ground truth normals, shape (N, 3, H, W).
            - mask (torch.Tensor): Ground truth mask, shape (N, 1, H, W).

    Returns:
        torch.Tensor: RGB image, shape (H, W, 3).
        dict: Dictionary containing the following keys:
            - depth (torch.Tensor): Predicted depth map, shape (H, W, 1).
            - normals (torch.Tensor): Predicted normals, shape (H, W, 3).
            - mask (torch.Tensor): Predicted mask, shape (H, W, 1).
        dict: Dictionary containing the following keys:
            - depth (torch.Tensor): Ground truth depth map, shape (H, W, 1).
            - normals (torch.Tensor): Ground truth normals, shape (H, W, 3).
            - mask (torch.Tensor): Ground truth mask, shape (H, W, 1).
    """
    # Get first sample from batch
    image = image[0].cpu().numpy()
    outputs = {k: v[0].detach().cpu().numpy() for k, v in outputs.items()}
    targets = {k: v[0].cpu().numpy() for k, v in targets.items()}

    # Convert channels from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    outputs = {k: np.transpose(v, (1, 2, 0)) for k, v in outputs.items()}
    targets = {k: np.transpose(v, (1, 2, 0)) for k, v in targets.items()}

    return image, outputs, targets


def save_predictions(image, outputs, targets, path):
    """ Save prediction errors as an image.

    Args:
        image (torch.Tensor): RGB image, shape (N, 3, H, W).
        outputs (dict): Dictionary containing the following keys:
            - depth (torch.Tensor): Predicted depth map, shape (N, 1, H, W).
            - normals (torch.Tensor): Predicted normals, shape (N, 3, H, W).
            - mask (torch.Tensor): Mask of pixels to hide, shape (N, 1, H, W).
        targets (dict): Dictionary containing the following keys:
            - depth (torch.Tensor): Ground truth depth map, shape (N, 1, H, W).
            - normals (torch.Tensor): Ground truth normals, shape (N, 3, H, W).
            - mask (torch.Tensor): Mask of pixels to hide, shape (N, 1, H, W).
        path (str): Path to save the image.
    """
    # Prepare tensors for viewing
    image, outputs, targets = prepare_tensors_for_viewing(image, outputs, targets)
    fig = create_error_image(image, outputs, targets)

    fn = os.path.join(os.path.dirname(path), 'predictions.png')

    try:
        diff = np.abs(load_image(fn, dtype='uint8') - fig)
        save_image(diff, os.path.join(os.path.dirname(path), 'predictions_diff.png'))
    except Exception:
        pass

    save_image(fig, fn)


def show_predictions(image, outputs, targets):
    """ Show prediction errors.

    Args:
        image (torch.Tensor): RGB image, shape (N, 3, H, W).
        outputs (dict): Dictionary containing the following keys:
            - depth (torch.Tensor): Predicted depth map, shape (N, 1, H, W).
            - normals (torch.Tensor): Predicted normals, shape (N, 3, H, W).
            - mask (torch.Tensor): Mask of pixels to hide, shape (N, 1, H, W).
        targets (dict): Dictionary containing the following keys:
            - depth (torch.Tensor): Ground truth depth map, shape (N, 1, H, W).
            - normals (torch.Tensor): Ground truth normals, shape (N, 3, H, W).
            - mask (torch.Tensor): Mask of pixels to hide, shape (N, 1, H, W).
    """
    image, outputs, targets = prepare_tensors_for_viewing(image, outputs, targets)
    fig = create_error_image(image, outputs, targets)
    show_image(fig, 'Prediction errors')
