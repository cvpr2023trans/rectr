""" Evaluate the performance of a trained model."""
import argparse
import json
import os
import warnings

import numpy as np
import torch

from data.factory import DatasetFactory
from configs import DataConfig, ModelConfig
from metrics import m_n, m_d
from models import create_model
from saver import load_checkpoint
from utils.graphics import show_predictions
from utils.io import ls

warnings.filterwarnings("ignore")


def parse_arguments():
    """ Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")

    # Configuration files
    parser.add_argument('--model', '-m', type=str, default=None, help='Path to the model configuration file.')
    parser.add_argument('--data', '-d', type=str, default=None, help='Path to the test data configuration file.')

    # Options
    parser.add_argument('--show_errors', '-s', action='store_true', help='Whether to visualize errors while evaluating or not. If True, the function will block until the user closes the window.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print progress.')
    parser.add_argument('--parallel', '-p', action='store_true', help='Set if model was trained using data parallelism.')

    return parser.parse_args()


def mask_background(x, mask, flatten=True):
    """ Remove background from a tensor.

    Args:
        x (torch.Tensor): Tensor to remove background from. Shape (B, C, H, W).
        mask (torch.Tensor): Mask. Shape (B, 1, H, W).

    Returns:
        torch.Tensor: Tensor with background removed. Shape (N, C) where N is the
                      number of non-zero pixels in the mask and N <= B*H*W.
    """
    C = x.shape[1]
    if flatten:
        x = x.permute((0, 2, 3, 1)).reshape((-1, C))  # Shape: (B x H x W, C)
        mask = mask.permute((0, 2, 3, 1)).reshape((-1, 1))  # Shape: (B x H x W, 1)
        mask = mask.expand((-1, C))  # Shape: (B x H x W, C)

        EPS = 1e-8
        x[x == 0.0] = EPS  # Temporarily set zeros in x to a small value
        x[mask] = 0.0  # Set background pixels to zero
        x = x[x.abs().sum(dim=-1).bool(), :]  # Delete the zeros
        x[x == EPS] = 0.0  # Set the small values back to zero
    else:
        mask = mask.expand((-1, C, -1, -1))  # Shape: (B, C, H, W)
        x[mask] = 0  # Set background pixels to zero

    return x


def compute_errors(x_d, y_d, x_n, y_n, mask=None):
    """ Compute errors.

    Args:
        x_d (torch.Tensor): Predicted depth. Shape (B, 1, H, W).
        y_d (torch.Tensor): Ground truth depth. Shape (B, 1, H, W).
        x_n (torch.Tensor): Predicted normals. Shape (B, 3, H, W).
        y_n (torch.Tensor): Ground truth normals. Shape (B, 3, H, W).
        mask (torch.Tensor): Mask.

    Returns:
        tuple: Tuple containing (depth_errors, normals_errors).
    """
    # Remove background
    if mask is not None:
        x_d = mask_background(x_d, mask) # Shape: (N, 1)
        y_d = mask_background(y_d, mask) # Shape: (N, 1)
        x_n = mask_background(x_n, mask) # Shape: (N, 3)
        y_n = mask_background(y_n, mask) # Shape: (N, 3)

    # Compute errors
    md = 0
    if x_d is not None and y_d is not None:
        md = m_d(x_d, y_d)

    mae, mae10, mae20, mae30 = 0, 0, 0, 0
    if x_n is not None and y_n is not None:
        mae, mae10, mae20, mae30 = m_n(x_n, y_n)

    return md, (mae, mae10, mae20, mae30)


def get_batch(batch, device):
    """ Get a batch from the data loader.

    Converts the batch to the device and returns it.
    
    Args:
        batch (tuple|dict): The batch from the data loader.
        device (torch.device): The device to use.
    
    Returns:
        tuple: The batch converted to the device.
    """
    if isinstance(batch, tuple):
        image, targets = batch
    else: # dict
        image = batch['color']
        targets = {k: v for k, v in batch.items() if k != 'color'}

    image = image.to(device)
    targets = {k: v.to(device) for k, v in sorted(targets.items())}
    return image, targets


def align_targets(targets, outputs):
    """ Align the targets and outputs.

    Ensures that the targets and outputs have the same number of elements
    and that the elements are in the same order.

    Args:
        targets (dict): The targets.
        outputs (dict): The outputs.
    """
    # Keep only the targets that are in the outputs and validate length
    targets = {k: v for k, v in targets.items() if k in outputs}
    if len(targets) != len(outputs):
        raise ValueError('Number of targets ({}) and outputs ({}) must match'.format(len(targets), len(outputs)))

    return targets, outputs


def eval(model, dataloader, device, show_errors=False, verbose=False):
    """ Evaluate a trained model on a dataset.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader for dataset.
        show_errors (bool): Whether to visualize errors while evaluating or not. If True,
                            the function will block until the user closes the window.
        verbose (bool): Whether to print progress.

    Returns:
        dict: Dictionary containing the errors. Keys are 'mD', 'mAE', 'mAE10', 'mAE20',
              and 'mAE30'.
    """
    errors = {
        'mD': [],
        'mAE': [],
        'mAE10': [],
        'mAE20': [],
        'mAE30': [],
    }

    for batch in dataloader:
        image, targets = get_batch(batch, device)
        outputs = model(image)  # Forward pass

        x_d, y_d = None, None
        if 'depth' in outputs and 'depth' in targets:
            x_d, y_d = outputs['depth'], targets['depth']

        x_n, y_n = None, None
        if 'normals' in outputs and 'normals' in targets:
            x_n, y_n = outputs['normals'], targets['normals']

        # set mask=targets['mask'] to compute for foreground only
        md, (mae, mae10, mae20, mae30) = compute_errors(x_d, y_d, x_n, y_n, mask=None)

        if not np.isnan(md):
            errors['mD'].append(md)

        if not np.isnan(mae):
            errors['mAE'].append(mae)

        if not np.isnan(mae10):
            errors['mAE10'].append(mae10)

        if not np.isnan(mae20):
            errors['mAE20'].append(mae20)

        if not np.isnan(mae30):
            errors['mAE30'].append(mae30)

        if verbose:
            log = '\r{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'
            print(log.format(md, mae, mae10, mae20, mae30), end='')

        if show_errors:  # visualize predictions if show_errors
            show_predictions(image, outputs, targets)

    return errors


def load_models(path, config, device, parallel=False):
    """ Load all saved models from the given path.

    Args:
        path (str): Path to directory containing saved models. May also be a path to
                    a single saved model. If the path is a directory, all saved models
                    in the directory will be loaded.
        config (dict): Configuration dictionary.
        device (torch.device): Device to load models to.

    Returns:
        list: List of models.
    """
    # Get paths to saved models
    if os.path.isdir(path):
        checkpoints = [os.path.join(path, i) for i in ls(path, exts='.pt')]
    else:
        checkpoints = [path]

    # Create models and load saved weights
    models = [create_model(config, device, parallel=parallel) for _ in range(len(checkpoints))]
    for i, model in enumerate(models):
        load_checkpoint(checkpoints[i], model, device)

    return models


def start_eval(args):
    """ Start evaluation.

    Args:
        args: Arguments from command line.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the models to evaluate
    checkpoint_file = args.model
    checkpoint_dir = os.path.dirname(checkpoint_file)
    model_name = checkpoint_dir.split('/')[-1]
    model_config = ModelConfig(os.path.join(checkpoint_dir, 'config.yaml'))
    models = load_models(checkpoint_file, model_config, device, parallel=args.parallel)

    # Create the training dataloaders (one for whole dataset, and one for each object)
    df = DatasetFactory(batch_size=1)
    data_config = DataConfig(args.data)
    dataloader = df.create(data_config, mode='val', shuffle=False)

    for it, model in enumerate(models):
        error_means = {}
        error_stds = {}
        errors = {}
        if args.verbose:
            print(f'model, train, test, mD, mAE, mAE10, mAE20, mAE30')

        errors_i = eval(model, dataloader, device, args.show_errors, args.verbose)

        # Compute mean and standard deviation of errors
        error_mean = {k: np.mean(v) for k, v in errors_i.items()}
        error_std = {k: np.std(v) for k, v in errors_i.items()}
        error_means[it] = error_mean
        error_stds[it] = error_std
        
        print(f'\r{model_name}, proteus, proteus_real', end=', ')
        errors[it] = {}
        for k, (mean, std) in zip(error_mean.keys(), zip(error_mean.values(), error_std.values())):
            if k in ['mD', 'mAE']:
                print(f'{mean:.2f}±{std:.2f}', end=', ')
                errors[it][k] = f'{mean:.2f}±{std:.2f}'
            else:
                print(f'{mean:.2f}', end=', ')
                errors[it][k] = f'{mean:.2f}'
        print()

        # save_errors(args.path, errors, name=f'{args.model}_proteus.json')


def save_errors(path, errors, name='errors.json'):
    """ Save prediction errors.

    Args:
        path (str): Path to model.
        errors (dict): Dictionary containing errors.
    """
    # Create error output directory
    error_dir = os.path.join(os.path.dirname(path), 'eval')
    os.makedirs(error_dir, exist_ok=True)

    # Save dictionary as json file
    error_path = os.path.join(error_dir, name)
    with open(error_path, 'w') as f:
        json.dump(errors, f, indent=4)


if __name__ == '__main__':
    args = parse_arguments()
    start_eval(args)
