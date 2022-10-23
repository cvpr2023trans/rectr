import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_checkpoint(path, model, device, optim=None):
    """ Loads a checkpoint.

    Args:
        path (str): The path to the checkpoint directory.
        model (torch.nn.Module): The model to load.
        device (torch.device): The device to load the model to.
        optim (torch.optim.Optimizer|None): The optimizer to load. If None, the optimizer
                                            is not loaded. Default: None.

    Returns:
        epoch (int): The epoch number of the loaded checkpoint.
    """
    if path.endswith('.pt'):
        checkpoint_path = path
    elif os.path.isdir(path):
        paths = []
        for c in sorted(os.listdir(path)):
            if c.endswith('.pt'):
                paths.append(os.path.join(path, c))

        checkpoint_path = paths[-1]
    else:
        raise RuntimeError(f"{path} not a checkpoint or a folder")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Restore optimizer state if it exists
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']


def save_checkpoint(path, model, optimizer, epoch):
    """ Saves a checkpoint.

    Args:
        path: The folder where to save the checkpoint.
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: The current epoch number.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(path, f'checkpoint_{epoch}.pt'))


def save_checkpoint_losses(path, loss_tr, loss_va):
    """ Saves the training and validation losses in a log file, including plots.

    Args:
        path (str): The folder where to save the losses.
        loss_tr (float): The training loss at the current epoch.
        loss_va (float): The validation loss at the current epoch.
    """
    train_file = os.path.join(path, 'train_loss.txt')
    val_file = os.path.join(path, 'val_loss.txt')
    try:
        losses_tr = np.loadtxt(train_file).tolist()
        losses_va = np.loadtxt(val_file).tolist()

        if not isinstance(losses_tr, list):
            losses_tr = [losses_tr]

        if not isinstance(losses_va, list):
            losses_va = [losses_va]
    except IOError:
        losses_tr = []
        losses_va = []

    # Append the new losses
    losses_tr.append(loss_tr)
    losses_va.append(loss_va)

    # Convert to numpy arrays and save
    np.savetxt(train_file, np.array(losses_tr))
    np.savetxt(val_file, np.array(losses_va))

    epochs = range(1, len(losses_tr) + 1)
    try:
        # Plot the losses
        plt.plot(epochs, losses_tr, 'g', label='Training loss')
        plt.plot(epochs, losses_va, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(path, 'loss.png'))
        plt.close()
    except Exception as e:
        print('{}\nepochs: {}, train_loss: {}, val_loss: {}'.format(e, epochs, losses_tr, losses_va))
        pass
