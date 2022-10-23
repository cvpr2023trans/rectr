""" Train a model. """
import argparse
import shutil
import warnings
import traceback

from math import ceil
from timeit import default_timer as timer

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary

from configs import Configuration
from data.factory import DatasetFactory
from models import create_model
from optim import EarlyStopping, DepthLoss, NormalLoss, WeighedLoss, SilhouetteLoss, XYZLoss
from saver import *
from utils.graphics import save_predictions
from utils.io import printv

# Enable anomaly detection and disable all other warnings
# torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")


def parse_arguments():
    """ Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a 3D reconstruction network")

    # Configuration files
    parser.add_argument('--model_path', '-m', type=str, default=None, help='Path to the model configuration file.')
    parser.add_argument('--train_path', '-t', type=str, default=None, help='Path to the training configuration file.')
    parser.add_argument('--data_path', '-d', type=str, default=None, help='Path to the dataset configuration file.')
    parser.add_argument('--checkpoint_path', '-c', type=str, default=None, help='Path to the checkpoint directory.')
    parser.add_argument('--output_dir', '-o', type=str, default='checkpoints', help='Path to the output directory.')

    # Training options
    parser.add_argument('--save_interval', '-s', type=int, default=10,
                        help='Number of epochs between checkpoints. Default: 10')
    parser.add_argument('--early_stopping', '-e', type=int, default=0,
                        help='Number of epochs to wait before early stopping. Default: 0 (no early stopping).')
    parser.add_argument('--num_gpus', '-g', type=int, default=0,
                        help='Number of GPUs to use. Set 0 for all available GPUs, -1 for CPU. Default: 0.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Increase output verbosity.')

    # Override options (these values will override the ones in the configuration files)
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay.')

    try:
        args = parser.parse_args()
        
        # Either checkpoint or model, train and data must be provided
        if args.checkpoint_path is None:
            if args.model_path is None or args.train_path is None or args.data_path is None:
                raise ValueError('Either checkpoint or model, train and data must be provided')
    except:
        parser.print_help()
        raise

    return args


def init_device(device, num, verbose=True):
    """ Initialize the device to use.
    
    Args:
        device (str): The device to use (cpu or cuda).
        num (int): The number of GPUs to use (only for device='cuda'). Should be
                   0 or more. If 0 and device='cuda', all available GPUs will be
                   used. The same happens if num is greater than the number of
                   available GPUs.
        verbose (bool): If True, print the device used. Default: True.

    Returns:
        torch.device: The device to use.
        int: The number of devices to use. 1 for CPU, otherwise up to the number
             of available or requested GPUs, whichever is smaller.
    """
    if device == 'cpu':
        return torch.device('cpu'), 1
    else:
        if num < 0:
            raise ValueError('num_gpus must be 0 or more (got {})'.format(num))
        elif num == 0:
            num = torch.cuda.device_count()
        else:
            num = min(num, torch.cuda.device_count())

        if num == 0:
            printv('No GPUs available, using CPU instead', verbose)
            return torch.device('cpu'), 1
        else:
            printv('Using {} GPUs'.format(num), verbose)
            return torch.device('cuda'), num


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
    # if len(targets) != len(outputs):
    #     raise ValueError('Number of targets ({}) and outputs ({}) must match'.format(len(targets), len(outputs)))

    return targets, outputs


def start_training(conf, device, num_devices, save_interval=10, early_stopping=0, verbose=True):
    """ Start a training session.

    Args:
        conf (Configuration): The training configuration.
        device (torch.device): The device to use.
        num_devices (int): The number of GPUs to use. 1 for CPU, otherwise up to the number
                        of available or requested GPUs, whichever is smaller.
        save_interval (int): Number of epochs between checkpoints. Default: 10
        early_stopping (int): Number of epochs to wait before early stopping. Default: 0 (no early stopping).
        verbose (bool): More verbose output. Default: True.
    """
    # Set output path.
    output_path = conf.checkpoint_dir
    if not output_path.endswith('/'):
        output_path += '/'

    printv("Creating {} model...".format(conf.model.name), verbose)
    use_parallel = device.type == 'cuda' and num_devices > 1
    model = create_model(conf.model, device, parallel=use_parallel, verbose=verbose)
    
    # Write summary to file
    try:
        summary_path = os.path.join(output_path, 'model.txt')
        with open(summary_path, 'w') as f:
            print(summary(model, input_size=(128, 3, 224, 224)), file=f)
    except:
        printv('Could not write model summary to file. Skipping', verbose)

    printv("Creating the optimizer...", verbose)
    optimizer = Adam(model.parameters(), lr=conf.train.learning_rate, weight_decay=conf.train.weight_decay)

    # Define the loss functions
    printv('Creating loss functions...', verbose)
    losses = {}
    weights = {}
    streams = ['color']
    if conf.model.output_depth:
        losses['depth'] = DepthLoss(conf.model.mu, conf.model.eta, use_mask=False)
        weights['depth'] = conf.model.w_depth
        streams.append('depth')
    
    if conf.model.output_normals:
        losses['normals'] = NormalLoss(conf.model.kappa, conf.model.eta, conf.model.tau, use_mask=False)
        weights['normals'] = conf.model.w_normals
        streams.append('normals')

    if conf.model.output_xyz:
        losses['xyz'] = XYZLoss(conf.model.mu, conf.model.eta, use_mask=False)
        weights['xyz'] = conf.model.w_xyz
        streams.append('xyz')

    if conf.model.w_mask > 0:
        losses['mask'] = SilhouetteLoss()
        weights['mask'] = conf.model.w_mask
        streams.append('mask')

    printv("Created {} loss functions with weights {}".format(losses.keys(), weights.values()), verbose)
    loss_fn = WeighedLoss(losses, weights)

    # Load saved weights (if given)
    try:
        printv(f"Looking for existing checkpoints in {output_path}...", verbose)
        start_epoch = load_checkpoint(output_path, model, device, optimizer)
        printv(f"Resuming training from epoch # {start_epoch}", verbose)
    except Exception as ex:
        printv(ex, verbose)
        printv("Starting training from scratch...", verbose)
        start_epoch = 1

    printv('Checkpoints every {} epochs'.format(args.save_interval), verbose)
    printv('at {}'.format(output_path), verbose)

    # LR scheduler
    printv("with scheduler: ReduceLROnPlateau(mode='min', min_lr=1e-5, factor=0.5, patience=30, threshold=1e-4).", verbose)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', min_lr=0.00001,
                                  factor=0.5, patience=30,
                                  threshold=0.0001, verbose=verbose)

    # EarlyStopping.
    earlstop = None
    if args.early_stopping > 0:
        printv("and early stopping patience={}".format(args.early_stopping), verbose)
        earlstop = EarlyStopping(
            monitor='loss_va',
            min_delta=0.0001,
            patience=args.early_stopping,
            mode='min'
        )
        earlstop.model = model
        earlstop.on_train_begin()

    # Create data loaders
    batch_size_ = conf.train.batch_size * num_devices
    num_workers_ = conf.data.num_workers * num_devices
    input_size = conf.model.input_size, conf.model.input_size
    dataset_factory = DatasetFactory(batch_size_, num_workers=num_workers_, shape=input_size, verbose=verbose)

    printv("Using data streams: {}".format(streams), verbose)
    printv("Creating training data loader...", verbose)
    loader_tr = dataset_factory.create(conf.data, streams=streams, shuffle=True, mode='train')

    printv("Creating validation data loader...", verbose)
    loader_va = dataset_factory.create(conf.data, streams=streams, shuffle=False, mode='val')

    # Training loop
    epoch = start_epoch
    try:
        runtime = 0
        for epoch in range(start_epoch, conf.train.epochs + 1):
            print(f'#{epoch} \t', end='')
            time_st = timer()

            # Training step
            loss_ep = 0.0
            model.train()  # Set the model to training mode
            for it, batch in enumerate(loader_tr):
                image, targets = get_batch(batch, device)
                for key in targets.keys():
                    if key not in streams:
                        raise RuntimeError("Network output does not contain the {} stream".format(key))

                optimizer.zero_grad()  # Clear the gradients of the network
                outputs = model(image)  # Forward pass
                targets, outputs = align_targets(targets, outputs)  # Validate and align targets and outputs
                loss = loss_fn(x=outputs, y=targets)  # Compute the loss
                loss.backward()  # Backward pass
                loss_ep += loss.item()  # Accumulate loss
                optimizer.step()  # Update the weights

                # Print progress
                print('\r#{} \t tr: {:.4f} ({}/{})'.format(epoch, loss.item(), it, len(loader_tr)),
                      end='', flush=True)

            loss_tr = loss_ep / len(loader_tr)  # epoch_loss = \sum_i (iteration_loss) / # of batches

            # Validation step
            with torch.no_grad():
                loss_ep = 0.0
                model.eval()  # Set the model to validation mode
                for it, batch in enumerate(loader_va):
                    image, targets = get_batch(batch, device)
                    outputs = model(image)  # Forward pass
                    targets, outputs = align_targets(targets, outputs)  # Validate and align targets and outputs
                    loss = loss_fn(x=outputs, y=targets)  # Compute the loss
                    loss_ep += loss.item()  # Accumulate loss

                    # Print progress
                    print('\r#{} \t tr: {:.4f} \t\t va: {:.4f} ({}/{})'
                          .format(epoch, loss_tr, loss.item(), it, len(loader_va)),
                          end='', flush=True)

                loss_va = loss_ep / len(loader_va)  # epoch_loss = \sum_i (iteration_loss) / # of batches

            # Print epoch results
            runtime += (timer() - time_st)
            duration = runtime / (epoch - start_epoch + 1)  # average duration per epoch
            etf = ceil(duration * (conf.train.epochs - epoch))  # estimated time to finish

            if etf > 86400:
                etf = f'{etf // (3600*24)}d {etf // 3600 % 24}h {etf // 60 % 60}m {etf % 60}s'
            elif etf > 3600:
                etf = f'{etf // 3600}h {etf % 3600 // 60}m {etf % 60}s'
            elif etf > 60:
                etf = f'{etf // 60}m {etf % 60}s'
            else:
                etf = f'{etf}s'

            print('\r#{} \t tr: {:.4f} \t\t va: {:.4f} \t\t ends in {}   '
                  .format(epoch, loss_tr, loss_va, etf),
                  end='', flush=True)

            # Save last predictions
            save_predictions(image, outputs, targets, f'{output_path}')

            # Log the training and validation losses
            if epoch > 1:
                save_checkpoint_losses(output_path, loss_tr, loss_va)

            # Save training state
            if epoch % args.save_interval == 0:
                save_checkpoint(output_path, model, optimizer, epoch)

            # Adjust learning rate
            if scheduler is not None:
                scheduler.step(loss_va)

            # EarlyStopping
            if earlstop:
                earlstop.on_epoch_end(epoch, logs={'loss_va': loss_va})
                if earlstop.stop_training:
                    print('\nEarly stopping!')
                    break
    except KeyboardInterrupt:
        print('\nTraining manually interrupted at epoch #{}'.format(epoch))
        if epoch == 1:
            epoch = 0
    except Exception as e:
        print('\nTraining interrupted at epoch #{} due to an error: {}'.format(epoch, e))
        epoch -= 1

        # Print stack trace
        traceback.print_exc()

    # If the training was interrupted on the first epoch (e.g. manually or due to a CUDA error), delete the output folder
    if epoch == 0:
        shutil.rmtree(output_path)
        print('\nOutput folder deleted.')

    else:
        print('\nSaving the final model...')
        save_checkpoint(output_path, model, optimizer, epoch)


if __name__ == "__main__":
    args = parse_arguments()
    device, num_devices = init_device('cpu' if args.num_gpus < 0 else 'cuda', args.num_gpus, args.verbose)

    printv("Loading training configuration...", args.verbose)
    conf = Configuration(args.model_path, args.train_path, args.data_path,
                         args.checkpoint_path, args.output_dir)

    # Override configuration with command line arguments
    if args.batch_size is not None:
        conf.train.batch_size = args.batch_size

    if args.epochs is not None:
        conf.train.epochs = args.epochs

    if args.learning_rate is not None:
        conf.train.learning_rate = args.learning_rate

    if args.weight_decay is not None:
        conf.train.weight_decay = args.weight_decay

    # Start training
    start_training(conf, device, num_devices, args.save_interval,
                   args.early_stopping, args.verbose)
