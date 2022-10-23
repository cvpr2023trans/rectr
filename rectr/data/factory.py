from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from utils.io import printv
from ._datasets import TransparentDataset, TransProteus, TransProteusReal


class DatasetFactory:
    """ Factory class for creating data loaders. """

    def __init__(self, batch_size=1, num_workers=0, shape=(224, 224), verbose=False):
        """ Initialize the factory.

        Args:
            batch_size (int): Number of samples per batch. Default: 1.
            num_workers (int): Number of subprocesses to use for data loading. 0 means that
                               the data will be loaded in the main process. Default: 0.
            shape (tuple): The shape to resize the loaded samples to. Default: (224, 224)
            verbose (bool): If True, print out the progress of the data loading. Default: False.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shape = shape
        self.verbose = verbose

        printv('[DatasetFactory] batch_size={}, num_workers={}, shape={}'.format(batch_size, num_workers, shape), verbose)

    def create(self, config, streams=None, shuffle=False, mode='train'):
        """ Creates a data loader.

        Args:
            config (DataConfig): The data configuration.
            streams (list): A list of streams to load. If None, all streams are loaded.
            shuffle (bool): Set to True to have the data reshuffled at every epoch. Default: False.
            mode (str): The mode to create the data loader for. Possible values are 'train', 'val' and 'test'.
                        Default: 'train'.
        """
        name = config.dataset
        if mode == 'train':
            path = config.train_path
            seqs = config.train_seqs
            views = config.train_views
        elif mode == 'val':
            path = config.val_path
            seqs = config.val_seqs
            views = config.val_views
        elif mode == 'test':
            path = config.test_path
            seqs = config.test_seqs
            views = config.test_views

        printv(f"Reading \'{name}\' dataset from {path}..", self.verbose)
        if name == "trans":
            data = TransparentDataset(path, single_object=False, envs=seqs, seqs=views, shape=self.shape)
        elif name == "proteus":
            parts = []
            for s in seqs:
                parts += [TransProteus(path.format(v, s), streams, shape=self.shape) for v in views]

            data = ConcatDataset(parts)
        elif name == "proteus_real":
            data = TransProteusReal(path, streams, shape=self.shape)
        else:
            raise ValueError("Unsupported dataset")

        printv(f"Found {len(data)} samples.", self.verbose)

        # Create and return the data loader
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True)


if __name__ == '__main__':
    import argparse
    from configs import DataConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the dataset.')

    args = parser.parse_args()

    # Load the data configuration
    config = DataConfig(args.config)

    # Create dataset
    dataset = DatasetFactory(32, 0).create(config, streams=['depth'], shuffle=True, mode='test')

    # Print some statistics
    print('Dataset size: {}'.format(len(dataset)))
    print('Number of sequences: {}'.format(len(dataset.sequences)))
    print('Number of frames: {}'.format(sum(len(s) for s in dataset.sequences)))
    print('Number of cameras: {}'.format(len(dataset.cameras)))
    print('Number of objects: {}'.format(len(dataset.objects)))