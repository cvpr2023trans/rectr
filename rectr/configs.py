import os
import yaml

from datetime import datetime


class Config:
    """ A YAML configuation file. """

    def __init__(self, path, key=None):
        """ Load a YAML configuration file.
        
        Args:
            path (str): Path to the YAML configuration file.
            key (str): Dictionary key to load from the YAML file. If None, the
                       entire file is loaded. Default: None.
        """
        self.path = path
        self.data = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
        if key is not None:
            if key in self.data:
               self.data = self.data[key]
            else:
                raise ValueError(f'Key "{key}" not found in the configuration file.')


class ModelConfig(Config):
    """ Model configuration. """

    def __init__(self, path):
        """ Load a model configuration file.
        
        Args:
            path (str): Path to the YAML configuration file.
        """
        super().__init__(path, key='model')
        self.name = self.data['name']  # Required

        # Architecture parameters
        self.backbone = self.data.get('backbone', 'resnet')
        self.mask_encoder = self.data.get('mask_encoder', 'vit_tiny_patch16_224')
        self.mask_decoder = self.data.get('mask_decoder', 'transformer')
        self.use_shortcut = self.data.get('use_shortcut', True)
        self.use_unet = self.data.get('use_unet', False)
        self.use_attentive_fusion = self.data.get('use_attentive_fusion', False)

        # Input parameters
        self.input_size = self.data.get('input_size', 224)
        self.input_channels = self.data.get('input_channels', 3)

        # Output parameters
        self.output_size = self.data.get('output_size', 224)
        self.output_depth = self.data.get('output_depth', True)
        self.output_normals = self.data.get('output_normals', True)
        self.output_xyz = self.data.get('output_xyz', False)

        # Hyperparameters
        self.mu = self.data.get('mu', 1.0)
        self.kappa = self.data.get('kappa', 1.0)
        self.tau = self.data.get('tau', 0.1)
        self.eta = self.data.get('eta', 0.5)

        self.w_depth = self.data.get('w_depth', 1.0)
        self.w_normals = self.data.get('w_normals', 1.0)
        self.w_xyz = self.data.get('w_xyz', 1.0)
        self.w_mask = self.data.get('w_mask', 1.0)


class TrainConfig(Config):
    """ Training configuration. """

    def __init__(self, path):
        """ Load a training configuration file.

        Args:
            path (str): Path to the YAML configuration file.
        """
        super().__init__(path, key='train')
        self.batch_size = self.data.get('batch_size', 32)
        self.learning_rate = self.data.get('learning_rate', 1e-3)
        self.weight_decay = self.data.get('weight_decay', 1e-4)
        self.epochs = self.data.get('epochs', 100)


class DataConfig(Config):
    """ Data configuration. """

    def __init__(self, path):
        """ Load a data configuration file.

        Args:
            path (str): Path to the YAML configuration file.
        """
        super().__init__(path, key='data')
        self.dataset = self.data['dataset'] # Required
        self.num_workers = self.data.get('num_workers', 4)
        self.pin_memory = self.data.get('pin_memory', True)

        # Training Set
        self.train_path = self.data.get('train_path', path)
        self.train_seqs = self.data.get('train_seqs', None) # If None, all sequences are used
        self.train_views = self.data.get('train_views', None) # If None, all views are used

        # Validation Set
        self.val_path = self.data.get('val_path', path)
        self.val_seqs = self.data.get('val_seqs', None) # If None, all sequences are used
        self.val_views = self.data.get('val_views', None) # If None, all views are used

        # Test Set
        self.test_path = self.data.get('test_path', path)
        self.test_seqs = self.data.get('test_seqs', None) # If None, all sequences are used
        self.test_views = self.data.get('test_views', None) # If None, all views are used


class Configuration:
    """ Configuration for a training session.

    This class is used to store the configuration for a single training session.
    It contains the model, training, and data configurations. It also contains
    the path to the checkpoint to load, if any.

    Attributes:
        model (ModelConfig): Model configuration.
        train (TrainConfig): Training configuration.
        data (DataConfig): Data configuration.
        session_name (str): Name of the training session.
        checkpoint_dir (str): Path to the directory where the checkpoints are saved.

    Args:
        model_path (str): Path to the model configuration file.
        train_path (str): Path to the training configuration file.
        data_path (str): Path to the data configuration file.
        checkpoint_path (str): Path to the checkpoint directory to load the configuration
                               from. If None, the other three arguments are required.
                               Otherwise, the configuration is loaded from the checkpoint
                               and the other three arguments are ignored. Default: None.
        output_dir (str): Path to the directory where the checkpoints are saved. By default,
                          the checkpoints are saved in the `checkpoints` directory in the
                          current working directory. This is ignored if `checkpoint_path`
                          is not None.
    """

    def __init__(self, model_path, train_path, data_path, checkpoint_path=None, output_dir='checkpoints'):
        if checkpoint_path is None:
            self.model = ModelConfig(model_path)
            self.train = TrainConfig(train_path)
            self.data = DataConfig(data_path)
            self.session_name = self._create_session_name()
            self.checkpoint_dir = self._create_checkpoint_dir(output_dir, self.session_name)

            # Copy the configuration to a single YAML file
            with open(os.path.join(self.checkpoint_dir, 'config.yaml'), 'w') as f:
                yaml.dump(self._to_dict(), f, default_flow_style=False)
        else:
            # Check that the checkpoint path is a directory
            if not os.path.isdir(checkpoint_path):
                raise ValueError('Checkpoint path must be a directory.')

            # Remove the last slash from the checkpoint path (for 'os.path.basename')
            if checkpoint_path[-1] == '/':
                checkpoint_path = checkpoint_path[:-1]

            # Make sure the checkpoint directory contains a configuration file
            checkpoint_config_path = os.path.join(checkpoint_path, 'config.yaml')
            if not os.path.exists(checkpoint_config_path):
                raise ValueError('Checkpoint directory does not contain a config.yaml file.')

            self.model = ModelConfig(checkpoint_config_path)
            self.train = TrainConfig(checkpoint_config_path)
            self.data = DataConfig(checkpoint_config_path)
            self.session_name = os.path.basename(checkpoint_path)
            self.checkpoint_dir = checkpoint_path

    def _to_dict(self):
        """ Get the configuration as a dictionary.

        Returns:
            dict: Dictionary containing all the configuration parameters.
        """
        return {
            'model': self.model.data,
            'train': self.train.data,
            'data': self.data.data,
        }

    def _create_session_name(self):
        """ Create a unique name for the training session.
        
        The name is created by concatenating the model name and the dataset name, followed
        by names of the output streams enabled in the model configuration. These include the
        depth (D), normals (N), and XYZ (X) streams. The name is converted to lowercase and
        spaces are replaced with dashes.
         """
        outs = ''
        if self.model.output_depth:
            outs += 'D'
        if self.model.output_normals:
            outs += 'N'
        if self.model.output_xyz:
            outs += 'X'

        name = f'{outs}-{self.data.dataset}/{self.model.name}'
        return name.lower().replace(' ', '-')

    def _create_checkpoint_dir(self, output_dir, session_name):
        """ Create a unique directory for the training session.

        The name is created by concatenating the output directory and the session name. If
        the directory already exists, a timestamp is appended to the name to make it unique.
        A directory with the same name is created in the output directory and the path to
        this directory is returned.
        """
        name = os.path.join(output_dir, session_name)
        if os.path.exists(name):
            name += f'-{datetime.now().strftime("%Y%m%dT%H%M%S")}'

        os.makedirs(name, exist_ok=True)
        return name