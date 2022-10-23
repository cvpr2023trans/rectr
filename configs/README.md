# Configuration Files

This directory contains the configuration files that define the training and
evaluation pipelines for the models in this repository. The configuration files
are written in the [YAML](https://yaml.org/) format.

## Configuration File Parameters

The configuration files each contain a dictionary of parameters that define the training and evaluation pipelines. The parameters are described in the table below. If a parameter is not specified, the default value is used. The required parameters must be specified in the configuration file.

### Model Parameters

| Parameter | Description | Type | Default | Required |
| --------- | ----------- | ---- | ------- | -------- |
| `input_size` | The size of the input images. | `list` | `[3, 224, 224]` | No |
| `kappa` | The kappa value for the model. | `float` | `10.0` | No |
| `w_depth` | Weight of the depth loss. | `float` | `1.0` | No |
| `w_normal` | Weight of the normal loss. | `float` | `1.0` | No |

### Training Parameters

| Parameter | Description | Type | Default | Required |
| --------- | ----------- | ---- | ------- | -------- |
| `batch_size` | The batch size to use during training. | `int` | `32` | No |
| `epochs` | The number of epochs to train for. | `int` | `100` | No |
| `learning_rate` | The learning rate to use during training. | `float` | `0.001` | No |
| `weight_decay` | The weight decay to use during training. | `float` | `0.0001` | No |
| `checkpoint_period` | The number of epochs between checkpoints. | `int` | `10` | No |

### Data Parameters

| Parameter | Description | Type | Default | Required |
| --------- | ----------- | ---- | ------- | -------- |
| `dataset` | The name of the dataset to train on. Available: `bednarik`, `notex`, `notex-real`, `trans` | `str` | `notex` | No |
| `train_path` | Path of the training data. | `str` | - | __Yes__ |
| `train_mult` | Flag indicating if dataset contains one or multiple objects.<sup>&#10013;</sup> | `str` | `true` | No |
| `train_seqs` | The sequences to train on. | `list` | - | No |
| `val_path` | Path of the validation data. | `str` | - | __Yes__ |
| `val_mult` | Flag indicating if dataset contains one or multiple objects. | `str` | `true` | No |
| `val_seqs` | The sequences to validate on. | `list` | - | No |
| `test_path` | Path of the test data. | `str` | - | __Yes__ |
| `test_mult` | Flag indicating if dataset contains one or multiple objects. | `str` | `true` | No |
| `test_seqs` | The sequences to test on. | `list` | - | No |

<sup>&#10013;</sup> If `true`, the dataset path should contain a folder for each object, and each object folder should contain a folder for each sequence. If `false`, the dataset path should contain a folder for each sequence.
