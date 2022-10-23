# SRVT: Monocular 3D Shape Reconstruction Vision Transformer for Transparent Objects

## Introduction

This is the official implementation of the paper [SRVT: Monocular 3D Shape Reconstruction Vision Transformer for Transparent Objects](#).


## Installation

### Requirements

- Python3 (Tested on 3.8.13 and 3.10.6)
- CUDA (Required for training. Tested on CUDA 11.8)

The following python packages are required:

- PyTorch (`torch>=1.11.0`, `torchvision>=0.12.0`)
- PyTorch Image Models (`timm>=0.6.7`)
- TorchMetrics (`torchmetrics>=0.9.3`)
- Matplotlib (`matplotlib>=3.5.3`)
- Pillow (`Pillow>=9.2.0`)
- OpenCV (`opencv-python>=4.6.0`)
- PyYAML (`PyYAML>=6.0`)

### Install

1. Clone this repository and `cd` into it.

```bash
git clone https://github.com/cvpr2023trans/reconstructor.git
cd reconstructer
```

2. (Recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```


## Usage

### Configuration

Three different types of configuration files are used to define different aspects of the model architecture, the type of 3D shape to be reconstructed, and the training and evaluation parameters. The configuration files are YAML files and are located in the `configs` directory. These are described in more detail below.

#### Model Configuration

The model configuration file defines the architecture of the network and contains a dictionary with the following keys:

```yaml
model:
    use_shortcuts: (bool) Whether to use shortcut connections in the network or not.
    use_transformer_decoder: (bool) Whether to use a transformer decoder or a standard CNN decoder.
```

Sample configuration files are provided in the `configs/models` directory.

#### Data Configuration

The data type configuration file defines the type of 3D shape to be reconstructed and contains a dictionary with the following keys:

```yaml
data:
  in_size: (tuple) The (height, width) of the input image.
  in_channels: (int) The number of channels in the input image.

  out_size: (tuple) The (height, width) of the network output.
  out_channels: (int) The total number of channels in the network output.

  out_channels_split: (tuple) List of integers summimg to `out_channels` that defines the number of channels in each output tensor. This is used to train the network to reconstruct multiple 3D shape representations simultaneously (e.g. depth map, normals map, XYZ coordinates, etc.).
```

Three different types of 3D shapes are supported:

- Depth Map: Set `out_channels_split` to `(1,)` and `out_channels` to `1`.
- Normals Map: Set `out_channels_split` to `(3,)` and `out_channels` to `3`.
- XYZ Coordinates Map: Set `out_channels_split` to `(3,)` and `out_channels` to `3`.

Different combinations of these can be used to train the network for multiple data types. For example, if `out_channels_split` is `(1, 3, 3)` and `out_channels` is 7, then the network will output a depth map, a normals map, and a XYZ coordinates map in that order. Configuration files for all supported combinations are provided in the `configs/data` directory.

#### Training Configuration

The training configuration file defines the training parameters and network hyperparameters and contains a dictionary with the following keys:

```yaml
train:
    batch_size: (int) The batch size to use for training.
    num_workers: (int) The number of workers to use for data loading.
    num_epochs: (int) The number of epochs to train for.
    lr: (float) The learning rate to use for training.
    weight_decay: (float) The weight decay to use for training.

    w_depth: (float) The weight to use for the depth loss.
    w_normals: (float) The weight to use for the normals loss.
    w_xyz: (float) The weight to use for the XYZ coordinates loss.
    w_mask: (float) The weight to use for the mask loss.

    mu: (float) The mu parameter to use for the depth loss.
    kappa: (float) The weight of the angular errors in the normals loss.
    tau: (float) The weight of the vector magnitude errors in the normals loss.
    eta: (float) The influence of the edge loss term in the depth, normals, and XYZ coordinates losses.
```

The `w_depth`, `w_normals` and `w_xzy` terms and the respective loss functions are only used if the corresponding data type is being reconstructed. For example, if the network is only being trained to reconstruct depth maps, then the `w_normals` and `w_xyz` terms are ignored and the corresponding loss functions are not used. The mask loss is always used, unless the `w_mask` term is set to `0`. To omit the edge loss influence from the depth, normals, and XYZ coordinates losses, set the `eta` term to `0`.

Refer to the paper for more details on the loss functions.

Sample configuration files are provided in the `configs/train` directory.

### Training

To train the network, run the following command:

```bash
python train.py --model_config <model_config> --data_config <data_config> --train_config <train_config> --output_dir <output_dir>

```

To see a list of all available command line arguments, run:

```bash
python train.py --help
```
