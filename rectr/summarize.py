""" Script to print the model summary. """
import argparse

import torch
from torchsummary import summary

from models import create_model


def parse_args():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Name of the model to summarize.')
    return parser.parse_args()


def main(args):
    """ Main function.
    
    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # Create the model
    model = create_model(args.model)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print the model summary
    summary(model, (3, 224, 224))


if __name__ == '__main__':
    args = parse_args()
    main(args)
