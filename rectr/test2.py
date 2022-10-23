import argparse
import os

import numpy as np
import torch

from models import create_model
from saver import load_checkpoint
from .utils.graphics import load_image, save_image, apply_color_map

from torchvision.transforms import ToTensor, Resize, Compose


# hide torch warnings
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SRVT', help='name of the model to use')
    parser.add_argument('--path', '-p', type=str, help='path of the trained model')
    parser.add_argument('--image', '-i', type=str, help='path of the images to test')
    parser.add_argument('--use_cuda', '-c', action='store_true', help='use cuda')
    args = parser.parse_args()
    test_model(args)


def load_model(model_name, model_path):
    try:
        model = create_model(model_name, parallel=True)
        load_checkpoint(model_path, model)
    except:
        model = create_model(model_name, parallel=False)
        load_checkpoint(model_path, model)
    model.eval()
    return model


def load_input(img_path):
    im = load_image(img_path)
    h, w, _ = im.shape
    im_norm = Compose([ToTensor(), Resize((224, 224))])(im).unsqueeze(0)
    return im, im_norm, (h, w)


def test_model(args):
    # load model
    model = load_model(args.model, args.path)
    model = model.cuda() if args.use_cuda else model.cpu()

    # use all .jpg, .png and .tiff files in images folder
    paths = sorted([os.path.join(args.image, f) for f in os.listdir(args.image) if f.endswith(('.jpg', '.png', '.tiff'))])

    # load images
    originals, im_norms, sizes = zip(*[load_input(p) for p in paths])

    # load in './masks' subdirectory for masks if available
    masks = [None] * len(paths)
    for i, p in enumerate(paths):
        mask_path = os.path.join(os.path.dirname(p), 'masks', os.path.basename(p))
        mask_path = os.path.splitext(mask_path)[0] + '.png'
        if os.path.exists(mask_path):
            masks[i] = load_input(mask_path)[1][:, 0, :, :]  # (1, h, w)

    # get results
    results = []
    for image, im_norm, size, mask in zip(originals, im_norms, sizes, masks):
        im_norm = im_norm.cuda() if args.use_cuda else im_norm.cpu()
        
        # get predictions and resize them to original image size
        resize = Resize((224,224))
        if 'SRVT' in args.model:
            depth, normals, mask  = model(im_norm)
            depth, normals, mask = resize(depth), resize(normals), resize(mask)[:, 0, :, :]  # (1, h, w)
        else:
            depth, normals = model(im_norm)
            if mask is None:
                mask = (im_norm == 0).type(torch.FloatTensor).mean(dim=1)  # (1, h, w)       

        # convert to numpy
        depth = depth.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        normals = normals.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        mask = mask.repeat(3, 1, 1).detach().cpu().numpy().transpose(1, 2, 0)
        image = im_norm.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            
        if 'SRVT' not in args.model:
            bool_mask = (mask != 0)[:, :, 0]
            depth[bool_mask, :] = depth.max()
            normals[bool_mask, :] = 0
            normals[bool_mask, 2] = 1

        # rescale to [0,255] and convert to uint8
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        normals = ((normals - normals.min()) / (normals.max() - normals.min()) * 255).astype(np.uint8)
        # mask = ((mask - mask.min()) / (mask.max() - mask.min() + 1e-8) * 255).astype(np.uint8)

        depth = apply_color_map(depth[:, :, 0], 'jet')  # colormap on depth to get better visualization
        results.append(np.vstack((image, depth, normals[:, :, ::-1])))

    # save results
    model_name = os.path.split(args.path)[-1].split('.')[0]
    outfile = os.path.join(os.path.relpath(args.image), f'{model_name}/{args.model}.png')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    save_image(np.hstack(results), outfile)


if __name__ == '__main__':
    main()