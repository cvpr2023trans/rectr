import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tf

from utils.depth import depth2xyz
from utils.graphics import load_image
from utils.io import ls


class TransparentDataset(Dataset):
    """Dataset class for loading the transparent data from memory."""

    def __init__(self, path, single_object=False, envs=None, seqs=None, shape=(224, 224)):
        """
        Args:
            path (string): Path to the dataset.
            single_object (bool): If True, only load data from one object. If False, 'path' is the path to the
                                  directory containing all objects. Default: False.
            envs (list): List of world environments to load. If None, load all environments. Default: None.
            seqs (list): List of sequences to load. If None, load all sequences. Default: None.
        """
        self.images = []
        self.labels = []
        self._transform = tf.Compose([
            tf.ToTensor(),
            tf.Resize(shape),
        ])

        if single_object:
            objects = [path]
        else:
            objects = [os.path.join(path, o) for o in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, o))]

        for o in objects:
            sequences = [os.path.join(o, s) for s in sorted(os.listdir(o)) if
                         os.path.isdir(os.path.join(o, s)) and (seqs is None or s in seqs)]
            for s in sequences:
                images_path = os.path.join(s, 'images')
                images = ls(images_path, '.png')

                if envs is not None:
                    # Keep only the specified environments
                    images = [i for i in images if i.split('/')[-1].split('_')[-1].split('.')[0] in envs]

                labels_path = s
                labels = []
                for y in ls(labels_path, '.npz'):
                    labels += [y] * (len(envs) if envs is not None else 5)

                if len(images) == len(labels):
                    self.images += [os.path.join(images_path, x) for x in images]
                    self.labels += [os.path.join(labels_path, y) for y in labels]

    def __len__(self):
        """Return the size of dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get the item at index idx."""

        # Get the image
        image = load_image(self.images[idx])  # (H, W, C)

        # Get the ground truth
        depth = np.load(self.labels[idx])['dmap'].astype(np.float32) # (H, W, 1)
        norms = np.load(self.labels[idx])['nmap'].astype(np.float32) # (H, W, 3)
        norms = self.rotate_x(norms, 90 - int(self.images[idx].split('/')[-3]))
        norms = norms[..., [2, 1, 0]] # Swap the axes
        norms[:, :, 0] = -norms[:, :, 0] # Flip the x-axis
        norms[:, :, 2] = -norms[:, :, 2] # Flip the z-axis

        # Get background mask of shape (H, W) as zero-pixels in the image
        mask = depth >= 1

        # Mask out the background (only keep object)
        # image[mask, :] = 0
        depth[mask] = 1

        # Set background normals to vector pointing away from camera
        norms[mask, :] = 0
        norms[mask, 2] = 1

        # Change mask shape to (1, H, W)
        mask = mask[..., np.newaxis]

        # Apply the transform, if specified
        image = self._transform(image)  # (C, H, W)
        depth = self._transform(depth)  # (1, H, W)
        norms = self._transform(norms)  # (3, H, W)
        mask = self._transform(mask)  # (1, H, W)

        # Remove any NaNs
        depth[torch.isnan(depth)] = 1
        norms[torch.isnan(norms)] = 0

        # Return the data and label
        return image, {'depth': depth, 'normals': norms, 'mask': mask}

    def rotate_x(self, norms, angle):
        """ Rotate normals around x-axis by the given angle. """
        angle = np.radians(angle)
        c, s = np.cos(angle), np.sin(angle)
        x, y, z = norms[:, :, 0], norms[:, :, 1], norms[:, :, 2]
        norms[:, :, 0] = x
        norms[:, :, 1] = y * c - z * s
        norms[:, :, 2] = y * s + z * c
        return norms


class TransProteus(Dataset):
    """ Dataset class for loading the TransProteus [1] data from memory.

    [1] Eppel, S., Xu, H., Wang, Y. R., & Aspuru-Guzik, A. (2021). Predicting 3D shapes,
        masks, and properties of materials, liquids, and objects inside transparent
        containers, using the TransProteus CGI dataset. arXiv preprint arXiv:2109.07577.

    Args:
        path (string): Path to the dataset.
        streams (list): List of streams to load. If None, load all streams except 'xyz'.
                        Available streams: 'color', 'depth', 'normals', 'mask' and 'xyz'.
                        Default: None.
        shape (tuple): Shape of the image to load. Default: (224, 224).
    """

    def __init__(self, path, streams=None, shape=(224, 224)):
        self._data = []  # List of dicts with paths to requested streams for each sample
        self._streams = streams if streams is not None else ['color', 'depth', 'normals', 'mask']
        self._stream_channels = {
            "color": 3,
            "normals": 3,
            "depth": 1,
            "xyz": 3,
            "mask": 1,
        }
        # Check that all requested streams are valid
        for s in self._streams:
            if s.lower() not in self._stream_channels:
                raise ValueError(f"Invalid stream '{s}'")

        self._transform = tf.Compose([
            tf.ToTensor(),
            tf.Resize(shape),
        ])

        # Stream Mappings
        # camera_intrinsics: CameraParameters.json
        # color -> VesselWithContentRGB.jpg
        # depth -> VesselWithContentDepth.exr
        # normals -> VesselWithContentNormal.exr
        # mask -> VesselMask.png
        # xyz -> VesselWithContentXYZ.npy
        for sample in os.listdir(path):
            streams = {}
            sample_dir = os.path.join(path, sample)
            color_fn, depth_fn, normals_fn, xyz_fn = None, None, None, None
            for s in os.listdir(sample_dir):
                if 'VesselWithContent' in s and '_RGB' in s:
                    color_fn = s
                    depth_fn = s.replace('_RGB.jpg', '_Depth.exr')
                    normals_fn = s.replace('_RGB.jpg', '_Normal.exr')
                    xyz_fn = s.replace('_RGB.jpg', '_XYZ.npy')
                    break  # Take the first frame we find

            # Skip if can't find paths
            if color_fn is None:
                continue

            # Load the camera intrinsics
            camera_path = os.path.join(path, sample, 'CameraParameters.json')
            if os.path.isfile(camera_path):
                with open(camera_path) as f:
                    K = json.load(f)

                    if K is None and 'xyz' in self._streams:
                        continue

                    streams['K'] = K

            for stream in self._streams:
                stream = stream.lower()
                if stream == 'xyz':
                    xyz_path = os.path.join(path, sample, xyz_fn)
                    # if not os.path.exists(xyz_path):
                    #     # XYZ stream is not available in the dataset, so we need to compute it
                    #     # from the depth map and camera intrinsics. We will compute it once and
                    #     # save it to disk, so that we don't have to compute it every time we load
                    #     # a sample.
                    #     # Load the camera intrinsics
                    #     camera_path = os.path.join(path, sample, 'CameraParameters.json')
                    #     if os.path.isfile(camera_path):
                    #         with open(camera_path) as f:
                    #             K = json.load(f)

                    #         # Skip stream if camera intrinsics are not available
                    #         if K is None:
                    #             continue

                    #         # Load the depth map and compute the XYZ map
                    #         if depth_fn is not None:
                    #             depth_path = os.path.join(path, sample, depth_fn)
                    #             if os.path.isfile(depth_path):
                    #                 depth = self.__load_data(depth_path, num_channels=1)
                    #                 xyz = depth2xyz(depth, K)
                    #                 np.save(xyz_path, xyz)  # Save the XYZ map to disk
                    
                    # Add the path to the XYZ map to the streams dict
                    streams[stream] = xyz_path
                elif stream == 'color':
                    streams[stream] = os.path.join(path, sample, color_fn)
                elif stream == 'depth':
                    streams[stream] = os.path.join(path, sample, depth_fn)
                elif stream == 'normals':
                    streams[stream] = os.path.join(path, sample, normals_fn)
                elif stream == 'mask':
                    streams[stream] = os.path.join(path, sample, 'VesselMask.png')

            # If all requested streams are available, add the sample to the dataset
            use_sample = True
            for stream in streams.keys():
                if stream == 'K':
                    continue

                if stream != 'xyz' and not os.path.isfile(streams[stream]):
                    use_sample = False
                    break

            if use_sample:
                self._data.append(streams)

    def __len__(self):
        """ Return the size of dataset. """
        return len(self._data)

    def __load_data(self, path, num_channels, dtype=np.float32):
        if ".exr" in path:
            i = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if np.ndim(i) >= 3 and num_channels == 1:
                i = i[:, :, 0]
        elif ".npy" in path:
            i = np.load(path)
        else:
            i = cv2.imread(path, 0) if num_channels == 1 else cv2.imread(path)

        # Make sure the image is in the correct format
        if i.dtype != dtype:
            i = i.astype(dtype)

        return i.astype(dtype)

    def __getitem__(self, index):
        """ Load a sample of data. """
        paths = self._data[index]

        # Load the requested streams
        data = {}
        for stream in self._streams:
            stream = stream.lower()
            path = paths[stream]
            num_channels = self._stream_channels[stream]

            if stream == 'xyz':
                if not os.path.isfile(path):
                    K = paths['K']
                    depth_path = paths['depth']
                    depth = self.__load_data(depth_path, num_channels=1)
                    xyz = depth2xyz(depth, K)
                    np.save(path, xyz)  # Save the XYZ map to disk
                    data[stream] = xyz
                else:
                    data[stream] = self.__load_data(path, num_channels, dtype=np.float32)

            data[stream] = self.__load_data(path, num_channels, dtype=np.float32)

        # Preprocess mask
        mask = None
        if 'mask' in data:
            mask = data["mask"]
            mask[mask > 0] = 1  # Convert masks to 0/1
            data["mask"] = mask

        # Preprocess depth
        if "depth" in data:
            depth = data["depth"]
            max_depth = 5000
            depth[depth > max_depth] = max_depth  # Remove depth values that are too far away

            # Apply mask (if available) and normalize depth
            if mask is not None:
                depth[mask == 0] = 1         # 1 for background
                depth_fg = depth[mask == 1]  # [0, 1) for foreground
                if depth_fg.size > 0:
                    depth[mask == 1] = (depth_fg - depth_fg.min()) / (depth_fg.max() - depth_fg.min())
            else:
                depth = depth / max_depth    # [0, 1] for all pixels

            depth[np.isnan(depth)] = 1       # Remove NaN values
            data["depth"] = depth
        
        # Preprocess normals
        if "normals" in data:
            norms = data['normals']
            norms = -norms[:, :, [2, 1, 0]]  # Flip normals axes to match our convention

            # Apply mask (if available)
            if mask is not None:
                norms[mask == 0, :] = 0      # Keep only the vessel ground-truth
                norms[mask == 0, 2] = 1      # (i.e., no ground plane and/or background)

            norms[np.isnan(norms)] = 0       # Remove NaN values
            data["normals"] = norms

        if "xyz" in data:
            xyz = data["xyz"]

            # Apply mask (if available)
            if xyz is not None:
                xyz[mask == 0, :] = 0      # Keep only the vessel ground-truth

            xyz[np.isnan(xyz)] = 0           # Remove NaN values
            data["xyz"] = xyz

        # Apply the transform, if specified
        if self._transform is not None:
            for stream in data:
                data[stream] = self._transform(data[stream])

        return data


class TransProteusReal(Dataset):
    """ Dataset class for loading the TransProteus [1] data from memory.

    [1] Eppel, S., Xu, H., Wang, Y. R., & Aspuru-Guzik, A. (2021). Predicting 3D shapes,
        masks, and properties of materials, liquids, and objects inside transparent
        containers, using the TransProteus CGI dataset. arXiv preprint arXiv:2109.07577.

    Args:
        path (string): Path to the dataset.
        streams (list): List of streams to load. If None, load all streams except 'xyz'.
                        Available streams: 'color', 'depth', 'normals', 'mask' and 'xyz'.
                        Default: None.
        shape (tuple): Shape of the image to load. Default: (224, 224).
    """

    def __init__(self, path, streams=None, shape=(224, 224)):
        self._data = []  # List of dicts with paths to requested streams for each sample
        self._streams = streams if streams is not None else ['color', 'depth', 'xyz', 'mask']
        self._stream_channels = {
            "color": 3,
            "depth": 1,
            "xyz": 3,
            "mask": 1,
        }
        # Check that all requested streams are valid
        for s in self._streams:
            if s.lower() not in self._stream_channels:
                raise ValueError(f"Invalid stream '{s}'")

        self._transform = tf.Compose([
            tf.ToTensor(),
            tf.Resize(shape),
        ])

        # Stream Mappings
        # camera_intrinsics: CameraParameters.json
        # color -> VesselImg.png
        # depth -> Vessel_DepthMaps.exr
        # mask -> Vessel_ValidDepthMask.png
        # xyz -> Vessel_XYZMap.exr
        for sample in os.listdir(path):
            streams = {}
            for stream in self._streams:
                stream = stream.lower()
                if stream == 'xyz':
                    streams[stream] = os.path.join(path, sample, 'Vessel_XYZMap.exr')
                elif stream == 'color':
                    streams[stream] = os.path.join(path, sample, 'VesselImg.png')
                elif stream == 'depth':
                    streams[stream] = os.path.join(path, sample, 'Vessel_DepthMaps.exr')
                elif stream == 'mask':
                    streams[stream] = os.path.join(path, sample, 'Vessel_ValidDepthMask.png')

            # If all requested streams are available, add the sample to the dataset
            use_sample = True
            for stream in streams.keys():
                if not os.path.isfile(streams[stream]):
                    use_sample = False
                    print(f"WARNING: Stream '{stream}' not found for sample '{sample}'")
                    break

            if use_sample:
                self._data.append(streams)

    def __len__(self):
        """ Return the size of dataset. """
        return len(self._data)

    def __load_data(self, path, num_channels, dtype=np.float32):
        if ".exr" in path:
            i = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if np.ndim(i) >= 3 and num_channels == 1:
                i = i[:, :, 0]
        elif ".npy" in path:
            i = np.load(path)
        else:
            i = cv2.imread(path, 0) if num_channels == 1 else cv2.imread(path)

        # Make sure the image is in the correct format
        if i.dtype != dtype:
            i = i.astype(dtype)

        return i.astype(dtype)

    def __getitem__(self, index):
        """ Load a sample of data. """
        paths = self._data[index]

        # Load the requested streams
        data = {}
        for stream in self._streams:
            stream = stream.lower()
            path = paths[stream]
            num_channels = self._stream_channels[stream]

            if stream == 'xyz':
                if not os.path.isfile(path):
                    K = paths['K']
                    depth_path = paths['depth']
                    depth = self.__load_data(depth_path, num_channels=1)
                    xyz = depth2xyz(depth, K)
                    np.save(path, xyz)  # Save the XYZ map to disk
                    data[stream] = xyz
                else:
                    data[stream] = self.__load_data(path, num_channels, dtype=np.float32)

            data[stream] = self.__load_data(path, num_channels, dtype=np.float32)

        # Preprocess mask
        if 'mask' in data:
            mask = data["mask"]
            mask[mask > 0] = 1  # Convert masks to 0/1
            data["mask"] = mask

        # Preprocess depth
        if "depth" in data:
            depth = data["depth"]
            max_depth = 5000
            depth[depth > max_depth] = max_depth  # Remove depth values that are too far away

            # Apply mask (if available) and normalize depth
            if mask is not None:
                depth[mask == 0] = 1         # 1 for background
                depth_fg = depth[mask == 1]  # [0, 1) for foreground
                if depth_fg.size > 0:
                    depth[mask == 1] = (depth_fg - depth_fg.min()) / (depth_fg.max() - depth_fg.min())
            else:
                depth = depth / max_depth    # [0, 1] for all pixels

            depth[np.isnan(depth)] = 1       # Remove NaN values
            data["depth"] = depth
        
        # Preprocess normals
        if "normals" in data:
            norms = data['normals']
            norms = -norms[:, :, [2, 1, 0]]  # Flip normals axes to match our convention

            # Apply mask (if available)
            if mask is not None:
                norms[mask == 0, :] = 0      # Keep only the vessel ground-truth
                norms[mask == 0, 2] = 1      # (i.e., no ground plane and/or background)

            norms[np.isnan(norms)] = 0       # Remove NaN values
            data["normals"] = norms

        if "xyz" in data:
            xyz = data["xyz"]

            # Apply mask (if available)
            if xyz is not None:
                xyz[mask == 0, :] = 0      # Keep only the vessel ground-truth

            xyz[np.isnan(xyz)] = 0           # Remove NaN values
            data["xyz"] = xyz

        # Apply the transform, if specified
        if self._transform is not None:
            for stream in data:
                data[stream] = self._transform(data[stream])

        return data

    