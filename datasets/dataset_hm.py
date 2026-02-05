import os
import random
import rasterio
import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k, axes=(0, 1))

    axis = np.random.randint(1, 3)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis-1).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        c, x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


class hm_dataset(Dataset):
    def __init__(self, base_dir, split, has_annotations=True, transform=None):
        self.transform = transform
        self.split = split
        self.has_annotations = has_annotations
        self.img_dir = os.path.join(base_dir, split)
        self.label_dir = os.path.join(base_dir, 'annotation', split)
        self.file_names = os.listdir(os.path.join(base_dir, split))

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        
        with rasterio.open(img_path) as src:
            image = src.read()  # (C, H, W)
            
            if image.shape[0] > 3:
                image = image[:3, :, :]
        
        image = np.transpose(image, (1, 2, 0))
        
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        image = image.astype(np.float32) / 255.0

        image = np.transpose(image, (2, 0, 1))

        if self.has_annotations:
            label_path = os.path.join(self.label_dir, self.file_names[idx])

            
            with rasterio.open(label_path) as src:
                label = src.read(1)  # (H, W)

            label = (label > 0).astype(np.uint8)

            label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)

            sample = {'image': image, 'label': label}
        else:
            sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.file_names[idx][:-4]
        return sample
