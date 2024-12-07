import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
from torchvision import transforms as TF
from torchvision.transforms.functional import to_tensor, resize

import os
import random
from skimage import transform
join = os.path.join

IMG_SIZE = 1024


class AudioDenoisingDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.images_dir = join(data_root, "Images")
        self.masks_dir = join(data_root, "Masks")
        self.images = [img for img in os.listdir(self.images_dir) if img.endswith('.png')]
        self.masks = [mask.replace('.png', '.png') for mask in self.images]

        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.images)}")
        print(f"number of masks: {len(self.masks)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        img_ori = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask_ori = Image.open(os.path.join(self.masks_dir, img_name)).convert('L')
        img_256 = resize(img_ori, size=[256, 256], interpolation=Image.Resampling.NEAREST)
        mask_256 = resize(mask_ori, size=[256, 256], interpolation=Image.Resampling.NEAREST)
        img_1024 = np.array(img_256)
        mask = np.array(mask_256)
        resize_img_skimg = transform.resize(
                img_1024,
                (IMG_SIZE, IMG_SIZE),
                order=3,
                preserve_range=True,
                mode="constant",
                anti_aliasing=True,
            )
        resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
        resize_mask_skimg = transform.resize(
                mask,
                (IMG_SIZE, IMG_SIZE),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        resize_mask_skimg = np.uint8(resize_mask_skimg)
        resize_mask_skimg_01 = (resize_mask_skimg - resize_mask_skimg.min()) / np.clip(
                resize_mask_skimg.max() - resize_mask_skimg.min(), a_min=1e-8, a_max=None
            )

        assert resize_img_skimg_01.shape[:2] == resize_mask_skimg.shape

        img_1024 = resize_img_skimg_01
        mask = resize_mask_skimg_01

        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        
        mask = (mask > 0).astype(np.uint8)
        if not (np.max(mask) == 1 and np.min(mask) == 0.0):
            print("img_name: {}".format(img_name))
            
        default_xy = np.array([100, 900])
        y_indices, x_indices = np.where(mask > 0)
        if y_indices.size == 0:
            y_indices = default_xy
        if x_indices.size == 0:
            x_indices = default_xy
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )