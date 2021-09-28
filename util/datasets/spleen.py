from __future__ import print_function

import os

from PIL import Image

from util.augmentations import *
from .base import BaseDataset
import SimpleITK as sitk
import numpy as np
import os
import nibabel as nib
import imageio
from PIL import Image


class Spleen(BaseDataset):

    BASE_DIR, NUM_CLASS, CROP_SIZE = ('Task09_Spleen/', 2, 256)
    IN_CHANNELS = 1
    CLASS_WEIGHTS = None
    mean = [0.072520524]
    std = [0.18196131]

    def __init__(self, root, split='train', mode=None):
        super(Spleen, self).__init__(root, split, mode, norm={'mu': self.mean, 'std': self.std})
        self.root = os.path.expanduser(root)
        self.joint_transform = Compose([
            RandomTranslate(offset=(0.2, 0.1)),
            RandomVerticallyFlip(),
            RandomHorizontallyFlip(),
            RandomElasticTransform(alpha=1.5, sigma=0.07),
        ])
        base_path = os.path.join(self.root, self.BASE_DIR)
        image_path = os.path.join(base_path, 'imagesTr')
        mask_path = os.path.join(base_path, 'labelsTr')
        # self.go_through_gz(image_path, False)
        # self.go_through_gz(mask_path, True)

        self.data_info = []

        if mode in ['train', 'val']:
            for root, dirs, files in os.walk(image_path):
                path = root.split(os.sep)
                for file in files:
                    if '.nii.gz' in file:
                        continue
                    # image
                    # '../train_tiny_data/imgseg/Task09_Spleen/imagesTr/spleen_2/0.png'
                    image_dir = os.path.join(image_path, path[-1], file)
                    # mask
                    # '../train_tiny_data/imgseg/Task09_Spleen/labelsTr/spleen_2/0.png'
                    mask_dir = os.path.join(mask_path, path[-1], file)
                    self.data_info.append((image_dir, mask_dir))

            if len(self.data_info) == 0:
                raise (RuntimeError("Found 0 images in subfolders of: " + base_path + "\n"
                                                                                      "Supported image extensions are: " + ",".join(
                    'tif')))
        else:
            self.data_info = []

    def __getitem__(self, index):
        if len(self.data_info) == 0:
            return None, None

        img_path, target_path = self.data_info[index][0], self.data_info[index][1]

        # Read image
        img = Image.open(img_path)
        target = Image.open(target_path)

        # 1. do crop transform
        if self.mode == 'train':
            img, target = self.random_crop(img, target)
        elif self.mode == 'val':
            img, target = self.random_center_crop(img, target)

        # 2. do joint transform
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        ## 3.to tensor
        img, target = self.to_tensor(img, target)

        # 4. normalize for img
        img = self.img_normalize(img)

        # Convert label to 0, 1
        target[target == 255] = 1

        return img, target

    def __len__(self):
        return len(self.data_info)

    def calculate_mean_std(self):
        mean = torch.zeros(1)
        std = torch.zeros(1)
        for data in self.data_info:
            img_path, target_path = data[0], data[1]
            img = Image.open(img_path)
            target = Image.open(target_path)

            if self.joint_transform is not None:
                img, target = self.joint_transform(img, target)

            img, target = self.to_tensor(img, target)
            mean += img[:, :, :].mean()
            std += img[:, :, :].std()
        mean.div_(len(self.data_info))
        std.div_(len(self.data_info))
        return list(mean.numpy()), list(std.numpy())

    def go_through_gz(self, gz_folder, is_label):
        for root, dirs, files in os.walk(gz_folder):
            path = root.split(os.sep)
            for file in files:
                if '.nii.gz' not in file:
                    continue
                new_folder = os.path.join(gz_folder, file.split('.')[0])
                if not os.path.exists(new_folder):
                    os.mkdir(new_folder)
                gz = os.path.join(gz_folder, file)
                self.nii_to_image(gz, new_folder, is_label)

    def nii_to_image(self, img_path: str, write_path, is_label):
        img = nib.load(img_path)
        for i in range(img.shape[-1]):
            img_arr = img.dataobj[:, :, i]
            if is_label:
                img_arr = 255 * img_arr.astype('int')
            im = Image.fromarray(img_arr).convert("L")
            im.save(os.path.join(write_path, str(i)+'.png'), format="png")


