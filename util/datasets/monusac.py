from __future__ import print_function
import os
from .base import BaseDataset
import pydicom
from util.augmentations import *
from util.utils import create_class_weight
from PIL import Image


class MoNuSAC(BaseDataset):
    BASE_DIR, NUM_CLASS, CROP_SIZE = ('MoNuSAC/', 2, 256)
    IN_CHANNELS = 1
    CLASS_WEIGHTS = None
    mean = [0.5318197]
    std = [0.20710655]

    def __init__(self, root, split='train', mode=None):
        super(MoNuSAC, self).__init__(root, split, mode, norm={'mu': self.mean, 'std': self.std})
        self.root = os.path.expanduser(root)
        self.joint_transform = Compose([
            RandomTranslate(offset=(0.3, 0.3)),
            RandomHorizontallyFlip(),
            RandomElasticTransform(alpha=1.5, sigma=0.07),
        ])
        base_path = os.path.join(self.root, self.BASE_DIR)
        cleaned_image_path = os.path.join(base_path, 'MoNuSAC_cleaned', 'images')
        cleaned_mask_path = os.path.join(base_path, 'MoNuSAC_cleaned', 'masks')
        self.data_info = []

        if mode in ['train', 'val']:
            for root, dirs, files in os.walk(cleaned_mask_path):
                path = root.split(os.sep)
                for file in files:
                    # mask
                    # ../train_tiny_data/imgseg/MoNuSAC/MoNuSAC_cleaned/masks/TCGA-5P-A9K0-01Z-00-DX1_1_1.png
                    mask_dir = os.path.join(cleaned_mask_path, file)
                    # image
                    # ../train_tiny_data/imgseg/MoNuSAC/MoNuSAC_cleaned/images/TCGA-5P-A9K0-01Z-00-DX1_1_1.png
                    image_dir = os.path.join(cleaned_image_path, file)
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


