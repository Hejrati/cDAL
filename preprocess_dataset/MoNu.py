import os
from pathlib import Path

import imageio
import tifffile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess_dataset.transforms import \
    Compose, ToPILImage, ColorJitter, RandomHorizontalFlip, ToTensor, Normalize, RandomVerticalFlip, RandomAffine, \
    Resize, RandomCrop


def cv2_loader(path, is_mask):
    if is_mask:
        img = imageio.imread(path)
        img[img > 0] = 1
    else:
        img = tifffile.imread(path)
    return img

# this code for image transformation of MoNuSeg dataset
# these parameters are set based on MoNuSeg requirements
def get_monu_transform(image_size: int = 256, image_resize: int = 512):
    transform_train = Compose([
        ToPILImage(),
        Resize((image_resize, image_resize)),
        RandomCrop((image_size, image_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAffine(int(22), scale=(float(0.75), float(1.25))),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # type:ignore
        ToTensor(),
        Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])

    ])
    return transform_train, transform_test


class MonuDataset(torch.utils.data.Dataset):  # type:ignore
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=cv2_loader,
                 image_size: int = 0, fold: int = 0):
        self.root = root
        if train:
            self.imgs_root = os.path.join(self.root, 'Training', 'img')
            self.masks_root = os.path.join(self.root, 'Training', 'mask')
        else:
            self.imgs_root = os.path.join(self.root, 'Test', 'img')
            self.masks_root = os.path.join(self.root, 'Test', 'mask')

        self.paths = sorted(os.listdir(self.imgs_root))
        self.image_size = image_size
        self.transform = transform
        self.loader = loader

        self.target_transform = target_transform

        print('num of data:{}'.format(len(self.paths)))

    def __getitem__(self, index):

        mask_path = self.paths[index].split('.')[0] + '.png'
        img = self.loader(os.path.join(self.imgs_root, self.paths[index]), is_mask=False)
        mask = self.loader(os.path.join(self.masks_root, mask_path), is_mask=True)

        img, mask = self.transform(img, mask)  # type:ignore
        out_dict = {"conditioned_image": img}
        mask = 2 * mask - 1.0  # range is between [-1, 1] , -1 is background and 1 is objects
        return mask.unsqueeze(0), out_dict, f"{Path(self.paths[index]).stem}_{index}"

    def __len__(self):
        return len(self.paths)
