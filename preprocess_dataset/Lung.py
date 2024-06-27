import os
from pathlib import Path

import cv2
import torch

from torch.utils.data import DataLoader

from preprocess_dataset.transforms import \
    Compose, ToPILImage, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine, Normalize


def cv2_loader_lung(path, is_mask):
    if is_mask:
        img = cv2.imread(path)
        img[img > 0] = 1
    else:
        img = cv2.imread(path)

    return img

# similar Config used int the other baselines
def get_lung_transform(image_resize):
    transform_train = Compose([
        ToPILImage(),
        Resize((image_resize, image_resize)),
        # RandomHorizontalFlip(),
        # RandomVerticalFlip(),
        RandomAffine(int(22), scale=(float(0.75), float(1.25))),
        ToTensor(),
        Normalize(mean=[0.0, 0.0, 0.0], std=[1., 1., 1.]),
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((image_resize, image_resize)),
        ToTensor(),
        Normalize(mean=[0.0, 0.0, 0.0], std=[1., 1., 1.]),
    ])
    return transform_train, transform_test


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=cv2_loader_lung,
                 image_size: int = 0, fold: int = 0):
        self.root = root
        self.imgs_root = os.path.join(f"{self.root}/CXR_png/")
        self.masks_root = os.path.join(f"{self.root}/masks/")
        self.fold = fold
        # we have 704 masks but 800 images. Hence we are going to
        # make a 1-1 correspondance from mask to images, not the usual other way.
        mask = os.listdir(self.masks_root)
        mask = [fName.split(".png")[0] for fName in mask]

        check = [i for i in mask if "mask" in i]
        print("Total mask that has modified name:", len(check))

        self.testing_files = sorted(set(os.listdir(self.imgs_root)) & set(os.listdir(self.masks_root)))
        self.training_files = sorted(check)

        n = len(self.testing_files)
        if self.fold == 1:
            self.testing_files, self.training_files[:n] = self.training_files[:n], self.testing_files
        elif self.fold == 2:
            self.testing_files, self.training_files[-n:] = self.training_files[-n:], self.testing_files
        elif self.fold != 0:
            raise ValueError("Invalid fold value fo Lung Dataset. It should be 0, 1, or 2.")

        self.paths = sorted(os.listdir(self.imgs_root))
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.train = train
        self.target_transform = target_transform

        # print('num of data:{}'.format(len(self.paths)))
        print('num of train data:{}'.format(len(self.training_files)))
        print('num of test data:{}'.format(len(self.testing_files)))

    def __getitem__(self, index):

        if self.train:
            if self.fold == 0:
                img = self.loader(os.path.join(self.imgs_root, self.training_files[index].split("_mask")[0] + ".png"),
                                  is_mask=False)
                mask = self.loader(os.path.join(self.masks_root, self.training_files[index] + ".png"), is_mask=True)
            elif self.fold == 1:
                if index < len(self.testing_files):
                    img = self.loader(os.path.join(self.imgs_root, list(self.training_files)[index]), is_mask=False)
                    mask = self.loader(os.path.join(self.masks_root, list(self.training_files)[index]), is_mask=True)

                else:
                    img = self.loader(
                        os.path.join(self.imgs_root, self.training_files[index].split("_mask")[0] + ".png"),
                        is_mask=False)
                    mask = self.loader(os.path.join(self.masks_root, self.training_files[index] + ".png"), is_mask=True)
            elif self.fold == 2:
                if index > len(self.training_files) - len(self.testing_files) - 1:
                    img = self.loader(os.path.join(self.imgs_root, list(self.training_files)[index]), is_mask=False)
                    mask = self.loader(os.path.join(self.masks_root, list(self.training_files)[index]), is_mask=True)
                else:
                    img = self.loader(
                        os.path.join(self.imgs_root, self.training_files[index].split("_mask")[0] + ".png"),
                        is_mask=False)
                    mask = self.loader(os.path.join(self.masks_root, self.training_files[index] + ".png"), is_mask=True)
        else:
            if self.fold == 0:
                img = self.loader(os.path.join(self.imgs_root, list(self.testing_files)[index]), is_mask=False)
                mask = self.loader(os.path.join(self.masks_root, list(self.testing_files)[index]), is_mask=True)
            else:
                img = self.loader(
                    os.path.join(self.imgs_root, self.testing_files[index].split("_mask")[0] + ".png"),
                    is_mask=False)
                mask = self.loader(os.path.join(self.masks_root, self.testing_files[index] + ".png"), is_mask=True)

        img, mask = self.transform(img, mask)
        img = (img.mean(dim=0).unsqueeze(0) / 255.0) * 2 - 1
        out_dict = {"conditioned_image": img}
        mask = mask.mean(2)
        mask = 2 * -mask + 1.0

        return mask.unsqueeze(0), out_dict, f"{Path(self.paths[index]).stem}_{index}"

    def __len__(self):
        if self.train:
            return len(self.training_files)

        else:
            return len(self.testing_files)
