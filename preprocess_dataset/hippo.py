import os

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset, load_decathlon_datalist, partition_dataset, \
    decollate_batch
from monai.transforms import (
    AddChanneld,
    Compose,
    DeleteItemsd,
    FgBgToIndicesd,
    LoadImage,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    SpatialPadd,
    ToTensord, EnsureType, AsDiscrete,
)


class HippocampusDecathlonDataModule(pl.LightningDataModule):
    class_weight = np.asarray([0.01361341, 0.47459406, 0.51179253])

    def __init__(
            self,
            root_dir=".",
            fold=0,
            train_patch_size=(32, 32, 32),
            num_samples=16,
            batch_size=1,
            cache_rate=0.,
            cache_dir=None,
            num_workers=4,
            balance_sampling=True,
            train_transforms=None,
            val_transforms=None,
            **kwargs
    ):
        super().__init__()
        self.base_dir = root_dir + "/Task04_Hippocampus/"
        self.fold = fold
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        print('================')

        if balance_sampling:
            pos = neg = 0.5
        else:
            pos = np.sum(self.class_weight[1:])
            neg = self.class_weight[0]

        if train_transforms is None:
            self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="NibabelReader"),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="LPI"),
                    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                    SpatialPadd(keys=["image", "label"], spatial_size=train_patch_size, mode="edge"),
                    FgBgToIndicesd(keys=["label"], image_key="image"),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=train_patch_size,
                        pos=pos,
                        neg=neg,
                        num_samples=num_samples,
                        fg_indices_key="label_fg_indices",
                        bg_indices_key="label_bg_indices",

                    ),
                    DeleteItemsd(keys=["label_fg_indices", "label_bg_indices"]),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            self.train_transforms = train_transforms

        if val_transforms is None:
            self.val_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="NibabelReader"),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="LPI"),
                    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            self.val_transforms = val_transforms

    def _load_data_dicts(self, train=True):
        if train:
            data_dicts = load_decathlon_datalist(
                os.path.join(self.base_dir, "dataset.json"), data_list_key="training", base_dir=self.base_dir
            )
            data_dicts_list = partition_dataset(data_dicts, num_partitions=4, shuffle=True, seed=0)
            train_dicts, val_dicts = [], []
            for i, data_dict in enumerate(data_dicts_list):
                if i == self.fold:
                    val_dicts.extend(data_dict)
                else:
                    train_dicts.extend(data_dict)
            return train_dicts, val_dicts
        else:
            pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_data_dicts, val_data_dicts = self._load_data_dicts()

            if self.cache_rate is not None:
                self.trainset = CacheDataset(
                    data=train_data_dicts,
                    transform=self.train_transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                )
                self.valset = CacheDataset(
                    data=val_data_dicts, transform=self.val_transforms, cache_rate=self.cache_rate, num_workers=4,
                )
            elif self.cache_dir is not None:
                self.trainset = PersistentDataset(
                    data=train_data_dicts, transform=self.train_transforms, cache_dir=self.cache_dir
                )
                self.valset = PersistentDataset(
                    data=val_data_dicts, transform=self.val_transforms, cache_dir=self.cache_dir
                )
            else:
                self.trainset = Dataset(data=train_data_dicts, transform=self.train_transforms)
                self.valset = Dataset(data=val_data_dicts, transform=self.val_transforms)
        elif stage == "validate":
            _, val_data_dicts = self._load_data_dicts()
            self.valset = CacheDataset(
                data=val_data_dicts, transform=self.val_transforms, cache_rate=1.0, num_workers=4
            )

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=1, num_workers=4, shuffle=False)

    def test_dataloader(self):
        pass

    def calculate_class_weight(self):
        data_dicts = load_decathlon_datalist(
            os.path.join(self.base_dir, "dataset.json"), data_list_key="training", base_dir=self.base_dir
        )

        class_weight = []
        for data_dict in data_dicts:
            label = LoadImage(reader="NibabelReader", image_only=True)(data_dict["label"])

            _, counts = np.unique(label, return_counts=True)
            counts = np.sum(counts) / counts
            # Normalize
            counts = counts / np.sum(counts)
            class_weight.append(counts)

        class_weight = np.asarray(class_weight)
        class_weight = np.mean(class_weight, axis=0)
        print("Class weight: ", class_weight)

    def calculate_class_percentage(self):
        data_dicts = load_decathlon_datalist(
            os.path.join(self.base_dir, "dataset.json"), data_list_key="training", base_dir=self.base_dir
        )

        class_percentage = []
        for data_dict in data_dicts:
            label = LoadImage(reader="NibabelReader", image_only=True)(data_dict["label"])

            _, counts = np.unique(label, return_counts=True)
            # Normalize
            counts = counts / np.sum(counts)
            class_percentage.append(counts)

        class_percentage = np.asarray(class_percentage)
        class_percentage = np.mean(class_percentage, axis=0)
        print("Class Percentage: ", class_percentage)


if __name__ == "__main__":
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=3)])
    num_classes = 3
    data_module = HippocampusDecathlonDataModule(root_dir="/home/share/Data/")
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()

    for i, query in enumerate(train_loader):
        onehot_labels = [post_label(label) for label in decollate_batch(query['label'])]
        labels = torch.stack(onehot_labels, dim=0).permute(0, 4, 1, 2, 3).reshape(-1, num_classes, 32, 32)
        img = query['image'].permute(0, 4, 1, 2, 3).reshape(-1, 1, 32, 32)

        img = 2 * img - 1
        # here print conditioned image and label
        print(f"label shape: {labels.shape}")
        print(f"image shape: {img.shape}")
        plt.imshow(img[20, 0], cmap='gray')
        plt.title('Image')
        plt.show()
        print(labels[0, 1])
        plt.imshow(labels[0, 0], cmap='gray')
        plt.title('Background')
        plt.show()
        plt.imshow(labels[20, 1], cmap='gray')
        plt.title('Antorior')
        plt.show()
        plt.imshow(labels[20, 2], cmap='gray')
        plt.title('Posterior')
        plt.show()
        break
