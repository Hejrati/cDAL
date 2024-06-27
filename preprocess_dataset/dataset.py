import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess_dataset.Lung import LungDataset, get_lung_transform
from preprocess_dataset.MoNu import MonuDataset, get_monu_transform


def create_dataset(data_dir: str, mode: str = "train", image_size: int = 256, dataset_name: str = "monu", fold: int = 0):
    if dataset_name == "monu":
        dataset_class = MonuDataset
        transform_train, transform_test = get_monu_transform(image_size=image_size)
    elif dataset_name == "lung":
        dataset_class = LungDataset
        image_size = image_size
        transform_train, transform_test = get_lung_transform(image_resize=image_size)
    else:
        raise ValueError(
            "Dataset name should be either \"monu\" , \"lung\", "
            "Unknown dataset: {}".format(dataset_name))

    if mode == "train":
        return dataset_class(data_dir, train=True, transform=transform_train, image_size=image_size, fold=fold)
    else:
        return dataset_class(data_dir, train=False, transform=transform_test, image_size=image_size, fold=fold)


def load_data(*, data_dir: str, batch_size: int, image_size: int, deterministic=True, dataset_name: str = 'monu'):
    """
    For a dataset, create a generator over (images, labels) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    :param dataset_name: dataset name should be either "monu" or "davis"
    """

    dataset_date = create_dataset(data_dir=data_dir, image_size=image_size, mode="train", dataset_name=dataset_name, fold=0)

    if deterministic:
        loader = DataLoader(dataset_date, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    else:
        loader = DataLoader(dataset_date, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    while True:
        yield from loader


if __name__ == "__main__":
    # ['lung', 'monu']
    for dataset in ['lung', 'monu']:
        if dataset == 'monu':
            # data_dir = "C:\\Users\\behzad\\Desktop\\Data\\Medical"
            data_dir = "/home/share/Data/Medical"
        elif dataset == 'lung':
            # data_dir = "C:\\Users\\behzad\\Desktop\\Data\\Lung"
            data_dir = "/home/share/Data/Lung"
        else:
            raise ValueError(
                "Dataset name should be either \"monu\" or \"davis\", Unknown dataset: {}".format(dataset))

        val_dataset = create_dataset(mode='test', image_size=256, data_dir=data_dir, dataset_name=dataset, fold=0)
        ds = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False,
                                         drop_last=True)  # type:ignore
        pbar = tqdm(ds)

        for i, (query, out_dict, _) in enumerate(pbar):

            # here print conditioned image
            if dataset == 'monu':
                plt.imshow(out_dict['conditioned_image'].squeeze().permute(1, 2, 0).numpy().astype(np.uint8))
                plt.show()
            elif dataset == 'lung':
                plt.imshow(out_dict['conditioned_image'].squeeze().numpy())
                plt.show()

            # here print query image
            plt.imshow(query.squeeze().numpy().astype(np.uint8), cmap='gray')
            plt.show()
            if i == 0:
                break
