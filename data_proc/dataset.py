import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder

class CIFAR10Augment(torchvision.datasets.CIFAR10):
    def __init__(self, root: str, transform=Callable, n_augmentations: int = 2, train: bool = True, download: bool = False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            download=download
        )
        self.n_augmentations = n_augmentations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, _ = self.data[index], self.targets[index]
        pil_img = Image.fromarray(img)
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs

class STL10Augment(torchvision.datasets.STL10):
    def __init__(
            self,
            root: str,
            split: str,
            transform: Callable,
            n_augmentations: int = 2,
            download: bool = False,
        ) -> None:
            super().__init__(
                root=root,
                split=split,
                transform=transform,
                download=download)
            self.n_augmentations = n_augmentations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, _ = self.data[index], int(self.labels[index])
        else:
            img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs
    

class CIFAR100Augment(CIFAR10Augment):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10Biaugment` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class ImageFolderAugment(ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        pil_img = self.loader(path)
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs

def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"), on_bad_lines='skip')
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])).convert('RGB'))
        label = self.labels[idx]
        return image, label
        

class ImageNetAugment(torch.utils.data.Dataset):
    def __init__(self, root, transform, n_augmentations=2):
        self.root = root
        self.transform = transform 
        self.n_augmentations = n_augmentations
        df = pd.read_csv(os.path.join(root, "labels.csv"), on_bad_lines='skip')
        self.images = df["image"]
        self.labels = df["label"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pil_img = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs