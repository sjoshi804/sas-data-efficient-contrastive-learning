from typing import Callable, Optional

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder


class CIFAR10Imbalance(torchvision.datasets.CIFAR10):
        def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            imbalance: bool = False, 
            format_img: bool = False
        ) -> None:
            super().__init__(
                root=root,
                train=train,
                transform=transform,
                target_transform=target_transform,
                download=download)
                
            self.format_img = format_img

            if imbalance and self.train:
                rare_class_counter = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
                max_rare_class_count = 50
                indices_to_keep = []
                for i in range(len(self.data)):
                    if self.targets[i] in rare_class_counter:
                        if rare_class_counter[self.targets[i]] < max_rare_class_count:
                            rare_class_counter[self.targets[i]] += 1
                            indices_to_keep.append(i)
                    else:
                        indices_to_keep.append(i)
                self.data = [self.data[i] for i in indices_to_keep]
                self.targets = [self.targets[i] for i in indices_to_keep]
            if imbalance and not self.train:
                rare_classes = [3, 4, 5, 6, 7]
                indices_to_keep = []
                for i in range(len(self.data)):
                    if self.targets[i] in rare_classes:
                        indices_to_keep.append(i)
                self.data = [self.data[i] for i in indices_to_keep]
                self.targets = [self.targets[i] for i in indices_to_keep]

class CIFAR10Biaugment(CIFAR10Imbalance):


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(pil_img)
            img2 = self.transform(pil_img)
        else:
            img2 = img = pil_img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img, img2]

class CIFAR100Biaugment(CIFAR10Biaugment):
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


class STL10Biaugment(torchvision.datasets.STL10):
    def __init__(
            self,
            root: str,
            split: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            format_img: bool = False
        ) -> None:
            super().__init__(
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download)
            
            if format_img:
                raise NotImplementedError("format_img=True not supported for STL10 currently.")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(pil_img)
            img2 = self.transform(pil_img)
        else:
            img2 = img = pil_img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img, img2]

class STL10Multiaugment(torchvision.datasets.STL10):
    def __init__(
            self,
            root: str,
            split: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            format_img: bool = False,
            num_positive: int = 2
        ) -> None:
            super().__init__(
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download)
            
            self.num_positive = num_positive

            if format_img:
                raise NotImplementedError("format_img=True not supported for STL10 currently.")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        imgs = [self.transform(pil_img) for _ in range(self.num_positive)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs
    
class CIFAR10Multiaugment(CIFAR10Imbalance):

    def __init__(self, *args, n_augmentations=8, **kwargs):
        super(CIFAR10Multiaugment, self).__init__(*args, **kwargs)
        self.n_augmentations = n_augmentations
        assert self.transforms is not None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(img)

        imgs = [self.transform(pil_img) for _ in range(self.n_augmentations)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.format_img:
            return imgs, target, index

        return imgs


class CIFAR100Multiaugment(CIFAR10Multiaugment):
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


class ImageFolderBiaugment(ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            img = self.transform(sample)
            img2 = self.transform(sample)
        else:
            img2 = img = sample
        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img, img2]


def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass
