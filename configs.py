from enum import Enum
import json

import torchvision
import torchvision.transforms as transforms

from collections import namedtuple
from data_proc.augmentation import ColourDistortion
from data_proc.dataset import *
from resnet import *

class SupportedDatasets(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    TINY_IMAGENET = "tiny_imagenet"
    IMAGENET = "imagenet"
    STL10 = "stl10"

Datasets = namedtuple('Datasets', 'trainset testset clftrainset num_classes stem')

def get_datasets(dataset: str, augment_clf_train=False, add_indices_to_data=False, num_positive=2):

    CACHED_MEAN_STD = {
        SupportedDatasets.CIFAR10.value: ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        SupportedDatasets.CIFAR100.value: ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        SupportedDatasets.STL10.value: ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
        SupportedDatasets.TINY_IMAGENET.value: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        SupportedDatasets.IMAGENET.value: ((0.485, 0.456, 0.3868), (0.2309, 0.2262, 0.2237))
    }

    PATHS = {
        SupportedDatasets.CIFAR10.value: '/data/cifar10/',
        SupportedDatasets.CIFAR100.value: '/data/cifar100/',
        SupportedDatasets.STL10.value: '/data/stl10/',
        SupportedDatasets.TINY_IMAGENET.value: '/data/tiny_imagenet/',
        SupportedDatasets.IMAGENET.value: '/data/ILSVRC/'
    }

    try:
        with open('dataset-paths.json', 'r') as f:
            local_paths = json.load(f)
            PATHS.update(local_paths)
    except FileNotFoundError:
        pass
    root = PATHS[dataset]

    # Data
    if dataset == SupportedDatasets.STL10.value:
        img_size = 96
    elif dataset == SupportedDatasets.IMAGENET.value:
        img_size = 224
    elif dataset == SupportedDatasets.TINY_IMAGENET.value:
       img_size = 64
    else:
        img_size = 32

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        ColourDistortion(s=0.5),
        transforms.ToTensor(),
        transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    ])

    if augment_clf_train:
        transform_clftrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    else:
        transform_clftrain = transform_test

    trainset = testset = clftrainset = num_classes = stem = None
    
    if dataset == SupportedDatasets.CIFAR100.value:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR100)
        else:
            dset = torchvision.datasets.CIFAR100
            trainset = CIFAR100Augment(root=root, train=True, download=True, transform=transform_train, n_augmentations=num_positive)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        num_classes = 100
        stem = StemCIFAR

    elif dataset == SupportedDatasets.CIFAR10.value:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR10)
        else:
            dset = torchvision.datasets.CIFAR10 
            trainset = CIFAR10Augment(root=root, train=True, download=True, transform=transform_train, n_augmentations=num_positive)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        num_classes = 10
        stem = StemCIFAR
    elif dataset == SupportedDatasets.STL10.value:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.STL10)
        else:
            dset = torchvision.datasets.STL10
            trainset = STL10Augment(root=root, split='train+unlabeled', download=True, transform=transform_train)
        clftrainset = dset(root=root, split='train', download=True, transform=transform_clftrain)
        testset = dset(root=root, split='test', download=True, transform=transform_test)
        num_classes = 10
        stem = StemSTL

    elif dataset == SupportedDatasets.TINY_IMAGENET.value:
        if add_indices_to_data:
            raise NotImplementedError("Not implemented for TinyImageNet")
        trainset = ImageFolderAugment(root=f"{root}train/", transform=transform_train, num_positive=num_positive)  
        clftrainset = ImageFolder(root=f"{root}train/", transform=transform_clftrain)      
        testset = ImageFolder(root=f"{root}test/", transform=transform_train)    
        num_classes = 200
        stem = StemCIFAR
    
    elif dataset == SupportedDatasets.IMAGENET.value:
        if add_indices_to_data:
            raise NotImplementedError("Not implemented for ImageNet")
        trainset = ImageNetAugment(root=f"{root}train_full/", transform=transform_train, n_augmentations=num_positive)
        clftrainset = ImageNet(root=f"{root}train_full/", transform=transform_clftrain)      
        testset = ImageNet(root=f"{root}test/", transform=transform_clftrain)     
        num_classes = 1000
        stem = StemImageNet

    return Datasets(trainset=trainset, testset=testset, clftrainset=clftrainset, num_classes=num_classes, stem=stem)
