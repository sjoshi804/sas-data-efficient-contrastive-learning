from enum import Enum
import json

import torchvision
import torchvision.transforms as transforms

from collections import namedtuple
from data_proc.augmentation import ColourDistortion, GaussianBlur
from data_proc.dataset import *
from models import *

class SupportedDatasets(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    TINY_IMAGENET = "tiny_imagenet"
    STL10 = "stl10"

Datasets = namedtuple('Datasets', 'trainset testset clftrainset num_classes stem')

def get_datasets(dataset, augment_clf_train=False, add_indices_to_data=False, num_positive=2, imbalance=False, format_img=False):

    if imbalance and dataset != "cifar10":
        raise NotImplementedError("Imbalance supported only on CIFAR10 currently.")

    CACHED_MEAN_STD = {
        SupportedDatasets.CIFAR10.value: ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        SupportedDatasets.CIFAR100.value: ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        SupportedDatasets.STL10.value: ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
        SupportedDatasets.TINY_IMAGENET.value: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    }

    PATHS = {
        SupportedDatasets.CIFAR10.value: '/data/cifar10/',
        SupportedDatasets.CIFAR100.value: '/data/cifar100/',
        SupportedDatasets.STL10.value: '/data/stl10/',
        SupportedDatasets.TINY_IMAGENET.value: '/data/tiny_imagenet/'
    }
    try:
        with open('dataset-paths.json', 'r') as f:
            local_paths = json.load(f)
            PATHS.update(local_paths)
    except FileNotFoundError:
        pass
    root = PATHS[dataset]

    # Data
    if dataset == SupportedDatasets.STL10:
        img_size = 96
    #elif dataset == SupportedDatasets.TINY_IMAGENET:
    #    img_size = 64
    else:
        img_size = 32

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        ColourDistortion(s=0.5),
        transforms.ToTensor(),
        transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    ])

    if format_img:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            ColourDistortion(s=0.5)
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
        if num_positive is None:
            trainset = CIFAR100Biaugment(root=root, train=True, download=True, transform=transform_train, format_img=format_img)
        else:
            trainset = CIFAR100Multiaugment(root=root, train=True, download=True, transform=transform_train,
                                            n_augmentations=num_positive, format_img=format_img)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        num_classes = 100
        stem = StemCIFAR

    elif dataset == SupportedDatasets.CIFAR10.value:
        if add_indices_to_data:
            dset = add_indices(CIFAR10Imbalance)
        else:
            dset = CIFAR10Imbalance
        if num_positive is None:
            trainset = CIFAR10Biaugment(root=root, train=True, download=True, transform=transform_train, imbalance=imbalance, format_img=format_img)
        else:
            trainset = CIFAR10Multiaugment(root=root, train=True, download=True, transform=transform_train,
                                           n_augmentations=num_positive, imbalance=imbalance, format_img=format_img)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain, imbalance=imbalance)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        if imbalance:
            rare_testset = dset(root=root, train=False, download=True, transform=transform_test, imbalance=imbalance)
            testset =  (testset, rare_testset)
        num_classes = 10
        stem = StemCIFAR

    elif dataset == SupportedDatasets.STL10.value:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.STL10)
        else:
            dset = torchvision.datasets.STL10
        if num_positive is None:
            trainset = STL10Biaugment(root=root, split='train+unlabeled', download=True, transform=transform_train, format_img=format_img)
        else:
            trainset = STL10Multiaugment(root=root, split='train+unlabeled', download=True, transform=transform_train, format_img=format_img)
        clftrainset = dset(root=root, split='train', download=True, transform=transform_clftrain)
        testset = dset(root=root, split='test', download=True, transform=transform_test)
        num_classes = 10
        stem = StemSTL

    elif dataset == SupportedDatasets.TINY_IMAGENET.value:
        if add_indices_to_data or num_positive is not None:
            raise NotImplementedError("Not implemented for Tiny")
        trainset = ImageFolderBiaugment(root=f"{root}train/", transform=transform_train)  
        clftrainset = ImageFolder(root=f"{root}train/", transform=transform_clftrain)      
        testset = ImageFolder(root=f"{root}test/", transform=transform_train)    
        num_classes = 200
        stem = StemCIFAR
    
    if format_img:
        return trainset 
    
    return Datasets(trainset=trainset, testset=testset, clftrainset=clftrainset, num_classes=num_classes, stem=stem)
