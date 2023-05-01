# Data Efficient Contrastive Learing (ICML 2023)

Self-supervised learning (SSL) learns high-quality representations from large pools of unlabeled training data. As datasets grow larger, it becomes crucial to identify the examples that contribute the most to learning such representations. This enables efficient SSL by reducing the volume of data required for learning high-quality representations. Nevertheless, quantifying the value of examples for SSL has remained an open question. In this work, we address this for the first time, by proving that examples that contribute the most to contrastive SSL are those that have the most similar augmentations to other examples, in expectation. We provide rigorous guarantees for the generalization performance of SSL on such subsets. Empirically, we discover, perhaps surprisingly, the subsets that contribute the most to SSL are those that contribute the least to supervised learning. Through extensive experiments, we show that our subsets outperform random subsets by more than 3% on CIFAR100, CIFAR10, and STL10. Interestingly, we also find that we can safely exclude 20% of examples from CIFAR100 and 40% from STL10, without affecting downstream task performance.

Project Page: https://sjoshi804.github.io/data-efficient-contrastive-learning/

Paper: https://arxiv.org/abs/2302.09195

# Sample Usage

Samples shown for choosing subsets of CIFAR100 

TODO: Complete Sample Usage 
TODO: Add Contrastive Learning Code for SimCLR to this 

## CL-Core (default)

```python
# Approximate Latent Classes
import clcore 

trainset = co
partition = clcore.approximate_latent_classes.clip_approx()

# Get Subset
subset_dataset = 
```

## CL-Core (CLIP 0-shot Latent Classes)


## CL-Core (k-Means Latent Classes)


## Random Subset

## Custom Subset 

# Sample Implementation of Compatible Augmented Dataset (Required for Contrastive Learning)

```python 
import numpy as np
import torch
import torchvision
from PIL import Image

class CIFAR100Multiaugment(torchvision.datasets.CIFAR100):

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

        return imgs
```

# How to cite?

@misc{joshi2023dataefficient,
      title={Data-Efficient Contrastive Self-supervised Learning: Easy Examples Contribute the Most}, 
      author={Siddharth Joshi and Baharan Mirzasoleiman},
      year={2023},
      eprint={2302.09195},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}