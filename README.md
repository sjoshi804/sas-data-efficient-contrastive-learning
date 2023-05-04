# Data Efficient Contrastive Learing (ICML 2023)

## Abstract

Self-supervised learning (SSL) learns high-quality representations from large pools of unlabeled training data. As datasets grow larger, it becomes crucial to identify the examples that contribute the most to learning such representations. This enables efficient SSL by reducing the volume of data required for learning high-quality representations. Nevertheless, quantifying the value of examples for SSL has remained an open question. In this work, we address this for the first time, by proving that examples that contribute the most to contrastive SSL are those that have the most similar augmentations to other examples, in expectation. We provide rigorous guarantees for the generalization performance of SSL on such subsets. Empirically, we discover, perhaps surprisingly, the subsets that contribute the most to SSL are those that contribute the least to supervised learning. Through extensive experiments, we show that our subsets outperform random subsets by more than 3% on CIFAR100, CIFAR10, and STL10. Interestingly, we also find that we can safely exclude 20% of examples from CIFAR100 and 40% from STL10, without affecting downstream task performance.

Project Page: https://sjoshi804.github.io/data-efficient-contrastive-learning/

Paper: https://arxiv.org/abs/2302.09195

## BibTex Citation

```bibtex
@misc{joshi2023dataefficient,
      title={Data-Efficient Contrastive Self-supervised Learning: Easy Examples Contribute the Most}, 
      author={Siddharth Joshi and Baharan Mirzasoleiman},
      year={2023},
      eprint={2302.09195},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Sample Usage

Samples shown for choosing subsets of CIFAR100

TODO: Complete Sample Usage
TODO: Clarify what format of dataset is expected for each function, see if you can make that an interface and ask all to implement?

### SAS (default)

```python
# Approximate Latent Classes
from sas.approx_latent_classes import clip_approx
from sas.subset_dataset import SASSubsetDataset
import random 

cifar100 = torchvision.datasets.CIFAR100("/data/cifar100/", transform=transforms.ToTensor())
device = "cuda:0"

rand_labeled_examples_indices = random.sample(len(cifar100), 500)
rand_labeled_examples_labels = [cifar100[i][1] for i in rand_labeled_examples_indices]


partition = partition = clip_approx(
    img_trainset=cifar100,
    labeled_example_indices=rand_labeled_examples_indices, 
    labeled_examples_labels=rand_labeled_examples_labels,
    num_classes=100,
    device=device
)

# Get Subset
proxy = torch.load(f"/home/sjoshi/efficient-contrastive-learning/results/cifar100-resnet50-999-net.pt").module.to(device)
subset_dataset = SASSubsetDataset(
    dataset=CIFAR100Biaugment("/data/cifar100", transform=transforms.ToTensor()), 
    subset_fraction=0.2,
    num_downstream_classes=100,
    device=device,
    proxy_model=net,
    approx_latent_class_partition=partition,
    num_augmentations=10,
    verbose=True
)
```

### SAS (CLIP 0-shot Latent Classes)

```python

# Approximate Latent Classes
import sas 

trainset = co
partition = sas.approximate_latent_classes.clip_approx()


### SAS (k-Means Latent Classes)


### Random Subset

### Custom Subset

## Sample Implementation of Compatible Augmented Dataset (Required for Contrastive Learning)

```python
class CIFAR100Augment(torchvision.datasets.CIFAR100):
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

```
