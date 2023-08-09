# Data Efficient Contrastive Learing (ICML 2023)

## Abstract

Self-supervised learning (SSL) learns high-quality representations from large pools of unlabeled training data. As datasets grow larger, it becomes crucial to identify the examples that contribute the most to learning such representations. This enables efficient SSL by reducing the volume of data required for learning high-quality representations. Nevertheless, quantifying the value of examples for SSL has remained an open question. In this work, we address this for the first time, by proving that examples that contribute the most to contrastive SSL are those that have the most similar augmentations to other examples, in expectation. We provide rigorous guarantees for the generalization performance of SSL on such subsets. Empirically, we discover, perhaps surprisingly, the subsets that contribute the most to SSL are those that contribute the least to supervised learning. Through extensive experiments, we show we can safely exclude 20% of examples from CIFAR100 and 40% from STL10 and TinyImageNet, without affecting downstream task performance. We also show that our subsets outperform random subsets by more than 2% on CIFAR10. We also demonstrate that these subsets are effective across contrastive SSL methods (evaluated on SimCLR, MoCo, SimSiam, BYOL).

[Project Page](https://sjoshi804.github.io/data-efficient-contrastive-learning/)

[Paper](https://arxiv.org/abs/2302.09195)

## BibTex Citation

Please cite this if you use this code / paper in your work.

```bibtex
@InProceedings{pmlr-v202-joshi23b,
  title = {Data-Efficient Contrastive Self-supervised Learning: Most Beneficial Examples for Supervised Learning Contribute the Least},
  author = {Joshi, Siddharth and Mirzasoleiman, Baharan},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {15356--15370},
  year = {2023},
  editor = {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = {202},
  series = {Proceedings of Machine Learning Research},
  month = {23--29 Jul},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v202/joshi23b/joshi23b.pdf},
  url = {https://proceedings.mlr.press/v202/joshi23b.html},
}
```

## Examples

Example subsets can be found in `/examples`

Subsets Provided:

- CIFAR100 - 20%, 40%, 60%, 80% subsets
- STL10 - 20%, 40%, 60%, 80% subsets
- CIFAR10 - 5%, 10%, 15%, 20% subsets
- TinyImageNet (coming soon)
- ImageNet (coming soon)

To get the subset indices:

```python
import pickle

with open(f"<subset-file-name>", "rb") as f:
    subset_indices = pickle.load(f)
```

And then pass this to CustomSubsetDataset as the subset indices argument to get the corresponding subset dataset object. (More details below)

## Sample Usage

See `cifar100_example_subset_creation.ipynb` for a complete example of how to create a subset with proxy models provided. 

```bash
pip install sas-pip/
pip install -r requirements.txt
```

Samples shown for choosing subsets of CIFAR100

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

partition = clip_approx(
    img_trainset=cifar100,
    labeled_example_indices=rand_labeled_examples_indices, 
    labeled_examples_labels=rand_labeled_examples_labels,
    num_classes=100,
    device=device
)

# Get Subset
proxy_model = torch.load(f"cifar100-proxy-encoder.pt").module.to(device)
subset_dataset = SASSubsetDataset(
    dataset=cifar100,
    subset_fraction=0.2,
    num_downstream_classes=100,
    device=device,
    proxy_model=proxy_model,
    approx_latent_class_partition=partition,
    verbose=True
)
```

### SAS (CLIP 0-shot Latent Classes)

```python
# Approximate Latent Classes
from sas.approx_latent_classes import clip_0shot_approx

partition = clip_0shot_approx(
    img_trainset=cifar100,
    class_names=cifar100_classes,
    device=device
)
```

### SAS (k-Means Latent Classes)

```python
# Approximate Latent Classes
from sas.approx_latent_classes import kmeans_approx

partition = kmeans_approx(
    trainset=cifar100,
    proxy_model=net, 
    num_classes=100,
    device=device
)
```

### Random Subset

```python
from sas.subset_dataset import RandomSubsetDataset

cifar100 = torchvision.datasets.CIFAR100("/data/cifar100/", transform=transforms.ToTensor())
subset_dataset = RandomSubsetDataset(cifar100, subset_fraction=0.2)

```

### Custom Subset

```python
from sas.subset_dataset import CustomSubset

cifar100 = torchvision.datasets.CIFAR100("/data/cifar100/", transform=transforms.ToTensor())
subset_dataset = CustomSubsetDataset(cifar100, subset_indices=range(10000))
```

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
