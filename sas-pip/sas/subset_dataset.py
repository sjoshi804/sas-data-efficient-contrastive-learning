from abc import ABC
from typing import Dict, List, Optional
import math 
import pickle
import random 

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sas.submodular_maximization import lazy_greedy
from tqdm import tqdm

# Efficient alternate implementation of np.block
def efficient_block(mat_list:List[List[np.ndarray]]):
    if type(mat_list[0]) is not list:
        mat_list = [mat_list]

    x_size = 0
    y_size = 0

    for i in mat_list[0]:
        x_size += i.shape[1]

    for j in range(len(mat_list)):
        y_size += mat_list[j][0].shape[0]

    output_data = np.zeros((y_size, x_size))

    x_cursor = 0
    y_cursor = 0

    for mat_row in mat_list:
        y_offset = 0

        for matrix_ in mat_row:
            shape_ = matrix_.shape
            output_data[y_cursor: y_cursor + shape_[0], x_cursor: x_cursor + shape_[1]] = matrix_
            x_cursor += shape_[1]
            y_offset = shape_[0]

        y_cursor += y_offset
        x_cursor = 0

    return output_data

"""
Base Subset Dataset (Abstract Base Class)
"""
class BaseSubsetDataset(ABC, Dataset):
    def __init__(
        self,
        dataset: Dataset,
        subset_fraction: float,
        verbose: bool = False
    ):
        """
        :param dataset: Original Dataset
        :type dataset: Dataset
        :param subset_fraction: Fractional size of subset
        :type subset_fraction: float
        :param verbose: verbose
        :type verbose: boolean
        """
        self.dataset = dataset
        self.subset_fraction = subset_fraction
        self.len_dataset = len(self.dataset)
        self.subset_size = int(self.len_dataset * self.subset_fraction)
        self.subset_indices = None
        self.verbose = verbose 

    def initialization_complete(self):
        if self.verbose:
            print(f"Subset Size: {self.subset_size}")
            print(f"Discarded {self.len_dataset - self.subset_size} examples")

    def __len__(self):
        return self.subset_size
    
    def __getitem__(self, index):
        # Get the index for the corresponding item in the original dataset
        original_index = self.subset_indices[index]
        
        # Get the item from the original dataset at the corresponding index
        original_item = self.dataset[original_index]
        
        return original_item
    
    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.subset_indices, f)

"""
Random Subset
"""
class RandomSubsetDataset(BaseSubsetDataset):
    def __init__(
        self,
        dataset: Dataset,
        subset_fraction: float,
        partition: Optional[Dict[int, List[int]]] = None,
        verbose: bool = False
    ):
        """
        :param dataset: Original Dataset
        :type dataset: Dataset
        :param subset_fraction: Fractional size of subset
        :type subset_fraction: float
        :param verbose: verbose
        :type verbose: boolean
        """
        super().__init__(
            dataset=dataset, 
            subset_fraction=subset_fraction,
            verbose=verbose
        )
        
        self.subset_indices = []
        if partition is not None:
            if self.verbose:
                print("Partition provided => returning balanced random subset from all latent classes")
            self.subset_indices = RandomSubsetDataset.get_random_balanced_indices(partition, subset_fraction)
        else:   
            if self.verbose:
                print("No partition => random subset from full data")
            self.subset_indices = random.sample(range(self.len_dataset), self.subset_size)
        self.initialization_complete()
        
    def __len__(self):
        return self.subset_size
    
    def __getitem__(self, index):
        # Get the index for the corresponding item in the original dataset
        original_index = self.subset_indices[index]
        
        # Get the item from the original dataset at the corresponding index
        original_item = self.dataset[original_index]
        
        return original_item
    
    @staticmethod
    def get_random_balanced_indices(partition: Dict[int, List[int]], subset_fraction: float):
        """
        Randomly selects a subset of fractional size = 'subset_fraction' from each latent class in partition.
        The subset selected from each list is the same fraction of the whole.

        Parameters:
        - partition: Dict[int, List[int]]
        - subset_fraction: float

        Returns:
        - selected_subset: List containing the selected subset.
        """
        
        def random_subset_with_fixed_size(original_list, subset_size):
            subset_size = min(subset_size, len(original_list))
            return random.sample(original_list, subset_size)

        selected_subset = []

        for key in partition.keys():
            subset_size = int(len(partition[key]) * subset_fraction)
            subset = random_subset_with_fixed_size(partition[key], subset_size)
            selected_subset.extend(subset)

        return selected_subset

"""
Custom Subset
"""
class CustomSubsetDataset(BaseSubsetDataset):
    def __init__(
        self,
        dataset: Dataset,
        subset_indices: List[int],
        verbose: bool = False,
    ):
        """
        :param dataset: Original Dataset
        :type dataset: Dataset
        :param subset_fraction: Fractional size of subset
        :type subset_fraction: float
        :param subset_indices: Indices of custom subset
        :type subset_indices: List[int]
        :param verbose: verbose
        :type verbose: boolean
        """
        super().__init__(
            dataset=dataset, 
            subset_fraction=1.0,
            verbose=verbose
        )
        self.subset_size = len(subset_indices)
        self.subset_fraction = self.subset_size / len(dataset)
        self.subset_indices = subset_indices
        self.initialization_complete()

"""
Subsets that maximize Augmentation Similarity Subset Dataset
"""
class SubsetSelectionObjective:
    def __init__(self, distance, threshold=0, verbose=False):
        '''
        :param distance: (n, n) matrix specifying pairwise augmentation distance
        :type distance: np.array
        :param threshold: minimum cosine similarity to consider to be significant (default=0)
        :type threshold: float
        '''
        self.distance = distance 
        self.threshold = threshold
        self.verbose = verbose
        if self.verbose:
            print("Masking pairwise distance matrix")
        for i in range(len(self.distance)):        
            self.distance[i] *= (self.distance[i] >= self.threshold)

    def inc(self, sset, i):
        return np.sum(self.distance[i]) - np.sum(self.distance[np.ix_(sset, [i])])
    
    def add(self, i):
        self.distance[:][i] = 0
        return 
    
class SASSubsetDataset(BaseSubsetDataset):
    def __init__(
        self,
        dataset: Dataset,
        subset_fraction: float,
        num_downstream_classes: int,
        device: torch.device,
        approx_latent_class_partition: Dict[int, int],
        proxy_model: Optional[nn.Module] = None,
        augmentation_distance: Optional[Dict[int, np.array]] = None,
        num_runs=1,
        pairwise_distance_block_size: int = 1024, 
        threshold: float = 0.0,
        verbose: bool = False
    ):
        """
        dataset: Dataset
            Original dataset for contrastive learning. Assumes that dataset[i] returns a list of augmented views of the original example i.

        subset_fraction: float
            Fractional size of subset.

        num_downstream_classes: int
            Number of downstream classes (can be an estimate).

        proxy_model: nn.Module
            Proxy model to calculate the augmentation distance (and kmeans clustering if the avoid clip option is chosen).

        augmentation_distance: Dict[int, np.array]
            Pass a precomputed dictionary containing augmentation distance for each latent class.

        num_augmentations: int
            Number of augmentations to consider while approximating the augmentation distance.

        pairwise_distance_block_size: int
            Block size for calculating pairwise distance. This is just to optimize GPU usage while calculating pairwise distance and will not affect the subset created in any way.

        verbose: boolean
            Verbosity of the output.
        """
        super().__init__(
            dataset=dataset, 
            subset_fraction=subset_fraction,
            verbose=verbose
        )
        self.device = device
        self.num_downstream_classes = num_downstream_classes
        self.proxy_model = proxy_model
        self.partition = approx_latent_class_partition
        self.augmentation_distance = augmentation_distance
        self.num_runs = num_runs
        self.pairwise_distance_block_size = pairwise_distance_block_size
        print("Here1")
        if self.augmentation_distance == None:
            self.augmentation_distance = self.approximate_augmentation_distance()

        print("Here2")
        class_wise_idx = {}
        for latent_class in tqdm(self.partition.keys(), desc="Subset Selection", disable=not verbose):
            F = SubsetSelectionObjective(self.augmentation_distance[latent_class].copy(), threshold=threshold, verbose=self.verbose)
            class_wise_idx[latent_class] = lazy_greedy(F, range(len(self.augmentation_distance[latent_class])), len(self.augmentation_distance[latent_class]), verbose=self.verbose)
            class_wise_idx[latent_class] = [self.partition[latent_class][i] for i in class_wise_idx[latent_class]]
        
        print("Here3")
        self.subset_indices = []
        for latent_class in class_wise_idx.keys():
            l = len(class_wise_idx[latent_class])
            self.subset_indices.extend(class_wise_idx[latent_class][:int(self.subset_fraction * l)])

        self.initialization_complete()


    def approximate_augmentation_distance(self):
        self.proxy_model = self.proxy_model.to(self.device)

        # Initialize augmentation distance with all 0s
        augmentation_distance = {}
        Z = self.encode_trainset()
        for latent_class in tqdm(list(self.partition.keys()), desc="Computing augmentation distance", disable=not(self.verbose)):
            Z_partition = Z[self.partition[latent_class]]
            pairwise_distance = SASSubsetDataset.pairwise_distance(Z_partition, Z_partition, verbose=self.verbose)
            augmentation_distance[latent_class] = pairwise_distance.copy()
        return augmentation_distance

    def encode_trainset(self):
        trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.pairwise_distance_block_size, shuffle=False, num_workers=2, pin_memory=True)
        with torch.no_grad():
            Z = []
            for input in  tqdm(trainloader, desc="Encoding trainset", disable=not(self.verbose)):
                Z.append(self.proxy_model(input[0].to(self.device)))
        return torch.cat(Z, dim=0)
    
    def encode_augmented_trainset(self, num_positives=1):
        trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.pairwise_distance_block_size, shuffle=False, num_workers=2, pin_memory=True)
        with torch.no_grad():
            Z = []
            for _ in range(num_positives):
                Z.append([])
            for X in tqdm(trainloader, desc="Encoding augmented trainset", disable=not(self.verbose)):
                for j in range(num_positives):
                    Z[j].append(self.proxy_model(X[j].to(self.device)))
        for i in range(num_positives):
            Z[i] = torch.cat(Z[i], dim=0)
        Z = torch.cat(Z, dim=0)
        return Z

    @staticmethod
    def pairwise_distance(Z1: torch.tensor, Z2: torch.tensor, block_size: int = 1024, verbose=False):
        similarity_matrices = []
        for i in tqdm(range(Z1.shape[0] // block_size + 1), desc="Computing pairwise distances", disable=not(verbose)):
            similarity_matrices_i = []
            e = Z1[i*block_size:(i+1)*block_size]
            for j in range(Z2.shape[0] // block_size + 1):
                e_t = Z2[j*block_size:(j+1)*block_size].t()
                similarity_matrices_i.append(
                    np.array(
                    torch.cosine_similarity(e[:, :, None], e_t[None, :, :]).detach().cpu()
                    )
                )
            similarity_matrices.append(similarity_matrices_i)
        similarity_matrix = efficient_block(similarity_matrices)

        return similarity_matrix
