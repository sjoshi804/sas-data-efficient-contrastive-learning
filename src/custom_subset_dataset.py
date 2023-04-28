from typing import List
from torch.utils.data import Dataset

from base_subset_dataset import BaseSubsetDataset

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