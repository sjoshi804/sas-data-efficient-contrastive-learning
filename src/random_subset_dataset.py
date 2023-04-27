import random
from torch.utils.data import Dataset
from base_subset_dataset import BaseSubsetDataset

class RandomSubsetDataset(BaseSubsetDataset):
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
        super().__init__(
            dataset=dataset, 
            subset_fraction=subset_fraction,
            verbose=verbose
        )
        self.subset_indices = random.sample(range(self.len_dataset), self.subset_size)
        if self.verbose: 
            print("Subset Dataset Ready.")
        
    def __len__(self):
        return self.subset_size
    
    def __getitem__(self, index):
        # Get the index for the corresponding item in the original dataset
        original_index = self.subset_indices[index]
        
        # Get the item from the original dataset at the corresponding index
        original_item = self.dataset[original_index]
        
        return original_item