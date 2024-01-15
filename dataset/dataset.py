import datasets

from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


"""
class TagDataset

indices: list or a numpy ndarray, which contains the indices to load from the dataset
        applicable for the train and validation split. For the test split, indices will be none.
        
        reason: 'batterydata/pos_tagging' dataset provides test and train splits. the validation split
                will need to be created from the train split.
        
dataset: a datasets.Dataset object, which contains the whole dataset
pad_token_idx: int, the padding token index in the vocabulary, used for padding
max_seq_len: int, maximum sequence length to pad all sequences to
"""


class TagDataset(Dataset):
    def __init__(self, indices: list | np.ndarray | None,
                 dataset: datasets.Dataset,
                 pad_token_idx: int,
                 max_seq_len: int) -> None:
        self.indices = indices
        self.dataset = dataset
        self.pad_token_idx = pad_token_idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        if self.indices is None:
            # this is for the test split
            return len(self.dataset)
        else:
            return len(self.indices)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        if self.indices is None:
            idx = index
        else:
            idx = self.indices[index]

        data = self.dataset[idx]

        # padding to 300
        # pad token idx is 1
        words = np.ones((300,), dtype=np.int32)
        words[:len(data["words"])] = data["words"]

        labels = np.ones((300,), dtype=np.int32)
        labels[:len(data["labels"])] = data["labels"]

        return words, labels
