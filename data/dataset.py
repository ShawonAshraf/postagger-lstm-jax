import datasets

from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

"""
class TagDataset

dataset_split: a datasets.Dataset object, which contains the train / test / validation split
pad_token_idx: int, the padding token index in the vocabulary, used for padding
max_seq_len: int, maximum sequence length to pad all sequences to
"""


class TagDataset(Dataset):
    def __init__(self, dataset_split: datasets.Dataset,
                 pad_token_idx: int,
                 max_seq_len: int,
                 word_to_idx: dict,
                 label_to_idx: dict) -> None:
        self.dataset = dataset_split
        self.pad_token_idx = pad_token_idx
        self.max_seq_len = max_seq_len
        self.word_to_idx = word_to_idx
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    # use word_to_idx and label_to_idx to convert
    # the string sequences to int sequences
    def __encode(self, data_instance: dict) -> Tuple[list, list]:
        words = data_instance["words"]
        labels = data_instance["labels"]

        # convert to int sequences
        words = [self.word_to_idx.get(w, 0) for w in words]
        labels = [self.label_to_idx.get(l) for l in labels]

        return words, labels

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        data = self.dataset[index]
        words, labels = self.__encode(data)

        # padding
        words_padded = np.ones((self.max_seq_len,), dtype=np.int32) * self.pad_token_idx
        words_padded[:len(words)] = words

        labels_padded = np.ones((self.max_seq_len,), dtype=np.int32) * self.pad_token_idx
        labels_padded[:len(labels)] = labels

        # return padded words and labels
        return words_padded, labels_padded
