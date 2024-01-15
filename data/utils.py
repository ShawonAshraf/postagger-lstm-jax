import datasets
from typing import Dict, Tuple

# type aliases
MAP_TYPE = Dict[str, int]
SPLIT_TUPLE = Tuple[datasets.Dataset, datasets.Dataset] | Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]
PREPROCESSED = Tuple[dict, dict, datasets.Dataset, datasets.Dataset, datasets.Dataset]


# a dict containing word -> idx mapping
# pad token is 1 by default
def map_word_to_idx(dataset: datasets.Dataset,
                    pad_token_idx: int = 1) -> MAP_TYPE:
    unique_words = set()
    word_to_idx = dict()

    # add an out of vocab token
    oov_token = "<OOV>"
    # pad token
    pad_token = "<PAD>"

    word_to_idx[oov_token] = 0
    word_to_idx[pad_token] = pad_token_idx

    # find the unique words
    for data in dataset:
        words = data["words"]
        for w in words:
            unique_words.add(w)

    # add index to them
    for idx, uw in enumerate(list(unique_words)):
        word_to_idx[uw] = idx + 2  # since oov is at 0 and pad at pad_token_idx

    return word_to_idx


# pos tag -> idx mapping
def map_label_to_idx(dataset: datasets.Dataset,
                     pad_token_idx: int = 1) -> MAP_TYPE:
    unique_labels = set()
    label_to_idx = dict()

    # add an out of vocab token
    oov_token = "<OOV>"
    # pad token
    pad_token = "<PAD>"

    label_to_idx[oov_token] = 0
    label_to_idx[pad_token] = pad_token_idx

    # find the unique labels
    for data in dataset:
        labels = data["labels"]
        for l in labels:
            unique_labels.add(l)

    # index
    for idx, label in enumerate(list(unique_labels)):
        label_to_idx[label] = idx + 2

    return label_to_idx


# returns train validation splits
# dataset is the train split
def make_train_validation_splits(dataset_split: datasets.Dataset,
                                 validation_split: float = 0.2,
                                 seed: int = 42) -> SPLIT_TUPLE:
    # make a copy of the data
    dataset_split = dataset_split.shuffle(seed=seed)
    # using the train test split method to create validation set
    dataset_split = dataset_split.train_test_split(test_size=validation_split,
                                                   shuffle=True,
                                                   seed=seed)
    return dataset_split["train"], dataset_split["test"]


# returns train validation and test splits
def prepare_splits(dataset: datasets.Dataset,
                   validation_size: float = 0.2,
                   seed: int = 42) -> SPLIT_TUPLE:
    # train and test split
    train_split = dataset["train"]
    test_split = dataset["test"]

    train_split, validation_split = make_train_validation_splits(train_split, validation_size, seed)
    return train_split, validation_split, test_split


# calls word to idx and label to idx map functions, and split functions
# returns word_to_idx, label_to_idx and the splits
def preprocess(dataset: datasets.Dataset,
               pad_token_idx: int = 1,
               validation_size: float = 0.2,
               seed: int = 42) -> PREPROCESSED:
    word_to_idx = map_word_to_idx(dataset["train"], pad_token_idx)
    label_to_idx = map_label_to_idx(dataset["train"], pad_token_idx)

    train_split, validation_split, test_split = prepare_splits(dataset, validation_size, seed)

    return word_to_idx, label_to_idx, train_split, validation_split, test_split
