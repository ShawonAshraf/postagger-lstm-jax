import jax.numpy as jnp
from tqdm.auto import tqdm

"""
categorical accuracy
find the number of overlapping tag matches, excluding padding
"""


def categorical_accuracy(preds, actual, pad_idx=1):
    non_padding_indices = jnp.nonzero((actual != pad_idx))

    matches = jnp.equal(preds[non_padding_indices], actual[non_padding_indices])
    acc = jnp.sum(matches) / actual[non_padding_indices].shape[0]

    return acc


def evaluate(params, test_loader, batch_infer):
    acc_per_batch = list()
    for batch in tqdm(test_loader):
        words, labels = batch
        preds = batch_infer(params, words, labels)

        acc = categorical_accuracy(preds, labels)
        acc_per_batch.append(acc)

    mean_acc = jnp.mean(jnp.array(acc_per_batch), axis=-1)

    return mean_acc
