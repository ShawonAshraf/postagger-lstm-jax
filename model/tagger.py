import flax.linen as nn
from jax import random
import jax.numpy as jnp

"""
class LSTMTagger

vocab_size: int, size of the vocabulary
embedding_dimensions: int, size of the embedding dimensions
lstm_hidden_dims: int, size of the lstm hidden dimensions
n_labels: int, number of labels in the dataset (the padded size, e.g. if padding length is 100, then n_labels = 100)
training: bool, whether the model is in training mode or not
"""


class LSTMTagger(nn.Module):
    vocab_size: int
    embedding_dimensions: int
    lstm_hidden_dims: int
    n_labels: int
    lstm_seed: int
    dropout_rate: float

    def setup(self) -> None:
        # embedding layer
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dimensions,
            name="embedding")

        # lstm layer
        self.lstm = nn.OptimizedLSTMCell(features=self.lstm_hidden_dims, name="lstm")

        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate, deterministic=True)

        # dense layer
        self.dense = nn.Dense(features=self.n_labels, name="dense")

    # lstm in flax: https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.LSTMCell.html
    def __call__(self, words: jnp.ndarray) -> jnp.ndarray:
        x = self.embedding(words)

        carry = self.lstm.initialize_carry(random.key(self.lstm_seed), x.shape)
        carry, x = self.lstm(carry=carry, inputs=x)
        x = self.dropout(x)

        x = self.dense(x)
        x = nn.leaky_relu(x)

        return x
