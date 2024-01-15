from jax import random, jit, vmap, grad, value_and_grad
import jax.numpy as jnp
from functools import partial
from tqdm.auto import trange
from flax.training import train_state
import optax


@partial(jit, static_argnums=0)
def calculate_loss(model, dropout_key, params, words, labels):
    logits = model.apply(params, words, rngs={"dropout": dropout_key})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean(axis=-1)


@partial(jit, static_argnums=0)
def batched_loss(model, params, words_batched, labels_batched):
    batch_loss = vmap(calculate_loss, in_axes=(None, None, 0, 0))(model, params, words_batched, labels_batched)
    return batch_loss.mean(axis=-1)


@partial(jit, static_argnums=(0, 2))
def train_step(model, dropout_key, criterion, state, words_batched, labels_batched):
    loss_value, grads = criterion(model, dropout_key, state.params, words_batched, labels_batched)
    updated_state = state.apply_gradients(grads=grads)
    return loss_value, updated_state


@partial(jit, static_argnums=(0, 2))
def validation_step(model, dropout_key, criterion, state, words_batched, labels_batched):
    loss_value, _ = criterion(model, dropout_key, state.params, words_batched, labels_batched)
    return loss_value


