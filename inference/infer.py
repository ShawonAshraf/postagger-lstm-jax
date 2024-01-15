from jax import random, jit, vmap, grad, value_and_grad
import jax.numpy as jnp
from functools import partial

import optax
import jax.nn as jnn
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key


@partial(jit, static_argnums=0)
def infer(model, dropout_key, params, words):
    logits = model.apply(params, words, rngs={"dropout": dropout_key})
    proba = jnn.log_softmax(logits, axis=-1)
    preds = jnp.argmax(proba, axis=-1)

    return preds


@partial(jit, static_argnums=0)
def batch_infer(model, dropout_key, params, words, labels):
    preds = vmap(infer, in_axes=(None, None, None, 0, 0))(params, words, labels)
    return preds
