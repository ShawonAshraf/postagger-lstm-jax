import jax
from jax import jit, vmap, value_and_grad
import jax.numpy as jnp
from tqdm.auto import trange
from flax.training import train_state
import optax
import flax.linen as nn
from typing import Any, Callable
from tqdm.auto import tqdm

"""
class Trainer

model: nn.Module, the model to train
init_params: dict, initial parameters of the model
lr: float, learning rate
max_epochs: int, number of epochs to train for
log_every_n_step: int, log every n steps
model_rngs: dict, random number generators for the model
logger: Any, logger to log to (wandb)
"""


class Trainer:
    def __init__(self, model: nn.Module,
                 init_params: dict,
                 lr: float,
                 max_epochs: int,
                 log_every_n_step: int,
                 model_rngs: dict,
                 logger: Any) -> None:

        self.model = model
        self.init_params = init_params
        self.model_rngs = model_rngs
        self.lr = lr
        self.max_epochs = max_epochs
        self.log_every_n_step = log_every_n_step
        self.logger = logger

        # optimiser, Adam
        self.optimiser = optax.adam(lr)

        # create util functions for training the model
        self.__create_utils()

    def __create_utils(self):
        # ================== #
        # loss function and criterion #
        @jit
        def softmax_ce(params, words, labels):
            logits = self.model.apply(params, words, rngs={"dropout": self.model_rngs["dropout"]})
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            # mean over all tag dimensions
            # for k tags, k dimensions in cross entropy, so mean over k dimensions
            return loss.mean(axis=-1)

        @jit
        def batched_softmax_ce(params, words_batched, labels_batched):
            # vmap over batch
            batch_loss = vmap(softmax_ce, in_axes=(None, 0, 0))(params, words_batched, labels_batched)

            # return mean over batch
            return batch_loss.mean(axis=-1)

        self.loss_fn = batched_softmax_ce
        self.gradient_computer = jit(value_and_grad(self.loss_fn))

        # ================== #
        # training step #
        @jit
        def train_step(state, batch):
            loss_value, grads = self.gradient_computer(state.params, *batch)
            updated_state = state.apply_gradients(grads=grads)
            return loss_value, updated_state

        self.train_step = train_step

        # ================== #
        # validation step #
        @jit
        def validation_step(state, batch):
            loss_value = self.loss_fn(state.params, *batch)
            return loss_value

        self.validation_step = validation_step

        # ================== #
        # accuracy #
        def categorical_accuracy(preds, actual, pad_idx=1):
            non_padding_indices = jnp.nonzero((actual != pad_idx))

            matches = jnp.equal(preds[non_padding_indices], actual[non_padding_indices])
            acc = jnp.sum(matches) / actual[non_padding_indices].shape[0]

            return acc

        # ================== #
        # inference #
        @jit
        def infer(params, words):
            logits = self.model.apply(params, words, rngs={"dropout": self.model_rngs["dropout"]})
            proba = jax.nn.log_softmax(logits, axis=-1)
            preds = jnp.argmax(proba, axis=-1)

            return preds

        @jit
        def batch_infer(params, words_batched):
            preds = vmap(infer, in_axes=(None, 0))(params, words_batched)
            return preds

        # ================== #
        # evaluate #
        def evaluate(params, test_loader):
            acc_per_batch = list()
            for batch in tqdm(test_loader):
                words, labels = batch
                preds = batch_infer(params, words)

                acc = categorical_accuracy(preds, labels)
                acc_per_batch.append(acc)

            mean_acc = jnp.mean(jnp.array(acc_per_batch), axis=-1)

            return mean_acc

        self.evaluate = evaluate

    def fit_and_eval(self, train_loader, val_loader, test_loader):
        # initialise the train state
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,  # the forward function
            params=self.init_params,
            tx=self.optimiser
        )

        # for logging
        train_losses = list()
        mean_validation_losses = list()
        step_counter = 0

        # ================== #
        # train loop
        for _ in trange(self.max_epochs):

            # ================== #
            # epoch start #
            for batch in train_loader:
                # loss and updated state
                loss_value, state = self.train_step(state, batch)
                step_counter += 1

                # no gradient computation for validation step
                # so only loss will be collected

                if step_counter % self.log_every_n_step == 0:
                    # collect train loss
                    train_losses.append(loss_value)

                    # ================== #
                    # run validation
                    validation_losses = list()
                    for validation_batch in val_loader:
                        val_loss = self.validation_step(state, validation_batch)
                        validation_losses.append(val_loss)

                    # collect all validation losses for this batch
                    validation_losses = jnp.array(validation_losses)
                    # mean over all losses for this batch
                    mean_validation_losses.append(validation_losses.mean(axis=-1))

                    # log
                    self.logger.log({
                        "train_loss": loss_value,
                        # mean over all validation batches
                        "val_loss": validation_losses.mean(axis=-1),
                        "step": step_counter
                    })

                    # ================== #
                    # evaluation
                    train_accuracy = self.evaluate(state.params, train_loader)
                    val_accuracy = self.evaluate(state.params, val_loader)
                    test_accuracy = self.evaluate(state.params, test_loader)

                    # log
                    self.logger.log({
                        "train_accuracy": train_accuracy,
                        "val_accuracy": val_accuracy,
                        "test_accuracy": test_accuracy,
                        "step": step_counter
                    })

        # return trained state
        return state
