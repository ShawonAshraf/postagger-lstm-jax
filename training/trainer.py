from jax import random, jit, vmap, grad, value_and_grad
import jax.numpy as jnp
from tqdm.auto import trange
from flax.training import train_state
import optax
from functools import partial


class Trainer:
    def __init__(self, model,
                 init_params,
                 lr,
                 max_epochs,
                 log_every_n_step,
                 model_rngs):
        self.model = model
        self.init_params = init_params
        self.model_rngs = model_rngs
        self.lr = lr
        self.max_epochs = max_epochs
        self.optimiser = optax.adam(lr)
        self.log_every_n_step = log_every_n_step

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

    def train(self, train_loader, val_loader):
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,  # the forward function
            params=self.init_params,
            tx=self.optimiser
        )

        train_losses = list()
        mean_validation_losses = list()
        step_counter = 0

        for _ in trange(self.max_epochs):
            for batch in train_loader:
                loss_value, state = self.train_step(state, batch)
                step_counter += 1

                if step_counter % self.log_every_n_step == 0:
                    # log train loss
                    train_losses.append(loss_value)
                    # run validation
                    vlosses = list()
                    for vbatch in val_loader:
                        val_loss = self.validation_step(state, vbatch)
                        vlosses.append(val_loss)

                    vlosses = jnp.array(vlosses)
                    mean_validation_losses.append(vlosses.mean(axis=-1))

                    print(
                        f"Step [{step_counter + 1}] ---- Loss/Train :: {loss_value} ---- Loss/Val :: {vlosses.mean(axis=-1)}")

        return state, train_losses, mean_validation_losses
