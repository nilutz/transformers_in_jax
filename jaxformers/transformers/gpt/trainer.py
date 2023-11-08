from typing import Callable

import flax.linen as nn
import optax
from flax.training import train_state
from flax.training.train_state import TrainState
from jaxtyping import Float, PRNGKeyArray
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import random
from jaxformers.trainer import DefaultTrainer

BatchType = tuple


class GPTTrainerModule(DefaultTrainer):
    def __init__(
        self,
        model,
        model_name: str,
        config,
        init_batch: tuple,
    ):
        super().__init__(
            model=model, model_name=model_name, config=config, init_batch=init_batch
        )

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(
            params: dict,
            rng: PRNGKeyArray,
            batch: BatchType,
            train: bool = True,
        ) -> tuple[Float, tuple[Float, Float, Float, PRNGKeyArray]]:
            # data
            (
                src,
                targets,
            ) = batch

            rng, dropout_apply_rng = random.split(rng)
            src_mask = nn.make_causal_mask(src, dtype=bool)

            logits = self.model.apply(
                {"params": params},
                src,
                src_mask,
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )

            flat_logits = logits.reshape((-1, logits.shape[-1]))
            flat_targets = targets.reshape((-1,))

            # Calculate cross-entropy loss
            loss = -jnp.sum(
                jax.nn.log_softmax(flat_logits)
                * jax.nn.one_hot(flat_targets, logits.shape[-1]),
                axis=-1,
            )

            # Mask out the ignored index
            mask = flat_targets != -1  # TODO which index ? shouldn't that be 0
            loss = jnp.where(mask, loss, 0.0)

            # Compute the average loss
            loss = jnp.sum(loss) / jnp.sum(mask)

            preds = jnp.argmax(logits, axis=-1)
            acc = jnp.equal(preds, targets)
            acc = acc.sum()

            return loss, (acc, rng)

        return calculate_loss

    def create_functions(self):
        # Create jitted train and eval functions
        calculate_loss = self.get_loss_function()

        # Training function
        def train_step(
            state: TrainState, rng: PRNGKeyArray, batch
        ) -> tuple[TrainState, PRNGKeyArray, Float, Float]:
            loss_fn = lambda params: calculate_loss(params, rng, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, rng = (
                ret[0],
                *ret[1],
            )
            state = state.apply_gradients(grads=grads)
            # jax.debug.print("loss2 {loss}", loss= loss)

            return (
                state,
                rng,
                loss,
                acc,
            )

        self.train_step = jax.jit(train_step)

        # Evaluation function
        def eval_step(state, rng, batch):
            loss, (
                acc,
                rng,
            ) = calculate_loss(state.params, rng, batch, train=False)
            return loss, acc, rng

        self.eval_step = jax.jit(eval_step)

    def init_model(
        self,
        init_batch: BatchType,
    ):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)
        src, _ = init_batch

        src_mask = nn.make_causal_mask(src, dtype=bool)

        params = self.model.init(
            {
                "params": init_rng,
                "dropout": dropout_init_rng,
            },
            src,
            src_mask,
            train=True,
        )[
            "params"
        ]  # Apply transform

        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.lr,
            warmup_steps=self.config.warmup,
            decay_steps=self.config.train_steps,
            end_value=0.0,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adamw(lr_schedule),
        )
        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )
        print("... model initialized")

    def train_model(
        self,
        get_batch: Callable,
        eval_step: int = 1,
    ):
        best_val_loss = 0.0
        train_start = 0  # TODO for continue training
        for step in tqdm(range(train_start, self.config.train_steps)):
            X, Y = get_batch(split="train")
            (
                self.state,
                self.rng,
                loss,
                acc,
            ) = self.train_step(self.state, self.rng, (X, Y))

            # jax.debug.print("loss4 {loss}", loss= loss)
            dloss = jax.device_get(loss)
            # jax.debug.print("dloss {dloss}", dloss= dloss)
            # jax.debug.print("step {s}", s= step)

            self.logger.add_scalar("train/loss", dloss, global_step=step)
            self.logger.add_scalar("train/acc", jax.device_get(acc), global_step=step)
            self.logger.flush()

            if step % eval_step == 0:
                loss, acc = self.eval_model(get_batch)
                self.logger.add_scalar(
                    "val/loss", jax.device_get(loss), global_step=step
                )
                self.logger.add_scalar("val/acc", jax.device_get(acc), global_step=step)
                self.logger.flush()

                if loss >= best_val_loss:
                    best_val_loss = loss
                    self.save_model(step=step)

        self.save_model(step=step + 1)

    def eval_model(self, get_batch: Callable):
        X, Y = get_batch(split="validation")
        (
            loss,
            acc,
            self.rng,
        ) = self.eval_step(self.state, self.rng, (X, Y))

        return loss, acc
