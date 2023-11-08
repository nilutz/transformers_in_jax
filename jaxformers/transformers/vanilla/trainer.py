import numpy as np
import optax

from flax.training import train_state
from flax.training.train_state import TrainState
from jaxtyping import Array, Bool, Float, Int32, PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import random
from jaxformers.modules.label_smoothing import LabelSmoothing
from jaxformers.trainer import DefaultTrainer


class VanillaTrainerModule(DefaultTrainer):
    def __init__(
        self,
        model,
        model_name: str,
        config,
        init_batch: tuple[
            Int32[Array, "batch seq_len"],
            Int32[Array, "batch seq_len"],
            Bool[Array, "batch 1 1 seq_len"],
            Bool[Array, "batch 1 seq_len seq_len"],
        ],
        max_iters: int,
        padding_index: int,
    ):
        """
        Args:
            model:  A Vanilla Transformers model
            model_name: A name for the model for storage
            exmp_batch: tuple containing src, tgt batch
        """
        self.max_iters: int = max_iters
        self.padding_index: int = padding_index

        super().__init__(
            model=model, model_name=model_name, config=config, init_batch=init_batch
        )

        self.criterion = LabelSmoothing(
            smoothing=config.label_smoothing,
            padding_idx=self.padding_index,
        )

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(
            params: dict,
            rng: PRNGKeyArray,
            batch: tuple[
                Int32[Array, "batch_size seq_len"],
                Int32[Array, "batch_size seq_len"],
                Int32[Array, "batch_size 1 1 seq_len"],
                Int32[Array, "batch_size 1 seq_len seq_len"],
            ],
            train: bool,
        ) -> tuple[Float, Float, PRNGKeyArray]:
            src, tgt, src_mask, tgt_mask = batch

            rng, dropout_apply_rng = random.split(rng)

            shifted_tgt = tgt[:, :-1]
            shifted_tgt_mask = tgt_mask[:, :, :-1, :-1]
            # Apply to network
            _, logits = self.model.apply(
                {"params": params},
                src,
                shifted_tgt,  # shift
                src_mask,
                shifted_tgt_mask,  # shift
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )
            weights = jnp.where(tgt[:, 1:] != self.padding_index, 1, 0).astype(
                jnp.float32
            )
            loss, weight_sum = self.criterion(logits, tgt[:, 1:], weights)
            mean_loss = loss / weight_sum

            # simple accuracy
            preds = jnp.argmax(logits, axis=-1)
            acc = jnp.equal(preds, tgt[:, 1:])
            if weights is not None:
                acc = acc * weights
            acc = acc.sum()

            return mean_loss, (acc, loss, rng)

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
            loss, acc, sloss, rng = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads)
            return state, rng, loss, acc, sloss

        self.train_step = jax.jit(train_step)

        # Evaluation function
        def eval_step(state, rng, batch):
            loss, (acc, sloss, rng) = calculate_loss(
                state.params, rng, batch, train=False
            )
            return loss, acc, sloss, rng

        self.eval_step = jax.jit(eval_step)

    def init_model(
        self,
        init_batch: tuple[
            Int32[Array, "batch seq_len"],
            Int32[Array, "batch seq_len"],
            Bool[Array, "batch 1 1 seq_len"],
            Bool[Array, "batch 1 seq_len seq_len"],
        ],
    ):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)
        src, tgt, src_mask, tgt_mask = init_batch
        shifted_tgt = tgt[:, :-1]
        shifted_tgt_mask = tgt_mask[:, :, :-1, :-1]

        params = self.model.init(
            {
                "params": init_rng,
                "dropout": dropout_init_rng,
            },
            src,
            shifted_tgt,  # shifted
            src_mask,
            shifted_tgt_mask,  # shifted
            train=True,
        )[
            "params"
        ]  # Apply transform

        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.lr,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.max_iters,
            end_value=0.0,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adam(
                b1=self.config.adam_beta,
                b2=self.config.adam_beta2,
                eps=self.config.adam_epsilon,
                learning_rate=lr_schedule,
            ),
        )
        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )
        print("... model initialized")

    def train_model_multi_device(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 500,
        eval_step: int = 1,
    )

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 500,
        eval_step: int = 1,
    ):
        best_acc = 0.0
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.train_epoch(train_loader, epoch=epoch_idx, total_epochs=num_epochs)
            if epoch_idx % eval_step == 0:
                eval_acc, avg_loss, avg_sloss = self.eval_model(val_loader)
                self.logger.add_scalar("val/accuracy", eval_acc, global_step=epoch_idx)
                self.logger.add_scalar("val/loss", avg_loss, global_step=epoch_idx)
                self.logger.add_scalar("val/sloss", avg_sloss, global_step=epoch_idx)

                if eval_acc >= best_acc:
                    best_acc = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()
        self.save_model(step=epoch_idx + 1)

    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int):
        accs, losses, slosses = [], [], []
        for i, batch in enumerate(
            tqdm(
                train_loader,
                unit="batch",
                desc=f"Epoch: {epoch}/{total_epochs} ",
                bar_format="{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}",
                ascii=" #",
            )
        ):
            self.state, self.rng, loss, acc, sloss = self.train_step(
                self.state, self.rng, batch
            )
            losses.append(loss)
            accs.append(acc)
            slosses.append(sloss)
            self.logger.add_scalar(f"batch/{epoch}/loss", np.array(loss), global_step=i)
            self.logger.add_scalar(
                f"batch/{epoch}/accuracy", np.array(acc), global_step=i
            )
            self.logger.flush()

        avg_loss = np.stack(jax.device_get(losses)).mean()
        avg_acc = np.stack(jax.device_get(accs)).mean()
        avg_sloss = np.stack(jax.device_get(slosses)).mean()

        self.logger.add_scalar("train/loss", avg_loss, global_step=epoch)
        self.logger.add_scalar("train/accuracy", avg_acc, global_step=epoch)
        self.logger.add_scalar("train/sloss", avg_sloss, global_step=epoch)

    def eval_model(self, data_loader: DataLoader) -> tuple[Float, Float]:
        accs, losses, slosses = [], [], []

        for batch in data_loader:
            loss, acc, sloss, self.rng = self.eval_step(self.state, self.rng, batch)
            losses.append(loss)
            accs.append(acc)
            slosses.append(sloss)

        avg_loss = np.stack(jax.device_get(losses)).mean()
        avg_acc = np.stack(jax.device_get(accs)).mean()
        avg_sloss = np.stack(jax.device_get(slosses)).mean()

        return avg_acc, avg_loss, avg_sloss
