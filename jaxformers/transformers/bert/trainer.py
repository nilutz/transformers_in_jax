from pathlib import Path

import flax.linen as nn
import numpy as np
import optax
from flax.training import train_state
from flax.training.common_utils import onehot
from flax.training.train_state import TrainState
from jaxtyping import Array, Bool, Float, Int32, PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import random
from jaxformers.trainer import DefaultTrainer, create_custom_optimizer

CHECKPOINT_PATH = Path(__file__).parent.parent.parent.parent / "runs"

if not CHECKPOINT_PATH.is_dir():
    Path(CHECKPOINT_PATH).mkdir()


BatchType = tuple[
    Int32[Array, "batch seq_len"],  # input_ids/src (with MASK)
    Bool[Array, "batch 1 1 seq_len"],  # attention mask/src_mask
    Int32[Array, "batch seq_len"],  # segment ids
    Int32[Array, "batch seq_len"],  # ground truth src / ids without MASK
    Int32[Array, "batch"],  # nsp
]


class BertTrainerModule(DefaultTrainer):
    def __init__(
        self,
        model,
        model_name: str,
        config,
        init_batch: tuple,
        max_iters: int,
        padding_index: int,
    ):
        super().__init__(
            model=model, model_name=model_name, config=config, init_batch=init_batch
        )

        self.max_iters: int = max_iters
        self.padding_index = padding_index

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(
            params: dict,
            rng: PRNGKeyArray,
            batch: BatchType,
            train: bool,
        ) -> tuple[Float, tuple[Float, Float, Float, PRNGKeyArray]]:
            # data
            (
                src,
                src_mask,
                segment_mask,
                lm_labels,
                next_sentence_labels,
            ) = batch

            # dropout_apply_rng = jax.random.fold_in(dropout_apply_rng, self.optimizer.state.step)
            # rng = jax.random.fold_in(rng, self.optimizer.state.step)
            rng, dropout_apply_rng = random.split(rng)

            # Apply to network
            next_sentence_logits, lm_logits = self.model.apply(
                {"params": params},
                src,
                src_mask,
                segment_mask,
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )

            # MLM uses negative log likelihood
            # TODO could reuse label smoothing with no smoothing and rename to some cross entropy loss
            soft_labels = onehot(lm_labels, lm_logits.shape[-1])
            loss = -jnp.sum(soft_labels * jax.nn.log_softmax(lm_logits), axis=-1)
            weights = jnp.where(lm_labels != self.padding_index, 1, 0).astype(
                jnp.float32
            )
            # TODO the weights is padding 1 all other that are not masked
            mask_id = 4
            # weights = jnp.where(src == mask_id, 1, 0).astype(
            #     jnp.float32
            # ) #or  ??? and what about acc ? and nsp ?
            # TODO fix fuckgin losses and accs
            loss = loss * weights
            masked_lm_loss = loss.sum() / weights.sum()

            preds = jnp.argmax(lm_logits, axis=-1)  # TODO before throug log softmax ?
            acc = jnp.equal(preds, lm_labels)
            if weights is not None:
                acc = acc * weights
            acc_mlm = acc.sum()

            # jax.debug.print("acc_mlm {x}", x=acc_mlm)
            # jax.debug.print("masked_lm_loss {x}", x=masked_lm_loss)

            # NSP uses signmoid binary cross entropy loss
            nsp_soft_labels = onehot(
                next_sentence_labels, next_sentence_logits.shape[-1]
            )
            next_sentence_loss = -jnp.mean(
                jnp.sum(
                    nsp_soft_labels * nn.log_softmax(next_sentence_logits, axis=-1),
                    axis=-1,
                )
            )

            next_sentence_labels = next_sentence_labels.reshape((-1,))
            acc_nsp = jnp.sum(
                jnp.argmax(next_sentence_logits, axis=-1) == next_sentence_labels
            )  # TODO before throug log softmax ?

            loss = masked_lm_loss + next_sentence_loss
            return loss, (acc_mlm, masked_lm_loss, acc_nsp, next_sentence_loss, rng)

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
            loss, acc_mlm, masked_lm_loss, acc_nsp, next_sentence_loss, rng = (
                ret[0],
                *ret[1],
            )
            state = state.apply_gradients(grads=grads)
            return (
                state,
                rng,
                loss,
                acc_mlm,
                masked_lm_loss,
                acc_nsp,
                next_sentence_loss,
            )

        self.train_step = jax.jit(train_step)

        # Evaluation function
        def eval_step(state, rng, batch):
            loss, (
                acc_mlm,
                masked_lm_loss,
                acc_nsp,
                next_sentence_loss,
                rng,
            ) = calculate_loss(state.params, rng, batch, train=False)
            return loss, acc_mlm, masked_lm_loss, acc_nsp, next_sentence_loss, rng

        self.eval_step = jax.jit(eval_step)

    def init_model(
        self,
        init_batch: BatchType,
    ):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)
        src, src_mask, segment_mask, _, _ = init_batch

        params = self.model.init(
            {
                "params": init_rng,
                "dropout": dropout_init_rng,
            },
            src,
            src_mask,
            segment_mask,
            train=True,
        )[
            "params"
        ]  # Apply transform

        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.lr,
            warmup_steps=self.config.warmup,
            decay_steps=self.max_iters,
            end_value=0.0,
        )
        optimizer = create_custom_optimizer(
            learning_rate=self.config.lr,
            weight_decay_rate=self.config.weight_decay_rate,
            beta_1=self.config.beta_1,
            beta_2=self.config.beta_2,
            epsilon=self.config.epsilon,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            # optax.adamw(lr_schedule),
            optimizer,
        )
        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )
        print("... model initialized")

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
                (
                    avg_loss,
                    avg_acc_mlm,
                    avg_masked_lm_loss,
                    avg_acc_nsp,
                    avg_next_sentence_loss,
                ) = self.eval_model(val_loader)
                self.logger.add_scalar("val/loss", avg_loss, global_step=epoch_idx)
                self.logger.add_scalar(
                    "val/acc_mlm", avg_acc_mlm, global_step=epoch_idx
                )
                self.logger.add_scalar(
                    "val/masked_lm_loss", avg_masked_lm_loss, global_step=epoch_idx
                )
                self.logger.add_scalar(
                    "val/acc_nsp", avg_acc_nsp, global_step=epoch_idx
                )
                self.logger.add_scalar(
                    "val/next_sentence_loss",
                    avg_next_sentence_loss,
                    global_step=epoch_idx,
                )

                if avg_acc_mlm + avg_acc_nsp >= best_acc:
                    best_acc = avg_acc_mlm + avg_acc_nsp
                    self.save_model(step=epoch_idx)
                self.logger.flush()
        self.save_model(step=epoch_idx + 1)

    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int):
        acc_mlms, losses, masked_lm_losses, acc_nsps, next_sentence_losses = (
            [],
            [],
            [],
            [],
            [],
        )
        for i, batch in enumerate(
            tqdm(
                train_loader,
                unit="batch",
                desc=f"Epoch: {epoch}/{total_epochs} ",
                bar_format="{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}",
                ascii=" #",
            )
        ):
            (
                self.state,
                self.rng,
                loss,
                acc_mlm,
                masked_lm_loss,
                acc_nsp,
                next_sentence_loss,
            ) = self.train_step(self.state, self.rng, batch)
            losses.append(loss)
            acc_mlms.append(acc_mlm)
            masked_lm_losses.append(masked_lm_loss)
            acc_nsps.append(acc_nsp)
            next_sentence_losses.append(next_sentence_loss)

            self.logger.add_scalar(f"batch/{epoch}/loss", np.array(loss), global_step=i)
            self.logger.add_scalar(
                f"batch/{epoch}/masked_lm_loss", np.array(masked_lm_loss), global_step=i
            )
            self.logger.add_scalar(
                f"batch/{epoch}/next_sentence_loss",
                np.array(next_sentence_loss),
                global_step=i,
            )

            self.logger.add_scalar(
                f"batch/{epoch}/acc_mlm", np.array(acc_mlm), global_step=i
            )
            self.logger.add_scalar(
                f"batch/{epoch}/acc_nsp", np.array(acc_nsp), global_step=i
            )
            self.logger.flush()

            # if i % 10 == 0:
            #     self.save_model(step=i)

        avg_loss = np.stack(jax.device_get(losses)).mean()
        avg_acc_mlm = np.stack(jax.device_get(acc_mlms)).mean()
        avg_masked_lm_loss = np.stack(jax.device_get(masked_lm_losses)).mean()
        avg_acc_nsp = np.stack(jax.device_get(acc_nsps)).mean()
        avg_next_sentence_loss = np.stack(jax.device_get(next_sentence_losses)).mean()

        self.logger.add_scalar("train/loss", avg_loss, global_step=epoch)
        self.logger.add_scalar("train/acc_mlms", avg_acc_mlm, global_step=epoch)
        self.logger.add_scalar(
            "train/masked_lm_loss", avg_masked_lm_loss, global_step=epoch
        )
        self.logger.add_scalar("train/acc_nsp", avg_acc_nsp, global_step=epoch)
        self.logger.add_scalar(
            "train/next_sentence_loss", avg_next_sentence_loss, global_step=epoch
        )

    def eval_model(self, data_loader: DataLoader) -> tuple[Float, Float]:
        acc_mlms, losses, masked_lm_losses, acc_nsps, next_sentence_losses = (
            [],
            [],
            [],
            [],
            [],
        )

        for batch in data_loader:
            (
                loss,
                acc_mlm,
                masked_lm_loss,
                acc_nsp,
                next_sentence_loss,
                self.rng,
            ) = self.eval_step(self.state, self.rng, batch)
            losses.append(loss)
            acc_mlms.append(acc_mlm)
            masked_lm_losses.append(masked_lm_loss)
            acc_nsps.append(acc_nsp)
            next_sentence_losses.append(next_sentence_loss)

        avg_loss = np.stack(jax.device_get(losses)).mean()
        avg_acc_mlm = np.stack(jax.device_get(acc_mlms)).mean()
        avg_masked_lm_loss = np.stack(jax.device_get(masked_lm_losses)).mean()
        avg_acc_nsp = np.stack(jax.device_get(acc_nsps)).mean()
        avg_next_sentence_loss = np.stack(jax.device_get(next_sentence_losses)).mean()

        return (
            avg_loss,
            avg_acc_mlm,
            avg_masked_lm_loss,
            avg_acc_nsp,
            avg_next_sentence_loss,
        )
