# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
from jax import random
from jaxformers.transformers import VanillaTransformer
from jaxformers.transformers.vanilla.config import Config

from jaxformers.transformers.vanilla.preprocessing_wmt14 import make_loaders
from jaxformers.transformers.vanilla.trainer import VanillaTrainerModule

main_rng = random.PRNGKey(42)


def train():
    print("Devices:", jax.devices())
    print("JAX processes", jax.process_index(), jax.process_count())
    config = Config()

    (
        train_dataloader,
        test_dataloader,
        validation_dataloader,
        pad_id,
        len_vocab,
    ) = make_loaders(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        max_length=config.max_length,
        vocab_size=config.vocab_size,
    )

    print("pad_id", pad_id)
    config.vocab = len_vocab

    model = VanillaTransformer(
        src_vocab_size=len_vocab,
        model_dim=config.model_dim,
        enc_num_layers=config.encoder_num_layers,
        dec_num_layers=config.decoder_num_layers,
        num_heads=config.heads,
        dropout_prob=config.dropout,
        tgt_vocab_size=len_vocab,
    )

    exmp_batch = list(test_dataloader)[0]
    print("Example Batch on device:", exmp_batch[0].device_buffer.device())

    max_epochs = 5

    num_train_iters = len_vocab * max_epochs

    trainer = VanillaTrainerModule(
        model=model,
        model_name="vanilla__wmt14_train_0",
        init_batch=exmp_batch,
        max_iters=num_train_iters,
        padding_index=pad_id,
        config=config,
    )

    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model(
            train_dataloader, validation_dataloader, num_epochs=max_epochs
        )
        trainer.load_model()
    else:
        trainer.load_model()
    val_acc = trainer.eval_model(validation_dataloader)
    test_acc = trainer.eval_model(test_dataloader)

    # Bind parameters to model for easier inference
    trainer.model_bd = trainer.model.bind({"params": trainer.state.params})

    print("val_acc", val_acc)
    print("test_acc", test_acc)


if __name__ == "__main__":
    train()
