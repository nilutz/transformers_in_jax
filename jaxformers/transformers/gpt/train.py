# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import time

import jax
from jax import random
from jaxformers.transformers import GPT
from jaxformers.transformers.gpt.config import Config

# from jaxformers.transformers.vanilla.preprocessing_id_tokens import make_loaders
from jaxformers.transformers.gpt.preprocessing_shakespeare import get_batch
from jaxformers.transformers.gpt.trainer import GPTTrainerModule

main_rng = random.PRNGKey(42)


def train():
    # jax.config.update('jax_platform_name', 'cpu')
    print("Devices:", jax.devices())
    print("JAX processes", jax.process_index(), jax.process_count())
    config = Config()

    model = GPT(
        src_vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        dec_num_layers=config.decoder_num_layers,
        num_heads=config.heads,
        dropout_prob=config.dropout,
        seq_len=config.max_length,
    )

    X, Y = get_batch(split="validation")

    trainer = GPTTrainerModule(
        model=model,
        model_name="gpt_train_14",
        init_batch=(X, Y),
        config=config,
    )

    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        start_data_time = time.time()
        print(start_data_time)
        trainer.train_model(
            get_batch,
        )
        end_data_time = time.time()
        print(end_data_time)
        print(end_data_time - start_data_time)

        trainer.load_model()
    else:
        trainer.load_model()
    val_loss, val_acc = trainer.eval_model(get_batch)

    # Bind parameters to model for easier inference
    trainer.model_bd = trainer.model.bind({"params": trainer.state.params})

    print("val_loss", val_loss)
    print("val_acc", val_acc)


if __name__ == "__main__":
    train()
