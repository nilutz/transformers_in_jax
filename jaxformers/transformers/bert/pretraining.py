import time

import jax
from jax import random
from jaxformers.transformers import BertPretrainHead
from jaxformers.transformers.bert.preprocessing_imdb import make_loaders
from jaxformers.transformers.bert.trainer import BertTrainerModule
from jaxformers.transformers.vanilla.config import Config

main_rng = random.PRNGKey(42)


def pretrain():
    # jax.config.update('jax_platform_name', 'cpu')
    print("Devices:", jax.devices())
    print("JAX processes", jax.process_index(), jax.process_count())
    config = Config()

    (
        train_dataloader,
        test_dataloader,
        validation_dataloader,
        pad_id,
    ) = make_loaders(
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    model = BertPretrainHead(
        src_vocab_size=config.vocab_size,
        model_dim=config.model_dim,
        enc_num_layers=config.encoder_num_layers,
        num_heads=config.heads,
        dropout_prob=config.dropout,
    )

    exmp_batch = list(test_dataloader)[0]
    print("Example Batch on device:", exmp_batch[0].device_buffer.device())

    max_epochs = 5

    num_train_iters = config.vocab_size * max_epochs

    trainer = BertTrainerModule(
        model=model,
        model_name="bert_pretrain_11",
        init_batch=exmp_batch,
        max_iters=num_train_iters,
        config=config,
        padding_index=pad_id,
    )

    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        start_data_time = time.time()
        print(start_data_time)
        trainer.train_model(
            train_dataloader, validation_dataloader, num_epochs=max_epochs
        )
        end_data_time = time.time()
        print(end_data_time)
        print(end_data_time - start_data_time)

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
    pretrain()
