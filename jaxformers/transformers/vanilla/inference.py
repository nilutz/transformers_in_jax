from pathlib import Path

from datasets import load_dataset

import jax
import jax.numpy as jnp
from jax import random
from jaxformers.transformers import VanillaTransformer
from jaxformers.transformers.vanilla.config import Config
from jaxformers.transformers.vanilla.preprocessing_wmt14 import (
    create_tgt_mask,
    make_tokenizer,
)


def inference():
    config = Config()

    dataset = load_dataset(config.dataset_name)
    tokenizer = make_tokenizer(config.dataset_name, max_length=config.max_length)

    len_vocab = tokenizer.get_vocab_size()

    # model
    model = VanillaTransformer(
        src_vocab_size=len_vocab,
        model_dim=config.model_dim,
        enc_num_layers=config.encoder_num_layers,
        dec_num_layers=config.decoder_num_layers,
        num_heads=config.heads,
        dropout_prob=config.dropout,
        tgt_vocab_size=len_vocab,
    )

    # weights
    CHECKPOINT_PATH = (
        Path(__file__).parent.parent.parent.parent / "runs/vanilla_train_15/5/default"
    )
    model._init_with_weights(CHECKPOINT_PATH)
    params = model.params

    main_rng = random.PRNGKey(42)

    apply_encode = jax.jit(
        lambda data: model.apply(
            {"params": params},
            data[0],
            data[1],
            rngs={"dropout": dropout_apply_rng},
            train=False,
            method=model.encode,
        )
    )

    apply_decode = jax.jit(
        lambda data: model.apply(
            {"params": params},
            data[0],  # encoder
            data[1],  # tgt
            data[2],  # src_mask
            data[3],  # tgt_maks
            rngs={"dropout": dropout_apply_rng},
            train=False,
            method=model.decode,
        )
    )

    apply_generate = jax.jit(
        lambda data: model.apply(
            {"params": params},
            data,  # decoder_out
            rngs={"dropout": dropout_apply_rng},
            method=model.generate,
        )
    )

    bos_id = 2
    eos_id = 3
    pad_id = tokenizer.padding["pad_id"]
    print(bos_id, eos_id, pad_id)

    for i_num, i in enumerate(dataset["test"]):
        encoding = tokenizer.encode(i["de"])
        src = jnp.expand_dims(
            jnp.array(encoding.ids, dtype=jnp.int32), axis=0
        )  # batch_size seq_len
        src_mask = jnp.expand_dims(
            jnp.array(encoding.attention_mask).astype(bool), axis=(0, 1, 2)
        )  # batch_size 1 seq_len seq_len = 1 1 1 seq_len

        main_rng, dropout_apply_rng = random.split(main_rng)
        encoded = apply_encode((src, src_mask))

        tgt_index = [bos_id]
        for j in range(config.max_length):
            # print("step", j, "/", max_length)
            tgt_tensor = jnp.expand_dims(
                jnp.array(tgt_index), axis=0
            )  # batch_size seq_len
            tgt_mask = create_tgt_mask(
                tgt_tensor, padding_idx=pad_id
            )  # batch_size 1 seq_len seq_len = 1 1 seq_len seq_len

            decoder_out = apply_decode((encoded, tgt_tensor, src_mask, tgt_mask))
            gen = apply_generate(decoder_out)

            # TODO insert beam search here...
            pred_token = jnp.argmax(gen, axis=2)[:, -1]
            pred_token = jax.device_get(pred_token[0].item())

            tgt_index.append(pred_token)
            if pred_token == 3:
                break

        decoded = tokenizer.decode(tgt_index, skip_special_tokens=False)
        print(i)
        print(decoded)
        print("=" * 20)
        # calc bleu
        if i_num > 10:
            break


if __name__ == "__main__":
    inference(dataset_name="bentrevett/multi30k")
