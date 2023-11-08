from pathlib import Path

import flax.linen as nn
import numpy as np
from flax.training.common_utils import onehot

import jax
import jax.numpy as jnp
from jax import random
from jaxformers.trainer import CrossEntropyLoss
from jaxformers.transformers import BertPretrainHead
from jaxformers.transformers.bert.config import Config
from jaxformers.transformers.bert.preprocessing_imdb import make_loaders


def nsp_loss(logits, labels):
    loss = jax.nn.log_softmax(logits)
    loss = -jnp.mean(jnp.sum(loss * labels, axis=-1))
    return loss


def inference_from_pretrained():
    config = Config()

    model = BertPretrainHead(
        src_vocab_size=config.vocab_size,
        model_dim=config.model_dim,
        enc_num_layers=config.encoder_num_layers,
        num_heads=config.heads,
        dropout_prob=config.dropout,
    )

    # weights
    CHECKPOINT_PATH = (
        Path(__file__).parent.parent.parent.parent / "runs/bert_pretrain_9/1/default"
    )
    model._init_with_weights(CHECKPOINT_PATH)
    params = model.params

    main_rng = random.PRNGKey(42)

    main_rng, dropout_apply_rng = random.split(main_rng)

    apply = jax.jit(
        lambda data: model.apply(
            {"params": params},
            data[0],
            data[1],
            data[2],
            rngs={"dropout": dropout_apply_rng},
            train=False,
        )
    )

    _, test_loader, _, pad_id = make_loaders(
        batch_size=1,
        vocab_size=config.vocab_size,
        max_length=config.max_length,
        debug=True,
    )

    print("pad", pad_id)

    for i_num, (
        input_ids,  # src
        src_mask,  # attenion mask
        type_ids,  # segment_mask
        original_input_ids,  # lm_labels
        nsp_label,  # next_sentence_labels
        original_input_tokens,
        tokens,
    ) in enumerate(test_loader):
        print("sentence", tokens)
        print("nsp_label", nsp_label)

        nsp_logits, logits_mlm = apply((input_ids, src_mask, type_ids))

        print("logits_mlm.shape", logits_mlm.shape)
        print("logits_mlm.shape", logits_mlm)

        pred_token = logits_mlm.argmax(-1)
        print(input_ids)
        print(original_input_ids)
        print(pred_token)

        # TODO HOW AND WHY this fuck does this not learn

        # * mask all others then not MASK ?
        # https://github.com/Meelfy/pytorch_pretrained_BERT/blob/1a95f9e3e5ace781623b2a0eb20d758765e61145/examples/run_lm_finetuning.py#L270
        # has an lm_mask where -1 is all and has ids and then crossentryopy with ignore -1

        # decoded = tokenizer.decode(tgt_index, skip_special_tokens=False)

        if i_num == 1:
            break


if __name__ == "__main__":
    inference_from_pretrained()
