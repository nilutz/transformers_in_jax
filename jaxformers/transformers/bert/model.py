from pathlib import Path

import orbax.checkpoint as checkpoint
from flax import linen as nn
from jaxtyping import Array, Bool, Float, Int32

import jax.numpy as jnp
from jaxformers.modules.encoder_block import EncoderBlock
from jaxformers.modules.positional_encoding import PositionalEncoding


class Bert(nn.Module):
    """
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
    """

    # Src Embedding
    src_vocab_size: int
    model_dim: int  # model_dim

    # Encoder
    enc_num_layers: int

    # EncoderBlock attributes
    num_heads: int
    dropout_prob: int

    def setup(self):
        self.word_emb_src = nn.Embed(self.src_vocab_size, self.model_dim)
        self.positional_embedding = PositionalEncoding(self.model_dim)
        self.segment_embedding = nn.Embed(
            2, self.model_dim
        )  # TODO What is the correct value here =? 2
        self.dropout_emb = nn.Dropout(self.dropout_prob)

        self.encoder_blocks = [
            EncoderBlock(
                input_dim=self.model_dim,
                num_heads=self.num_heads,
                dim_feedforward=2 * self.model_dim,
                dropout_prob=self.dropout_prob,
            )
            for _ in range(0, self.enc_num_layers)
        ]

    def embeddings(
        self, input: Int32[Array, "batch_size seq_len"], segment_label
    ) -> Float[Array, "batch seq_len model_dim"]:
        word_embedding = self.word_emb_src(input)
        pos_enc = self.positional_embedding(input)
        if segment_label is not None:
            segments = self.segment_embedding(segment_label)
            return word_embedding + pos_enc + segments
        return word_embedding + pos_enc

    def __call__(
        self,
        src: Int32[Array, "batch_size seq_len"],
        src_mask: Bool[Array, "batch_size 1 1 seq_len"] = None,
        segment_label: Int32[Array, "batch_size seq_len"] = None,
        train: bool = True,
    ) -> Float[Array, "batch seq_len model_dim"]:
        x = self.embeddings(src, segment_label)
        # TODO layernorm
        x = self.dropout_emb(x, deterministic=not train)

        for encoder in self.encoder_blocks:
            x = encoder(x, src_mask, train=train)

        return x


class BertNextSentencePrediction(nn.Module):
    """
    Bert for Next Sentence Prediction Head: Predict whether sentences are next or not
    """

    model_dim: int

    def setup(self):
        self.pooler = nn.Dense(self.model_dim)
        self.linear = nn.Dense(2)

    def __call__(
        self, x: Float[Array, "batch seq_len model_dim"]
    ) -> Int32[Array, "batch_size 2"]:
        # use only the first token which is the [CLS]

        x = self.pooler(x[:, 0])
        x = jnp.tanh(x)
        x = self.linear(x)
        return x


class BertMaskedLanguageModel(nn.Module):
    """
    Bert for Language Modelling Head: Predicting MASKed sequences
    """

    model_dim: int
    vocab_size: int

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def __call__(
        self, x: Float[Array, "batch seq_len model_dim"]
    ) -> Float[Array, "batch_size seq_len vocab_size"]:
        return self.linear(x)


class BertPretrainHead(nn.Module):
    """
    Wrapper around `Bert` and `BertNextSentencePrediction`, `BertMaskedLanguageModel` for pre-training
    """

    # Src Embedding
    src_vocab_size: int
    model_dim: int

    # Encoder
    enc_num_layers: int

    # EncoderBlock attributes
    num_heads: int
    dropout_prob: int

    def setup(self):
        self.bert = Bert(
            src_vocab_size=self.src_vocab_size,
            model_dim=self.model_dim,
            enc_num_layers=self.enc_num_layers,
            num_heads=self.num_heads,
            dropout_prob=self.dropout_prob,
        )
        self.next_sentence_prediction = BertNextSentencePrediction(
            model_dim=self.model_dim
        )
        self.masked_lm = BertMaskedLanguageModel(
            model_dim=self.model_dim, vocab_size=self.src_vocab_size
        )

    def __call__(
        self,
        x: Int32[Array, "batch_size seq_len"],
        src_mask: Bool[Array, "batch_size 1 1 seq_len"],
        segment_label: Int32[Array, "batch_size seq_len"] = None,
        train: bool = True,
    ) -> tuple[
        Int32[Array, "batch_size 2"], Float[Array, "batch_size model_dim vocab_size"]
    ]:
        x = self.bert(x, src_mask, segment_label, train=train)
        return self.next_sentence_prediction(x), self.masked_lm(x)

    def _init_with_weights(self, path: Path):
        ckptr = checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler())
        params = ckptr.restore(path, item=None)
        self.params = params["state"]["params"]


if __name__ == "__main__":
    from jax import random

    main_rng = random.PRNGKey(42)

    src_vocab_size = 300
    model_dim = 512
    max_length = 12
    batch_size = 2
    enc_num_layers = 2
    num_heads = 4
    dropout_prob = 0.1

    example_input_src = random.randint(
        main_rng, (batch_size, max_length), minval=0, maxval=src_vocab_size
    )
    example_input_src_mask = random.randint(
        main_rng, (batch_size, 1, 1, max_length), minval=0, maxval=1
    )
    example_input_segment_mask = random.randint(
        main_rng, (batch_size, max_length), minval=0, maxval=1
    )

    bert = BertPretrainHead(
        src_vocab_size=src_vocab_size,
        model_dim=model_dim,
        enc_num_layers=enc_num_layers,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
    )

    init_rng, dropout_init_rng = random.split(main_rng, 2)

    params = bert.init(
        {"params": init_rng, "dropout": dropout_init_rng},
        example_input_src,
        example_input_src_mask,
        example_input_segment_mask,
    )["params"]

    main_rng, dropout_apply_rng = random.split(main_rng)

    nsp_out, masked_lm = bert.apply(
        {"params": params},
        example_input_src,
        example_input_src_mask,
        example_input_segment_mask,
        rngs={"dropout": dropout_apply_rng},
    )

    print("nsp_out", nsp_out.shape)
    print("masked_lm", masked_lm.shape)
