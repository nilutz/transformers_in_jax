from pathlib import Path

import orbax.checkpoint as checkpoint
from flax import linen as nn
from jaxtyping import Array, Bool, Float, Int32

import jax
import jax.numpy as jnp
from jaxformers.modules.decoder_block import DecoderBlock
from jaxformers.modules.encoder_block import EncoderBlock
from jaxformers.modules.generator import Generator
from jaxformers.modules.positional_encoding import PositionalEncoding


class VanillaEncoder(nn.Module):
    """
    The encoder is composed of a stack of N = 6 identical layers.
    ![VanillaEncoder](../../src/vanilla_transformer/encoder.png "VanillaEncoder")

    This module has N `jaxformers.modules.EncoderBlock` blocks stacked!
    """

    num_layers: int

    # EncoderBlock attributes
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: int

    def setup(self):
        self.layers = [
            EncoderBlock(
                input_dim=self.input_dim,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout_prob=self.dropout_prob,
            )
            for _ in range(0, self.num_layers)
        ]

    def __call__(
        self,
        x: Float[Array, "batch seq_len model_dim"],
        mask: Bool[Array, "batch_size seq_len"] = None,
        train: bool = True,
    ) -> Float[Array, "batch seq_len model_dim"]:
        for l in self.layers:
            x = l(x, mask=mask, train=train)
        return x


class VanillaDecoder(nn.Module):
    """
    The encoder is composed of a stack of N = 6 identical layers.
    ![VanillaDecoder](../../src/vanilla_transformer/decoder.png "VanillaDecoder")

    This module has N `jaxformers.modules.DecoderBlock` blocks stacked!
    """

    num_layers: int

    # EncoderBlock attributes
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: int

    def setup(self):
        self.layers = [
            DecoderBlock(
                input_dim=self.input_dim,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout_prob=self.dropout_prob,
            )
            for _ in range(0, self.num_layers)
        ]

    def __call__(
        self,
        encoder_out: Float[Array, "batch_size seq_len model_dim"],
        tgt: Float[Array, "batch seq_len-1 model_dim"],
        src_mask: Bool[Array, "batch_size 1 seq_len"] = None,
        tgt_mask: Bool[Array, "batch_size seq_len-1 seq_len-1"] = None,
        train: bool = True,
    ) -> Float[Array, "batch_size seq_len-1 model_dim "]:
        for l in self.layers:
            tgt = l(encoder_out, tgt, src_mask=src_mask, tgt_mask=tgt_mask, train=train)
        return tgt


class VanillaTransformer(nn.Module):
    """
    [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    # Src Embedding
    src_vocab_size: int
    model_dim: int  # model_dim

    # Encoder
    enc_num_layers: int

    # EncoderBlock attributes
    num_heads: int
    dropout_prob: int

    # Tgt Embedding
    tgt_vocab_size: int

    # Decoder
    dec_num_layers: int

    def setup(self) -> None:
        self.word_emb_src = nn.Embed(self.src_vocab_size, self.model_dim)
        self.positional_embedding = PositionalEncoding(self.model_dim)
        self.dropout_src = nn.Dropout(self.dropout_prob)

        self.encoder = VanillaEncoder(
            num_layers=self.enc_num_layers,
            input_dim=self.model_dim,
            num_heads=self.num_heads,
            dim_feedforward=4 * self.model_dim,
            dropout_prob=self.dropout_prob,
        )

        self.word_emb_tgt = nn.Embed(self.tgt_vocab_size, self.model_dim)

        self.decoder = VanillaDecoder(
            num_layers=self.dec_num_layers,
            input_dim=self.model_dim,
            num_heads=self.num_heads,
            dim_feedforward=4 * self.model_dim,
            dropout_prob=self.dropout_prob,
        )

        self.generator = Generator(self.tgt_vocab_size)

    def src_embed(
        self, src: Int32[Array, "batch_size seq_len"], train=True
    ) -> Float[Array, "batch seq_len model_dim"]:
        """
        1.: src -> Input Embedding + Positional Encoding
        """
        x = self.word_emb_src(src)

        word_embedding = self.word_emb_src(src) * jnp.sqrt(
            self.model_dim
        )  # batch_size, seq_len model_dim
        pos_enc = self.positional_embedding(src)
        x = word_embedding + pos_enc

        return x

    def tgt_embed(
        self, tgt: Int32[Array, "batch_size seq_len"]
    ) -> Float[Array, "batch seq_len model_dim"]:
        """
        3. Embed: Bottom up decoder tgt embedding:
        tgt -> Input Embedding + Positional Encoding
        """
        word_embedding = self.word_emb_tgt(tgt) * jnp.sqrt(
            self.model_dim
        )  # batch_size, seq_len model_dim
        pos_enc = self.positional_embedding(tgt)
        x = word_embedding + pos_enc

        return x

    def encode(
        self,
        src: Int32[Array, "batch_size seq_len"],
        src_mask: Bool[Array, "batch_size 1 1 seq_len"] = None,
        train: bool = True,
    ) -> Float[Array, "batch_size seq_len model_dim"]:
        """
        1. Encoder: Left hand side
        """

        # 1. Embed
        input = self.src_embed(src)
        input = self.dropout_src(input, deterministic=not train)

        # 2. Run through encoder
        encoder_out = self.encoder(input, src_mask, train=train)

        return encoder_out

    def decode(
        self,
        encoder_out: Float[Array, "batch_size seq_len model_dim"],
        tgt: Int32[Array, "batch_size seq_len-1"],
        src_mask: Bool[Array, "batch_size 1 1 seq_len"],
        tgt_mask: Bool[Array, "batch_size 1 seq_len-1 seq_len-1"],
        train: bool,
    ) -> Float[Array, "batch_size seq_len-1 model_dim"]:
        # 3. Embed: Bottom up decoder tgt embedding

        tgt_embed = self.tgt_embed(tgt)

        # 4. Run through decoder
        decoder_out = self.decoder(
            encoder_out, tgt_embed, src_mask, tgt_mask, train=train
        )
        return decoder_out

    def generate(
        self, decoder_out: Float[Array, "batch_size seq_len-1 model_dim"]
    ) -> Float[Array, "batch_size seq_len-1 tgt_vocab_size"]:
        return self.generator(decoder_out)
        # return self.fc(decoder_out)

    def _init_with_weights(self, path: Path):
        ckptr = checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler())
        params = ckptr.restore(path, item=None)
        self.params = params["state"]["params"]

    def __call__(
        self,
        src: Int32[Array, "batch_size seq_len"],
        tgt: Int32[Array, "batch_size seq_len-1"],  # shifted
        src_mask: Bool[Array, "batch_size 1 1 seq_len"] = None,
        tgt_mask: Bool[Array, "batch_size 1 seq_len-1 seq_len-1"] = None,  # shifted
        train: bool = True,
    ) -> tuple[
        Float[Array, "batch_size seq_len-1 model_dim "],
        Float[Array, "batch_size seq_len-1 tgt_vocab_size"],
    ]:
        encoder_out = self.encode(src, src_mask=src_mask, train=train)

        decoder_out = self.decode(encoder_out, tgt, src_mask, tgt_mask, train=train)

        out = self.generate(decoder_out)

        return decoder_out, out


if __name__ == "__main__":
    from jax import random

    main_rng = random.PRNGKey(42)

    src_vocab_size = 300
    tgt_vocab_size = 300
    model_dim = 512
    max_length = 12
    batch_size = 2
    enc_num_layers = 2
    dec_num_layers = 3
    num_heads = 4
    dropout_prob = 0.1

    example_input_src = random.randint(
        main_rng, (batch_size, max_length), minval=0, maxval=src_vocab_size
    )

    example_input_trg = random.randint(
        main_rng, (batch_size, max_length), minval=0, maxval=tgt_vocab_size
    )

    example_input_src_mask = random.randint(
        main_rng, (batch_size, 1, 1, max_length), minval=0, maxval=1
    )
    example_input_tgt_mask = random.randint(
        main_rng, (batch_size, 1, max_length, max_length), minval=0, maxval=1
    )
    # Create Transformer encoder decoder
    encdec = VanillaTransformer(
        src_vocab_size=src_vocab_size,
        model_dim=model_dim,
        enc_num_layers=enc_num_layers,
        dec_num_layers=dec_num_layers,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        tgt_vocab_size=tgt_vocab_size,
    )

    # Initialize
    init_rng, dropout_init_rng, embedding_init_rng = random.split(main_rng, 3)
    params = encdec.init(
        {
            "params": init_rng,
            "dropout": dropout_init_rng,
        },
        example_input_src,
        example_input_trg[:, :-1],
        example_input_src_mask,
        example_input_tgt_mask[:, :, :-1, :-1],
    )["params"]

    # Since dropout is stochastic, we need to pass a rng to the forward
    main_rng, dropout_apply_rng = random.split(main_rng)

    # Instead of passing params and rngs every time to a function call, we can bind them to the module
    # binded_mod = encdec.bind({"params": params}, rngs={"dropout": dropout_apply_rng},)
    print("src", example_input_src.shape)
    print("tgt", example_input_trg.shape)

    decoder_out, out = encdec.apply(
        {"params": params},
        example_input_src,
        example_input_trg[:, :-1],
        example_input_src_mask,
        example_input_tgt_mask[:, :, :-1, :-1],
        rngs={"dropout": dropout_apply_rng},
    )

    print("Out", decoder_out.shape)
    print("Out", decoder_out)

    # tabulate_fn = nn.tabulate(encdec, {"params": init_rng, "dropout": dropout_init_rng})
    # print(
    #     tabulate_fn(
    #         example_input_src,
    #         example_input_trg[:, :-1],
    #         example_input_src_mask,
    #         example_input_tgt_mask[:, :, :-1, :-1],
    #     )
    # )
