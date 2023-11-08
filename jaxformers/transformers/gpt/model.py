from flax import linen as nn
from jaxtyping import Array, Bool, Float, Int32

# from jaxformers.modules.positional_encoding import PositionalEncoding
import jax
import jax.numpy as jnp
from jaxformers.modules.gpt_decoder_block import GPTDecoderBlock


class GPTBlock(nn.Module):
    num_layers: int

    # GPTDecoderBlock attributes
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: int

    def setup(self):
        self.layers = [
            GPTDecoderBlock(
                input_dim=self.input_dim,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout_prob=self.dropout_prob,
            )
            for _ in range(0, self.num_layers)
        ]

    def __call__(
        self,
        x: Float[Array, "batch_size seq_len model_dim"],
        x_mask: Bool[Array, "batch_size 1 seq_len"] = None,
        train: bool = True,
    ) -> Float[Array, "batch_size seq_len model_dim "]:
        for l in self.layers:
            y = l(x, x_mask, train=train)
        return y


class GPT(nn.Module):
    src_vocab_size: int  # =tgt_vocab_size
    n_embd: int
    seq_len: int

    dec_num_layers: int

    num_heads: int
    dropout_prob: int

    def setup(self):
        self.word_emb_src = nn.Embed(
            self.src_vocab_size,
            self.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        self.positional_embedding = nn.Embed(
            self.seq_len,
            self.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )

        self.dropout_emb = nn.Dropout(self.dropout_prob)

        self.block = GPTBlock(
            num_layers=self.dec_num_layers,
            input_dim=self.n_embd,
            num_heads=self.num_heads,
            dim_feedforward=4 * self.n_embd,
            dropout_prob=self.dropout_prob,
        )

        self.lm_out = nn.Dense(self.src_vocab_size, use_bias=False)

    def embeddings(
        self, x: Int32[Array, "batch_size seq_len"]
    ) -> Float[Array, "batch seq_len model_dim"]:
        word_embedding = self.word_emb_src(x)

        pos = jnp.arange(0, x.shape[-1], dtype=jnp.int32)
        pos_enc = self.positional_embedding(pos)

        return word_embedding + pos_enc

    def __call__(
        self,
        src: Int32[Array, "batch_size seq_len"],
        src_mask: Int32[Array, "batch_size 1 1 seq_len"] = None,
        train=True,
    ) -> Float[Array, "batch_size seq_len src_vocab_size"]:
        x = self.embeddings(src)

        x = self.dropout_emb(x, deterministic=not train)

        block_out = self.block(x, x_mask=src_mask, train=train)

        # logits = self.word_emb_src.attend(x) instead of ?
        x = self.lm_out(block_out)

        return x


if __name__ == "__main__":
    from jax import random

    main_rng = random.PRNGKey(42)

    src_vocab_size = 300
    model_dim = 512
    max_length = 12
    batch_size = 2
    dec_num_layers = 2
    num_heads = 4
    dropout_prob = 0.1

    example_input_src = random.randint(
        main_rng, (batch_size, max_length), minval=0, maxval=src_vocab_size
    )
    attn_mask = nn.make_causal_mask(example_input_src, dtype=bool)

    print("example_input_src", example_input_src)
    # print(example_input_src.shape)

    # print("attn_mask", attn_mask)
    # print(attn_mask.shape)

    gpt = GPT(
        src_vocab_size=src_vocab_size,
        n_embd=model_dim,
        dec_num_layers=dec_num_layers,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        seq_len=max_length,
    )

    init_rng, dropout_init_rng, embed_rng = random.split(main_rng, 3)

    params = gpt.init(
        {"params": init_rng, "dropout": dropout_init_rng},
        example_input_src,
        attn_mask,
    )["params"]

    main_rng, dropout_apply_rng = random.split(main_rng)

    out = gpt.apply(
        {"params": params},
        example_input_src,
        attn_mask,
        rngs={"dropout": dropout_apply_rng},
    )

    print("out", out)
    print("out.shape", out.shape)
