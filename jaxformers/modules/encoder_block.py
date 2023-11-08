from flax import linen as nn
from jaxtyping import Array, Bool, Float

from .multi_attention_head import MultiAttentionHead


class EncoderBlock(nn.Module):
    """
        Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two
    sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-
    wise fully connected feed-forward network. We employ a residual connection [ 11 ] around each of
    the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
    LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
    itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
    layers, produce outputs of dimension dmodel = 512.

    ![Encoder](../../src/vanilla_transformer/encoder.png "Encoder")


    """

    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: int

    def setup(self):
        self.self_attn = MultiAttentionHead(
            embed_dim=self.input_dim, num_heads=self.num_heads
        )
        self.norm1 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

        # two layer mlp
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim),
        ]
        self.norm2 = nn.LayerNorm()

    def __call__(
        self,
        x: Float[Array, "batch seq_len model_dim"],
        mask: Bool[Array, "batch_size seq_len"] = None,
        train: bool = True,
    ) -> Float[Array, "batch seq_len model_dim"]:
        # Attention Part
        # Multihead(x) + Add(x+attn) + Norm(x)
        attn_out, _ = self.self_attn(x, x, x, mask=mask)

        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        # Feedforward Part
        # Feedfoward + Add + Norm
        linear_out = x
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else l(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)
        return x


if __name__ == "__main__":
    from jax import random

    main_rng = random.PRNGKey(42)
    ## Test EncoderBlock implementation
    # Example features as input
    main_rng, x_rng = random.split(main_rng)
    x = random.normal(x_rng, (3, 16, 128))
    # Create encoder block
    encblock = EncoderBlock(
        input_dim=128, num_heads=4, dim_feedforward=512, dropout_prob=0.1
    )
    # Initialize parameters of encoder block with random key and inputs
    main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
    params = encblock.init(
        {"params": init_rng, "dropout": dropout_init_rng}, x, train=True
    )["params"]
    # Apply encoder block with parameters on the inputs
    # Since dropout is stochastic, we need to pass a rng to the forward
    main_rng, dropout_apply_rng = random.split(main_rng)
    out = encblock.apply(
        {"params": params}, x, train=True, rngs={"dropout": dropout_apply_rng}
    )

    del encblock, params
