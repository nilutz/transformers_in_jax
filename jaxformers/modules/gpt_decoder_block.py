from flax import linen as nn
from jaxtyping import Array, Bool, Float

from .multi_attention_head import MultiAttentionHead


class GPTDecoderBlock(nn.Module):
    """
    Decoder: Similar to `jaxformers.modules.DecoderBlock` just not using targets

    """

    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: int

    def setup(self):
        self.self_attn = MultiAttentionHead(
            embed_dim=self.input_dim, num_heads=self.num_heads
        )
        self.norm1 = nn.LayerNorm(epsilon=1e-5)
        self.norm2 = nn.LayerNorm(epsilon=1e-5)

        # two layer mlp
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.gelu,
            nn.Dense(self.input_dim),
            nn.Dropout(self.dropout_prob),
        ]

        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(
        self,
        x: Float[Array, "batch_size seq_len model_dim"],
        x_mask: Bool[Array, "batch_size 1 1 seq_len"] = None,
        train: bool = True,
    ) -> Float[Array, "batch_size seq_len-1 model_dim "]:
        # the order in  `jaxformers.modules.DecoderBlock`  is different is is more close to nanoGPT

        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, mask=x_mask)
        x = x + attn_out
        x = self.norm2(x)

        linear_out = x.copy()
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else l(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        return x
