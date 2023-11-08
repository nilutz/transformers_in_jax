from flax import linen as nn
from jaxtyping import Array, Bool, Float

from .multi_attention_head import MultiAttentionHead


class DecoderBlock(nn.Module):
    """
        Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two
    sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
    attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
    around each of the sub-layers, followed by layer normalization. We also modify the self-attention
    sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
    masking, combined with fact that the output embeddings are offset by one position, ensures that the
    predictions for position i can depend only on the known outputs at positions less than i

    ![Decoder](../../src/vanilla_transformer/decoder.png "Decoder")

    """

    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: int

    def setup(self):
        self.self_attn1 = MultiAttentionHead(
            embed_dim=self.input_dim, num_heads=self.num_heads
        )
        self.norm1 = nn.LayerNorm()

        self.dropout = nn.Dropout(self.dropout_prob)

        self.self_attn2 = MultiAttentionHead(
            embed_dim=self.input_dim, num_heads=self.num_heads
        )
        self.norm2 = nn.LayerNorm()

        # two layer mlp
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim),
        ]
        self.norm3 = nn.LayerNorm()

    def __call__(
        self,
        encoder_out: Float[Array, "batch_size seq_len model_dim"],
        tgt: Float[Array, "batch seq_len-1 model_dim"],
        src_mask: Bool[Array, "batch_size 1 1 seq_len"] = None,
        tgt_mask: Bool[Array, "batch_size 1 seq_len-1 seq_len-1"] = None,
        train: bool = True,
    ) -> Float[Array, "batch_size seq_len-1 model_dim "]:
        # First Attention bottom up
        # Masked Multi-Head(x, mask) + Add(x+attn) + Norm(x)
        x = tgt.copy()
        attn_out1, _ = self.self_attn1(tgt, tgt, tgt, mask=tgt_mask)
        x = tgt + self.dropout(attn_out1, deterministic=not train)
        x = self.norm1(x)

        # MultiheadAttentionKQV but with encoder input
        y = x.copy()
        attn_out2, _ = self.self_attn2(encoder_out, y, encoder_out, mask=src_mask)
        y = x + self.dropout(attn_out2, deterministic=not train)
        y = self.norm2(y)

        # Feedforward Part
        # Feedfoward + Add + Norm
        linear_out = y.copy()
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else l(linear_out, deterministic=not train)
            )
        y = y + self.dropout(linear_out, deterministic=not train)
        y = self.norm3(y)
        return y
