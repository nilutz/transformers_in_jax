import numpy as np
from flax import linen as nn
from jaxtyping import Array, Bool, Float

import jax.numpy as jnp


def scaled_dot_product(
    q: Float[Array, "batch_size num_heads seq_len num_hiddens/num_heads"],
    k: Float[Array, "batch_size num_heads seq_len num_hiddens/num_heads"],
    v: Float[Array, "batch_size num_heads seq_len num_hiddens/num_heads"],
    mask: Bool[Array, "batch_size 1 1 seq_len"],
) -> tuple[
    Float[Array, "batch_size num_heads seq_len "],
    Float[Array, "batch_size num_heads dims dims"],
]:
    r"""

    ![Scaled Dot Product](../../src/vanilla_transformer/scaled_dot_product.png "Scaled Dot Product")

    ...In practice, we compute the attention function on a set of queries simultaneously, packed together
    into a matrix Q. The keys and values are also packed together into matrices K and V . We compute
    the matrix of outputs as:

    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

    """

    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / np.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)  # 0 = padindex
    attention = nn.softmax(attn_logits, axis=-1)

    # TODO if dropout
    values = jnp.matmul(attention, v)
    return values, attention


# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert (
        mask.ndim > 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = jnp.expand_dims(mask, axis=1)
    while mask.ndim < 4:
        mask = jnp.expand_dims(mask, axis=0)
    return mask


class MultiAttentionHead(nn.Module):
    """
    An attention function can be described as mapping a query and a set of key-value pairs to an output,
    where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values,
    where the weight assigned to each value is computed by a compatibility function of the
    query with the corresponding key.

    Same as MultiheadAttention just with a different input to __call__ the separte k,q,v which are then concatenated.



    ![Attention](.././src/vanilla_transformer/attention.png "Attention")

    ![Attention Detail](../../src/vanilla_transformer/attention_detail.png "Attention Detail")


    """

    embed_dim: int  # Output dimension
    num_heads: int  # Number of heads

    def setup(self):
        self.W_q = nn.Dense(self.embed_dim)
        self.W_k = nn.Dense(self.embed_dim)
        self.W_v = nn.Dense(self.embed_dim)
        self.output = nn.Dense(self.embed_dim)

    def transpose_qkv(
        self, X: Float[Array, "batch_size seq_len mode_dim"]
    ) -> Float[Array, "batch_size num_heads seq_len num_hiddens/num_heads"]:
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(
            (
                X.shape[0],
                X.shape[1],
                self.num_heads,
                int(self.embed_dim / self.num_heads),
            )
        )

        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = jnp.transpose(X, (0, 2, 1, 3))

        return X

    def __call__(
        self,
        k: Float[Array, "batch_size seq_len model_dim"],
        q: Float[Array, "batch_size seq_len model_dim"],
        v: Float[Array, "batch_size seq_len model_dim"],
        mask: Float[Array, "batch_size 1 seq_len seq_len"] = None,
    ) -> tuple[
        Float[Array, "batch_size seq_len model_dim"],
        Float[Array, "batch_size num_heads seq_len seq_len"],
    ]:
        keys = self.transpose_qkv(self.W_k(k))
        queries = self.transpose_qkv(self.W_q(q))
        values = self.transpose_qkv(self.W_v(v))

        nbatches = queries.shape[0]

        if mask is not None:
            mask = expand_mask(mask)

        # print("queries", queries.shape)
        # print("keys", keys.shape)
        # print("values", values.shape)
        # print("mask", mask.shape)

        values, attention = scaled_dot_product(queries, keys, values, mask=mask)

        values = jnp.transpose(values, (0, 2, 1, 3))

        values = values.reshape(
            nbatches, -1, self.num_heads * int(self.embed_dim / self.num_heads)
        )

        o = self.output(values)

        return o, attention
