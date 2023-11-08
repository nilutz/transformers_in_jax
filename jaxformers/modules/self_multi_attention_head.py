import numpy as np
from flax import linen as nn
from jaxtyping import Array, Bool, Float

import jax.numpy as jnp


def scaled_dot_product(
    q: Float[Array, "batch_size num_heads seq_len dim"],
    k: Float[Array, "batch_size num_heads seq_len dims"],
    v: Float[Array, "batch_size num_heads seq_len seq_len"],
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
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
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


class SelfMultiheadAttention(nn.Module):
    """
    An attention function can be described as mapping a query and a set of key-value pairs to an output,
    where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values,
    where the weight assigned to each value is computed by a compatibility function of the
    query with the corresponding key.

    MultiHeadSelfAttention
    MaskedMultiHeadSelfAttention



    ![Attention](.././src/vanilla_transformer/attention.png "Attention")

    ![Attention Detail](../../src/vanilla_transformer/attention_detail.png "Attention Detail")


    """

    embed_dim: int  # Output dimensino
    num_heads: int  # Number of heads

    def setup(self):
        # Instead of having multiple linear layers have a Dense layer
        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )
        self.o_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )

    def __call__(
        self,
        x: Float[Array, "batch_size seq_len model_dim"],
        mask: Bool[Array, "batch_size 1 1 seq_len"] = None,
    ) -> tuple[
        Float[Array, "batch_size seq_len model_dim "],
        Float[Array, "batch_size num_heads seq_len seq_len"],
    ]:
        batch_size, seq_length, embed_dim = x.shape  # get the dim of the input
        if mask is not None:
            mask = expand_mask(mask)

        qkv = self.qkv_proj(x)

        # separte q,k,v from the dense: Linear
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)  # [batch, head, seqLen, dims]

        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Scaled Dot-Product Attention
        values, attention = scaled_dot_product(q, k, v, mask=mask)

        values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        # Concat
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o, attention


if __name__ == "__main__":
    from jax import random

    main_rng = random.PRNGKey(42)

    seq_len, d_k = 3, 2
    main_rng, rand1 = random.split(main_rng)
    qkv = random.normal(rand1, (3, seq_len, d_k))
    q, k, v = qkv[0], qkv[1], qkv[2]
    values, attention = scaled_dot_product(q, k, v)
    print("Q\n", q)
    print("K\n", k)
    print("V\n", v)
    print("Values\n", values)
    print("Attention\n", attention)

    ## Test MultiheadAttention implementation
    # Example features as input
    main_rng, x_rng = random.split(main_rng)
    x = random.normal(x_rng, (3, 16, 128))
    # Create attention
    mh_attn = SelfMultiheadAttention(embed_dim=128, num_heads=32)
    # Initialize parameters of attention with random key and inputs
    main_rng, init_rng = random.split(main_rng)
    params = mh_attn.init(init_rng, x)["params"]
    # Apply attention with parameters on the inputs
    out, attn = mh_attn.apply({"params": params}, x)
    print("Out", out.shape, "Attention", attn.shape)

    del mh_attn, params
