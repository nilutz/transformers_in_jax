from flax import linen as nn
from jaxtyping import Array, Float

import jax


class Generator(nn.Module):
    vocab: int
    use_bias: bool = False

    def setup(self):
        self.proj = nn.Dense(self.vocab, use_bias=self.use_bias)

    def __call__(
        self, x: Float[Array, "batch_size seq_len-1 model_dim"]  # seq_len-1 if shifted
    ) -> Float[Array, "batch_size seq_len-1 tgt_vocab_size"]:
        return jax.nn.log_softmax(self.proj(x), axis=-1)
