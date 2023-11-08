import numpy as np
from flax.training.common_utils import onehot
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp


class LabelSmoothing:

    """

    Label Smoothing with cross entropy loss

    > We implement label smoothing using the KL div loss. Instead of using a one-hot target distribution, we create a distribution that has confidence of the correct word and the rest of the smoothing mass distributed throughout the vocabulary.
    """

    def __init__(self, smoothing: float = 0.1, padding_idx: int = 0):
        self.smoothing = smoothing
        self.padding_idx = padding_idx
        self.confidence = 1.0 - self.smoothing

    def __call__(
        self,
        logits: Float[Array, "batch_size*seq_len-1 vocab_size"],
        target: Float[Array, "batch_size*seq_len-1"],
        weights: Float[Array, "batch_size*seq_len-1"] = None,
    ) -> tuple[Float, int]:
        vocab_size = logits.shape[-1]
        low_confidence = (1.0 - self.confidence) / (vocab_size - 1)
        normalizing_constant = -(
            self.confidence * jnp.log(self.confidence)
            + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )

        soft_labels = onehot(
            target, vocab_size, on_value=self.confidence, off_value=low_confidence
        )

        loss = -jnp.sum(soft_labels * jax.nn.log_softmax(logits), axis=-1)
        loss = loss - normalizing_constant

        normalizing_factor = np.prod(target.shape)
        if weights is not None:
            loss = loss * weights
            normalizing_factor = weights.sum()

        return loss.sum(), normalizing_factor
