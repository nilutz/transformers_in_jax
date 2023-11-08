import flax.linen as nn
from jaxtyping import Array, Int

import jax.numpy as jnp


class CrossEntropyLoss:
    """
    Cross entropy that has the possibility ignore an index
    otherwise it is similar to "optax.softmax_cross_entropy(logits, labels).sum()".
    """

    def __call__(self, logits: Array, labels: Array, ignore_index: int = None) -> Int:
        ce_loss = -jnp.sum(nn.log_softmax(logits) * labels, axis=-1)

        if ignore_index is not None:
            mask = jnp.not_equal(jnp.argmax(labels, axis=-1), ignore_index)
            ce_loss = ce_loss * mask
            return jnp.sum(ce_loss) / jnp.sum(mask)

        return jnp.sum(ce_loss)


if __name__ == "__main__":
    import optax

    from jax import random

    batch_size = 3
    num_classes = 10

    logits = random.normal(random.PRNGKey(0), (batch_size, num_classes))
    labels = nn.one_hot(
        random.randint(random.PRNGKey(1), (batch_size,), 0, num_classes), num_classes
    )
    print(labels.shape)
    labels = random.randint(
        random.PRNGKey(1), (batch_size, num_classes), 0, num_classes
    )

    loss_fn = CrossEntropyLoss()

    print("logits")
    print(logits)
    print(logits.shape)
    print("labels")
    print(labels)
    print(labels.shape)

    loss = loss_fn(logits, labels, ignore_index=0)
    print("loss")
    print(loss, type(loss))

    oloss = optax.softmax_cross_entropy(logits, labels)
    print("optaxloss")
    print(oloss, oloss.sum() / 2)
