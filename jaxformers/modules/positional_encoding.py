import numpy as np
from flax import linen as nn
from jaxtyping import Array, Float, Int32

import jax


class PositionalEncoding(nn.Module):
    r"""
    ...To this end, we add "positional encodings" to the input embeddings at the
    bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
    as the embeddings, so that the two can be summed. There are many choices of positional encodings,
    learned and fixed [9].
    In this work, we use sine and cosine functions of different frequencies:
        $$PE_{pos, 2i} = sin(pos/10000^{2i/d_model})$$
        $$PE_{pos, 2i+1} = sin(cos/10000^{2i/d_model})$$

    ![Positional Encoding](../../src/vanilla_transformer/positional_encoding.png "Positional Encoding")

    ...
    """

    d_model: int  # Hidden dimensionality of the input.
    max_len: int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        """
        Using standard numpy here and device_put shove it to the device.
        """

        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(
        self, x: Int32[Array, "batch_size seq_len"]
    ) -> Float[Array, "1 seq_len model_dim"]:
        # x = x + self.pe[:, : x.shape[1]]
        x = self.pe[:, : x.shape[1]]
        return x


if __name__ == "__main__":
    ## Imports for plotting
    import matplotlib.pyplot as plt

    import jax

    plt.set_cmap("cividis")
    # %matplotlib inline
    # from IPython.display import set_matplotlib_formats
    # set_matplotlib_formats('svg', 'pdf') # For export
    import matplotlib
    from matplotlib.colors import to_rgb

    matplotlib.rcParams["lines.linewidth"] = 2.0
    import seaborn as sns

    sns.reset_orig()

    # Create encoding block, bind to access positional encoding (module has no parameters)
    position = PositionalEncoding(d_model=48, max_len=100).bind({})
    # Obtain positional encodings as numpy array
    pe = jax.device_get(position.pe.squeeze().T)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Hidden dimension")
    ax.set_title("Positional encoding over hidden dimensions")
    ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
    ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])
    plt.show()

    sns.set_theme()
    fig, ax = plt.subplots(2, 2, figsize=(12, 4))
    ax = [a for a_list in ax for a in a_list]
    for i in range(len(ax)):
        ax[i].plot(
            np.arange(1, 17),
            pe[i, :16],
            color=f"C{i}",
            marker="o",
            markersize=6,
            markeredgecolor="black",
        )
        ax[i].set_title(f"Encoding in hidden dimension {i+1}")
        ax[i].set_xlabel("Position in sequence", fontsize=10)
        ax[i].set_ylabel("Positional encoding", fontsize=10)
        ax[i].set_xticks(np.arange(1, 17))
        ax[i].tick_params(axis="both", which="major", labelsize=10)
        ax[i].tick_params(axis="both", which="minor", labelsize=8)
        ax[i].set_ylim(-1.2, 1.2)
    fig.subplots_adjust(hspace=0.8)
    sns.reset_orig()
    plt.show()
