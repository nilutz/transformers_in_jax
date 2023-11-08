import os
from pathlib import Path

import numpy as np

from .config import Config

data_dir = Path(os.path.dirname(__file__)).parent.parent.parent / "data" / "gpt"

train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

config = Config()

batch_size = (
    config.batch_size
)  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = config.max_length


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = np.stack([(data[i : i + block_size]).astype(np.int64) for i in ix])
    y = np.stack(
        [(data[i + 1 : i + 1 + block_size]).astype(np.int64) for i in ix]
    )  # shifted +1
    return x, y
