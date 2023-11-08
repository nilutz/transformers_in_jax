# class Config:
#     decoder_num_layers: int = 12
#     n_embd: int = 768
#     heads: int = 12

#     dropout: float = 0.0

#     lr: float = 1e-3
#     warmup: int = 40

#     vocab_size: int = 50257  # enc.n_vocab

#     batch_size: int = 64
#     max_length: int = 100

#     dataset_name: str = "shakespeare"
#     train_steps: int = 100


class Config:
    decoder_num_layers: int = 2
    n_embd: int = 128
    heads: int = 4

    dropout: float = 0.0

    lr: float = 1e-3
    warmup: int = 40

    vocab_size: int = 50257  # enc.n_vocab

    batch_size: int = 64
    max_length: int = 100

    dataset_name: str = "shakespeare"
    train_steps: int = 100
