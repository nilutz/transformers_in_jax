class Config:
    encoder_num_layers: int = 3  # 6
    model_dim: int = 256  # 512
    heads: int = 8

    dropout: float = 0.1

    # lr: float = 1e-3
    # warmup: int = 40
    learning_rate = (1e-3,)
    weight_decay_rate = (0.01,)
    beta_1 = (0.9,)
    beta_2 = (0.999,)
    epsilon = (1e-6,)

    vocab_size: int = 15000

    batch_size: int = 64
    max_length: int = 100

    dataset_name: str = "imdb"
