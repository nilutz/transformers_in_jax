class Config:
    # model config
    encoder_num_layers: int = 3  # 6
    decoder_num_layers: int = 3  # 6
    model_dim: int = 512
    heads: int = 4  # 8

    dropout: float = 0.1

    label_smoothing: float = 0.1

    # bpe encoding
    vocab_size: int = 18000  # 37000

    # optimizer
    adam_beta: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 10e-9
    lr: float = 1e-3
    warmup_steps: int = 4000

    # batch
    batch_size: int = 128
    max_length: int = 100

    dataset_name: str = "wmt14"  # "de-en"

    # beam decoding
    beam_size = 4
    beam_penality_alpha = 0.6

    # beam size of 4 and length penalty α = 0.6


# class Config:
#     # model config
#     encoder_num_layers: int = 3  # 6
#     decoder_num_layers: int = 3  # 6
#     model_dim: int = 128
#     heads: int = 4  # 8

#     dropout: float = 0.1

#     label_smoothing: float = 0.1

#     # bpe encoding
#     vocab_size: int = 18000  # 37000

#     # optimizer
#     adam_beta: float = 0.9
#     adam_beta2: float = 0.98
#     adam_epsilon: float = 10e-9
#     lr: float = 1e-3
#     warmup_steps: int = 4000

#     # batch
#     batch_size: int = 128
#     max_length: int = 100

#     dataset_name: str = "wmt14"  # "de-en"

#     # beam decoding
#     beam_size = 4
#     beam_penality_alpha = 0.6

#     # beam size of 4 and length penalty α = 0.6
