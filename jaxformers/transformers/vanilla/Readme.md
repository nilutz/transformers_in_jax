# Vanilla Transformer
As described in https://arxiv.org/pdf/1706.03762.pdf

# Architecture:

* Encoder-Decoder
* Encoder:
    * N=6 layers
    * each layer 2 sublayers:
        * multi-head self attention
        * position-wise fully connected feed forward
    * residual connection around each of the two sub layers
    * followed by layer normalization
    * d_model = 512

* Decoder:
    * N=6 layers
    * each layer 3 sublayers:
        * multi-head self attention
        * position-wise fully connected feed forward
        * multi-head attention over the output of the encoder stack
    * residual connection around each of the two sub layers
    * followed by layer normalization
    * d_model = 512

    * modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent position
    * output embeddings are offset by one position

* Multi-head attention
    * h=8

* FeedForward
    * Dense, Relu, Dense
    * d_model = 512
    * d_ff = 4 * d_model


* Dropout = 0.1

* Embedding
    * scaled by sqrt d_Model

* Positional Encoding

* Optimizer: Adam
    * beta_1: 0.9, beta_2 = 0.98, epsilon=10-9
    * learning rate decay
        * warmup steps = 4000

* Label smoothing: e_ls = 0.1

* byte-pair encoding: shared source-target vocabulary of about 37000 token

* Sentence pairs were batched together by approximate sequence length. Each  training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

# Result on WMT 2014 English-to-German

* BLEU 28.4
* averaging the last 5 checkpoints
* beam search with a beam size of 4 and length penalty α = 0.6


# Report

## Training
Preprocess

    python -m jaxformers.transformers.vanilla.preprocessing_wmt14

Actual training

    python -m jaxformers.transformers.vanilla.train