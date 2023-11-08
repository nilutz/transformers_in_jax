from pathlib import Path

from datasets import load_dataset
from jaxtyping import Array, Bool, Int32
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader

import jax.numpy as jnp

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data/vanilla"
if not DATA_PATH.is_dir():
    Path(DATA_PATH).mkdir()

spl_tokens = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<SEP>",
    "<MASK>",
    "<CLS>",
]


def make_tokenizer(
    dataset: str, max_length: int = 64, vocab_size: int = 15000
) -> Tokenizer:
    """
    jointly training a bpe tokenizer on de/en
    """

    p = (
        DATA_PATH
        / f"{dataset.replace('/', '-')}_{vocab_size}_bpe_tokenizer-trained.json"
    )
    if not p.exists():
        print("Building BPE Tokenizer", p, "with vocab size", vocab_size)

        dataset = load_dataset(dataset, "de-en")

        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.enable_padding(pad_id=0, pad_token="<PAD>", length=max_length)
        for token in spl_tokens:
            tokenizer.add_special_tokens([token])
        tokenizer.enable_truncation(max_length=max_length)
        trainer = BpeTrainer(
            vocab_size=vocab_size, special_tokens=spl_tokens, min_frequency=3
        )
        tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[("<BOS>", 2), ("<EOS>", 3)],
        )

        count = sum(len(dataset[key]) for key in ["test", "train", "validation"])

        def simple_iterator():
            # multi_k/30
            # for key in ["test", "train", "validation"]:
            #     for row in dataset[key]:
            #         for lang in ["de", "en"]:
            #             yield row[lang]
            # wmt14 en-de
            for key in ["test", "train", "validation"]:
                for row in dataset[key]:
                    for lang in ["de", "en"]:
                        yield row["translation"][lang]

        tokenizer.train_from_iterator(simple_iterator(), trainer=trainer, length=count)
        tokenizer.save(str(p))
    else:
        print("Loading BPE Tokenizer", p)
        tokenizer = Tokenizer.from_file(str(p))

    return tokenizer


def create_src_mask(
    src: Int32[Array, "batch seq_len"], padding_idx: int
) -> Bool[Array, "batch 1 1 seq_len"]:
    return jnp.expand_dims(jnp.expand_dims(src != padding_idx, axis=-2), axis=1)


def create_tgt_mask(
    tgt: Int32[Array, "batch seq_len"], padding_idx: int
) -> Bool[Array, "batch 1 seq_len seq_len"]:
    mask = jnp.expand_dims(tgt != padding_idx, axis=-2)
    attn_shape = (1, tgt.shape[-1], tgt.shape[-1])
    subsequent_mask = jnp.triu(jnp.ones(attn_shape, dtype=jnp.uint8), k=1)
    subsequent_mask = subsequent_mask == 0
    return jnp.expand_dims(mask & subsequent_mask, axis=1).astype(bool)


def create_subsequent_mask(
    mask: Bool[Array, "batch seq_len"]
) -> Bool[Array, "batch 1 seq_len seq_len"]:
    mask = jnp.expand_dims(mask, axis=-2)
    attn_shape = (1, mask.shape[-1], mask.shape[-1])
    subsequent_mask = jnp.triu(jnp.ones(attn_shape, dtype=jnp.uint8), k=1)
    subsequent_mask = subsequent_mask == 0
    return jnp.expand_dims(mask & subsequent_mask, axis=1).astype(bool)


def make_loaders(
    dataset_name: str,
    batch_size: int = 64,
    max_length: int = 64,
    vocab_size: int = 15000,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    tokenizer = make_tokenizer(
        dataset_name, max_length=max_length, vocab_size=vocab_size
    )

    def collate_fn(
        batch,
    ) -> tuple[
        Int32[Array, "batch seq_len"],
        Int32[Array, "batch seq_len"],
        Bool[Array, "batch 1 1 seq_len"],
        Bool[Array, "batch 1 seq_len seq_len"],
    ]:
        batches = []
        attn = []
        for lang in ["de", "en"]:
            tokens = [b["translation"][lang] for b in batch]
            tokens = tokenizer.encode_batch(tokens)
            batchstack = jnp.stack([jnp.array(b.ids, dtype=jnp.int32) for b in tokens])
            batchstack_attention_mask = jnp.stack(
                [jnp.array(b.attention_mask).astype(bool) for b in tokens]
            )
            batches.append(batchstack)
            attn.append(batchstack_attention_mask)

        # custom calculations equals to the hugginface attention masks.
        # src_mask = create_src_mask(batches[0], pad_id)
        # tgt_mask = create_tgt_mask(batches[1], pad_id)
        return (
            batches[0],  # src
            batches[1],  # tgt
            jnp.expand_dims(jnp.expand_dims(attn[0], axis=1), axis=1).astype(
                bool
            ),  # src_mask
            create_subsequent_mask(attn[1]),  # tgt_mask,
        )

    dataset = load_dataset(dataset_name, "de-en")
    train_dataloader = DataLoader(
        dataset["train"].select(list(range(0,2000))), batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset["test"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    validation_dataloader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return (
        train_dataloader,
        test_dataloader,
        validation_dataloader,
        tokenizer.padding["pad_id"],
        tokenizer.get_vocab_size(),
    )


if __name__ == "__main__":
    from .config import Config

    config = Config
    tokenizer = make_tokenizer(
        dataset="wmt14", max_length=config.max_length, vocab_size=config.vocab_size
    )
    (
        train_dataloader,
        test_dataloader,
        validation_dataloader,
        pad_id,
        vocab_size,
    ) = make_loaders(
        dataset_name="wmt14",
        batch_size=1,
        max_length=config.max_length,
        vocab_size=config.vocab_size,
    )

    print("pad_id", pad_id)
    for src, tgt, src_mask, tgt_mask in train_dataloader:
        print(tokenizer.decode_batch(src))
        print("src", src, src.shape)
        print("src_mask", src_mask, src_mask.shape)

        print("tgt", tgt, tgt.shape)
        print("tgt_mask", tgt_mask, tgt_mask.shape)
        print(tokenizer.decode_batch(tgt))

        break
