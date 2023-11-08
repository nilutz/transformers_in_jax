import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from jaxtyping import Array, Bool, Int32
from tokenizers import Encoding, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from torch.utils.data import DataLoader

import jax.numpy as jnp

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data/vanilla"
if not DATA_PATH.is_dir():
    Path(DATA_PATH).mkdir()

MASK = "[MASK]"

spl_tokens = [
    # "[PAD]",
    "[UNK]",
    "[SEP]",
    "[BOS]",
    "[EOS]",
    MASK,
    "[CLS]",
]  # special tokens


def make_tokenizer(
    p: Path, max_length: int = 100, vocab_size: int = 15000
) -> Tokenizer:
    if not p.exists():
        print("building")

        sentences = load_csv()

        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        # tokenizer.normalizer = BertNormalizer(
        #        clean_text=clean_text,
        #        handle_chinese_chars=handle_chinese_chars,
        #        strip_accents=strip_accents,
        #        lowercase=lowercase,
        #    )
        #    tokenizer.pre_tokenizer = BertPreTokenizer()
        for token in spl_tokens:
            tokenizer.add_special_tokens([token])
        tokenizer.enable_padding(pad_token="[PAD]", length=max_length)
        tokenizer.enable_truncation(max_length=max_length)
        # tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]", length=max_length)
        # print(tokenizer.token_to_id("[CLS]"))
        # print(tokenizer.token_to_id("[SEP]"))

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=spl_tokens,
            min_frequency=2,
            limit_alphabet=1000,
            initial_alphabet=[],
        )

        count = len(sentences)

        def simple_iterator():
            for row in sentences:
                yield row

        tokenizer.train_from_iterator(simple_iterator(), trainer=trainer, length=count)
        tokenizer.save(str(p))
    else:
        tokenizer = Tokenizer.from_file(str(p))

    return tokenizer


MASK_PERCENTAGE = 0.15

argwhere = lambda l, F: [i for i, e in enumerate(l) if F(e)]


def mask_sent(
    enc_sent: Encoding, tokenizer: Tokenizer, tokenizer_vocab_size: int
) -> Encoding:
    new_tokens = [t for t in enc_sent.tokens if t != "[PAD]"]
    len_s = len(new_tokens)
    mask_amount = round(len_s * MASK_PERCENTAGE)

    possible_indexes = argwhere(enc_sent.special_tokens_mask, lambda x: x == 0)
    sampled = random.sample(possible_indexes, mask_amount)
    mask_mask = [0] * len(
        new_tokens
    )  # need a MASK attention where 0 on all but 1 where MASK is set

    for index in sampled:
        if random.random() < 0.8:
            new_tokens[index] = MASK
        else:
            new_tokens[index] = tokenizer.id_to_token(
                random.randint(0, tokenizer_vocab_size - 1)
            )
        mask_mask[index] = 1

    return (
        tokenizer.encode(new_tokens, is_pretokenized=True, add_special_tokens=False),
        mask_mask,
    )


def load_csv() -> pd.Series:
    df = pd.read_csv(f"{DATA_PATH}/IMDB-Dataset.csv")["review"]
    punctuation = r"(?<=[.!?])+"
    sentences = (
        df.str.replace("<br /><br />", "")
        .str.split(punctuation)
        .explode(ignore_index=False)
    )  # the index remains so we can index "paragraphs"
    sentences = sentences.map(lambda x: x.replace("\n", "\\n").strip().lower())
    sentences.replace("", np.nan, inplace=True)
    sentences.replace(".", np.nan, inplace=True)
    sentences.dropna(inplace=True)

    return sentences


def preprocess_sent(tokenizer: Tokenizer, cutoff: int = None) -> list[dict]:
    tokenizer_vocab_size = tokenizer.get_vocab_size()

    sentences = load_csv()

    data = []
    for document_index in range(0, max(sentences.index)):
        print(document_index, "/", max(sentences.index))
        for i in range(0, len(sentences[document_index]), 2):
            if (i + 1) < len(
                sentences[document_index]
            ):  # TODO :( we will loose upon uneven document length - but this is just a demo -so its fine
                # true nsp
                try:
                    sentence_a = sentences[document_index].iloc[i]
                    sentence_b = sentences[document_index].iloc[i + 1]
                except Exception as e:
                    continue  # continue on 1 length paragraphs

                encoded = tokenizer.encode(sentence_a, sentence_b)
                mask_encoded, _ = mask_sent(encoded, tokenizer, tokenizer_vocab_size)

                assert len(encoded.ids) == len(mask_encoded)
                data.append(
                    {
                        "original_input_ids": encoded.ids,  # label for MLM
                        "input_ids": mask_encoded.ids,  # encoded(MASK) input_ids
                        "attention_mask": mask_encoded.attention_mask,
                        "type_ids": encoded.type_ids,  # nsp sent mask
                        "nsp_label": 1,
                        # "mask_mask": mask_mask,
                        "original_tokens": encoded.tokens,
                        "tokens": mask_encoded.tokens,
                    }
                )

                # false nsp
                a_int = random.randint(0, len(sentences) - 1)
                sentence_a = sentences.iloc[a_int]

                b_int = random.randint(0, len(sentences) - 1)
                count = 0
                while (
                    b_int == a_int + 1
                ):  # don't allow the next one that would be a true nsp
                    b_int = random.randint(0, len(sentences) - 1)
                    count += 1
                    if count > 10:
                        print("breaking off", b_int, a_int)
                        break

                sentence_b = sentences.iloc[b_int]

                encoded = tokenizer.encode(sentence_a, sentence_b)
                mask_encoded, _ = mask_sent(encoded, tokenizer, tokenizer_vocab_size)
                data.append(
                    {
                        "original_input_ids": encoded.ids,  # label for MLM
                        "input_ids": mask_encoded.ids,  # encoded(MASK) input_ids
                        "attention_mask": mask_encoded.attention_mask,
                        "type_ids": encoded.type_ids,  # nsp sent mask
                        "nsp_label": 0,
                        # "mask_mask": mask_mask,
                        "original_tokens": encoded.tokens,
                        "tokens": mask_encoded.tokens,
                    }
                )

        if cutoff is not None and document_index == cutoff:
            break
    return data


def make_dataset(vocab_size: int = 15000, max_length: int = 100) -> Dataset:
    p = Path(f"{DATA_PATH}/imdb_preprocessed")

    tokenizer_path = Path(f"{DATA_PATH}/imdb_{vocab_size}_bpe_tokenizer-trained.json")
    tokenizer = make_tokenizer(
        tokenizer_path, vocab_size=vocab_size, max_length=max_length
    )
    if not p.exists():
        print("Building dataset")

        data = preprocess_sent(tokenizer)

        dataset = Dataset.from_pandas(pd.DataFrame(data=data))

        dataset = dataset.shuffle(42)

        train_testvalid = dataset.train_test_split(test_size=0.1)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.5)

        dataset = DatasetDict(
            {
                "train": train_testvalid["train"],
                "test": test_valid["test"],
                "validation": test_valid["train"],
            }
        )
        dataset.save_to_disk(str(p))
    else:
        dataset = load_from_disk(str(p))

    return dataset, tokenizer


def make_loaders(
    batch_size: int = 100,
    vocab_size: int = 15000,
    max_length: int = 100,
    debug: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset, tokenizer = make_dataset(vocab_size, max_length)

    def collate_fn(
        batch,
    ) -> tuple[
        Int32[Array, "batch seq_len"],
        Bool[Array, "batch 1 1 seq_len"],
        Int32[Array, "batch seq_len"],
        Int32[Array, "batch seq_len"],
        Int32[Array, "batch"],
        list[list[str]],
        list[list[str]],
    ]:
        # print(batch)

        original_input_ids = jnp.stack(
            [jnp.array(b["original_input_ids"], dtype=jnp.int32) for b in batch]
        )
        input_ids = jnp.stack(
            [jnp.array(b["input_ids"], dtype=jnp.int32) for b in batch]
        )
        attention_mask = jnp.stack(
            [jnp.array(b["attention_mask"], dtype=jnp.int32) for b in batch]
        )
        type_ids = jnp.stack([jnp.array(b["type_ids"], dtype=jnp.int32) for b in batch])

        # mask_mask = jnp.stack(
        #     [
        #         jnp.pad(
        #             jnp.array(b["mask_mask"], dtype=jnp.int32),
        #             (0, max_length - len(b["mask_mask"])),
        #             mode="constant",
        #         )
        #         for b in batch
        #     ]
        # )

        nsp_label = jnp.stack(
            [jnp.array(b["nsp_label"], dtype=jnp.int32) for b in batch]
        )

        src_mask = jnp.expand_dims(
            jnp.expand_dims(attention_mask, axis=1), axis=1
        ).astype(
            bool
        )  # src_mask = attention_mask

        res = (
            input_ids,  # src model input
            src_mask,  # attention mask model input
            type_ids,  # segment ids model input
            original_input_ids,  # ground truth MLM
            nsp_label,  # ground truth NSP
        )

        if debug:
            tokens = [b["tokens"] for b in batch]
            original_input_tokens = [b["original_tokens"] for b in batch]
            res += (original_input_tokens, tokens)

        return res

    train_loader = DataLoader(
        dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset["test"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn
    )
    validation_loader = DataLoader(
        dataset["validation"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        test_loader,
        validation_loader,
        tokenizer.padding["pad_id"],
    )


if __name__ == "__main__":
    train_loader, _, _, pad_id = make_loaders(batch_size=1, debug=True)

    for (
        input_ids,
        src_mask,
        type_ids,
        original_input_ids,
        nsp_label,
        original_input_tokens,
        tokens,
    ) in train_loader:
        print(input_ids.shape)
        print(src_mask.shape)
        print(type_ids.shape)
        print(nsp_label.shape)
        print(original_input_ids.shape)
        print(original_input_tokens)
        print(tokens)
        print("input_ids", input_ids)

        print("original_input_ids", original_input_ids)

        print("type_ids", type_ids)

        break

    print("pad", pad_id)
