from pathlib import Path

import pytest

from chempilot.reactions.tokenization import (
    ReactionSmilesRegexTokenizer,
    ReactionSmilesTokenizer,
)


CHECKPOINT = Path(
    "artifacts/pretrained/day5/"
    "rxnfp_bert_pretrained"
)


def test_regex_tokenization():
    tokenizer = (
        ReactionSmilesRegexTokenizer()
    )

    text = (
        "CC(=O)O.[Na+]>>"
        "BrC[C@@H](N)Cl"
    )

    tokens = tokenizer.tokenize_checked(
        text
    )

    assert "".join(tokens) == text
    assert ">>" in tokens
    assert "[Na+]" in tokens
    assert "[C@@H]" in tokens
    assert "Br" in tokens
    assert "Cl" in tokens


def test_reaction_arrow_is_single_token():
    tokenizer = (
        ReactionSmilesRegexTokenizer()
    )

    assert tokenizer.tokenize(
        "CCO>>CC=O"
    ).count(">>") == 1


def test_invalid_input_type():
    tokenizer = (
        ReactionSmilesRegexTokenizer()
    )

    with pytest.raises(
        TypeError,
        match="received int",
    ):
        tokenizer.tokenize(123)


def test_unmatched_characters_are_rejected():
    tokenizer = (
        ReactionSmilesRegexTokenizer()
    )

    with pytest.raises(
        ValueError,
        match="not recognized",
    ):
        tokenizer.tokenize_checked(
            "CCO_with_invalid_text"
        )


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="RXNFP checkpoint not prepared",
)
def test_pretrained_vocabulary_contract():
    tokenizer = ReactionSmilesTokenizer(
        CHECKPOINT / "vocab.txt"
    )

    assert tokenizer.vocab_size == 591
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.unk_token == "[UNK]"
    assert tokenizer.cls_token == "[CLS]"
    assert tokenizer.sep_token == "[SEP]"
    assert tokenizer.mask_token == "[MASK]"


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="RXNFP checkpoint not prepared",
)
def test_pretrained_tokenization():
    tokenizer = ReactionSmilesTokenizer(
        CHECKPOINT / "vocab.txt"
    )

    reaction = (
        "CC(=O)O.[Na+]>>"
        "BrC[C@@H](N)Cl"
    )

    tokens = tokenizer.tokenize(
        reaction
    )

    assert ">>" in tokens
    assert "[Na+]" in tokens
    assert "[C@@H]" in tokens
    assert "[UNK]" not in tokens


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="RXNFP checkpoint not prepared",
)
def test_batch_encoding_shape():
    tokenizer = ReactionSmilesTokenizer(
        CHECKPOINT / "vocab.txt"
    )

    encoded = tokenizer.tokenize_reactions(
        [
            "CCO>>CC=O",
            "CC(=O)O>>CC(=O)N",
        ],
        max_length=32,
        padding="max_length",
        return_tensors="pt",
    )

    assert encoded["input_ids"].shape == (
        2,
        32,
    )

    assert encoded[
        "attention_mask"
    ].shape == (
        2,
        32,
    )