"""RXNFP-compatible reaction-SMILES tokenization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from transformers import BertTokenizer


SMILES_TOKEN_PATTERN = (
    r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|"
    r"N|O|S|P|F|I|b|c|n|o|s|p|\||"
    r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|"
    r">>?|\*|\$|\%[0-9]{2}|[0-9])"
)


class ReactionSmilesRegexTokenizer:
    """Split reaction SMILES using the original RXNFP regex."""

    def __init__(
        self,
        pattern: str = SMILES_TOKEN_PATTERN,
    ) -> None:
        self.pattern = pattern
        self.regex = re.compile(pattern)

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string; "
                f"received {type(text).__name__}"
            )

        return self.regex.findall(text)

    def tokenize_checked(
        self,
        text: str,
    ) -> list[str]:
        """Tokenize and reject silently unmatched characters."""

        tokens = self.tokenize(text)
        reconstructed = "".join(tokens)

        if reconstructed != text:
            raise ValueError(
                "Reaction SMILES contains characters not "
                "recognized by the RXNFP tokenizer: "
                f"{text!r}"
            )

        return tokens


class ReactionSmilesTokenizer(BertTokenizer):
    """Modern Transformers wrapper for the RXNFP tokenizer."""

    def __init__(
        self,
        vocab_file: str | Path,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        do_lower_case: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_file=str(vocab_file),
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )

        self.smiles_tokenizer = (
            ReactionSmilesRegexTokenizer()
        )

    def _tokenize(
        self,
        text: str,
        split_special_tokens: bool = False,
    ) -> list[str]:
        del split_special_tokens

        return self.smiles_tokenizer.tokenize_checked(
            text
        )

    @property
    def vocabulary_tokens(self) -> list[str]:
        return list(self.vocab.keys())

    def tokenize_reactions(
        self,
        reactions: Sequence[str],
        *,
        max_length: int = 256,
        padding: bool | str = True,
        truncation: bool = True,
        return_tensors: str | None = None,
    ):
        return self(
            list(reactions),
            add_special_tokens=True,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors=return_tensors,
        )