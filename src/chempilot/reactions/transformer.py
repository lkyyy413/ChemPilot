"""Pretrained Transformer encoding for reaction SMILES."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as functional
from transformers import BertModel

from chempilot.reactions.tokenization import (
    ReactionSmilesTokenizer,
)


PoolingMethod = Literal[
    "cls",
    "masked_mean",
]


EXPECTED_UNUSED_PRETRAINING_KEYS = {
    "cls.predictions.bias",
    "cls.predictions.decoder.weight",
    (
        "cls.predictions.transform."
        "LayerNorm.bias"
    ),
    (
        "cls.predictions.transform."
        "LayerNorm.weight"
    ),
    (
        "cls.predictions.transform."
        "dense.bias"
    ),
    (
        "cls.predictions.transform."
        "dense.weight"
    ),
    "cls.seq_relationship.bias",
    "cls.seq_relationship.weight",
}


@dataclass(frozen=True)
class ReactionTransformerConfig:
    checkpoint_directory: Path = Path(
        "artifacts/pretrained/day5/"
        "rxnfp_bert_pretrained"
    )

    max_length: int = 256
    batch_size: int = 32
    pooling: PoolingMethod = "cls"
    normalize: bool = False
    device: str | None = None

    def __post_init__(self) -> None:
        if self.max_length <= 2:
            raise ValueError(
                "max_length must be greater "
                "than 2."
            )

        if self.max_length > 512:
            raise ValueError(
                "max_length cannot exceed "
                "the RXNFP checkpoint limit "
                "of 512."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.pooling not in {
            "cls",
            "masked_mean",
        }:
            raise ValueError(
                "pooling must be 'cls' or "
                "'masked_mean'."
            )


class ReactionTransformerEncoder:
    """Encode reaction SMILES with pretrained RXNFP BERT."""

    def __init__(
        self,
        config: (
            ReactionTransformerConfig
            | None
        ) = None,
    ) -> None:
        self.config = (
            config
            or ReactionTransformerConfig()
        )

        checkpoint = Path(
            self.config.checkpoint_directory
        )

        required_files = [
            checkpoint / "config.json",
            checkpoint / "pytorch_model.bin",
            checkpoint / "vocab.txt",
        ]

        missing_files = [
            path
            for path in required_files
            if not path.is_file()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Missing RXNFP checkpoint "
                "files: "
                + ", ".join(
                    str(path)
                    for path in missing_files
                )
            )

        self.tokenizer = (
            ReactionSmilesTokenizer(
                checkpoint / "vocab.txt"
            )
        )

        model, loading_info = (
            BertModel.from_pretrained(
                checkpoint,
                local_files_only=True,
                output_loading_info=True,
            )
        )

        self.loading_info = loading_info

        self._validate_loading_info()

        if self.config.device is None:
            device_name = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device_name = self.config.device

        self.device = torch.device(
            device_name
        )

        self.model = model.to(
            self.device
        )

        self.model.eval()

        self.hidden_size = int(
            self.model.config.hidden_size
        )

        if (
            self.config.max_length
            > self.model.config
            .max_position_embeddings
        ):
            raise ValueError(
                "Configured max_length exceeds "
                "the model position limit."
            )

    def _validate_loading_info(
        self,
    ) -> None:
        missing = set(
            self.loading_info[
                "missing_keys"
            ]
        )

        unexpected = set(
            self.loading_info[
                "unexpected_keys"
            ]
        )

        mismatched = self.loading_info[
            "mismatched_keys"
        ]

        errors = self.loading_info[
            "error_msgs"
        ]

        if missing:
            raise RuntimeError(
                "Missing encoder weights: "
                f"{sorted(missing)}"
            )

        if (
            unexpected
            != EXPECTED_UNUSED_PRETRAINING_KEYS
        ):
            raise RuntimeError(
                "Unexpected checkpoint keys "
                "differ from the expected "
                "unused pretraining heads: "
                f"{sorted(unexpected)}"
            )

        if mismatched:
            raise RuntimeError(
                "Mismatched checkpoint "
                f"weights: {mismatched}"
            )

        if errors:
            raise RuntimeError(
                "Checkpoint loading errors: "
                f"{errors}"
            )

    @staticmethod
    def _normalize_input(
        reactions: (
            str
            | Sequence[str]
            | np.ndarray
        ),
    ) -> list[str]:
        if isinstance(reactions, str):
            values = [reactions]

        elif isinstance(
            reactions,
            np.ndarray,
        ):
            values = reactions.tolist()

        elif isinstance(
            reactions,
            Sequence,
        ):
            values = list(reactions)

        else:
            raise TypeError(
                "reactions must be a reaction "
                "SMILES string or a sequence "
                "of strings; received "
                f"{type(reactions).__name__}"
            )

        normalized = []

        for index, value in enumerate(
            values
        ):
            if not isinstance(value, str):
                raise TypeError(
                    "Every reaction must be "
                    "a string; item "
                    f"{index} received "
                    f"{type(value).__name__}"
                )

            value = value.strip()

            if not value:
                raise ValueError(
                    "Reaction SMILES cannot "
                    f"be empty; item {index}."
                )

            normalized.append(value)

        return normalized

    @staticmethod
    def _pool_hidden_states(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        method: PoolingMethod,
    ) -> torch.Tensor:
        if method == "cls":
            return hidden_states[:, 0, :]

        if method == "masked_mean":
            mask = attention_mask.unsqueeze(
                -1
            ).to(
                dtype=hidden_states.dtype
            )

            summed = (
                hidden_states
                * mask
            ).sum(dim=1)

            counts = mask.sum(
                dim=1
            ).clamp(min=1.0)

            return summed / counts

        raise ValueError(
            "Unsupported pooling method: "
            f"{method}"
        )

    def encode(
        self,
        reactions: (
            str
            | Sequence[str]
            | np.ndarray
        ),
        *,
        pooling: (
            PoolingMethod
            | None
        ) = None,
        normalize: bool | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        values = self._normalize_input(
            reactions
        )

        selected_pooling = (
            pooling
            or self.config.pooling
        )

        if selected_pooling not in {
            "cls",
            "masked_mean",
        }:
            raise ValueError(
                "pooling must be 'cls' or "
                "'masked_mean'."
            )

        selected_normalize = (
            self.config.normalize
            if normalize is None
            else normalize
        )

        selected_batch_size = (
            batch_size
            or self.config.batch_size
        )

        if selected_batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if not values:
            return np.empty(
                (
                    0,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )

        batches = []

        for start in range(
            0,
            len(values),
            selected_batch_size,
        ):
            batch = values[
                start:
                start + selected_batch_size
            ]

            encoded = (
                self.tokenizer
                .tokenize_reactions(
                    batch,
                    max_length=(
                        self.config.max_length
                    ),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
            )

            encoded = {
                name: tensor.to(
                    self.device
                )
                for name, tensor in (
                    encoded.items()
                )
            }

            with torch.inference_mode():
                hidden_states = self.model(
                    **encoded
                ).last_hidden_state

                embeddings = (
                    self._pool_hidden_states(
                        hidden_states,
                        encoded[
                            "attention_mask"
                        ],
                        selected_pooling,
                    )
                )

                if selected_normalize:
                    embeddings = (
                        functional.normalize(
                            embeddings,
                            p=2,
                            dim=1,
                        )
                    )

            batches.append(
                embeddings.detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )

        return np.concatenate(
            batches,
            axis=0,
        )