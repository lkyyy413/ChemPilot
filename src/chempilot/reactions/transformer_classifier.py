"""Partially fine-tuned reaction Transformer classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from transformers import BertModel

from chempilot.reactions.transformer import (
    EXPECTED_UNUSED_PRETRAINING_KEYS,
)


PoolingMethod = Literal[
    "cls",
    "masked_mean",
]


@dataclass(frozen=True)
class ReactionTransformerClassifierConfig:
    number_of_labels: int

    checkpoint_directory: Path = Path(
        "artifacts/pretrained/day5/"
        "rxnfp_bert_pretrained"
    )

    pooling: PoolingMethod = (
        "masked_mean"
    )

    unfreeze_last_n_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.number_of_labels <= 0:
            raise ValueError(
                "number_of_labels must be "
                "positive."
            )

        if self.pooling not in {
            "cls",
            "masked_mean",
        }:
            raise ValueError(
                "pooling must be 'cls' or "
                "'masked_mean'."
            )

        if not (
            0
            <= self.unfreeze_last_n_layers
            <= 12
        ):
            raise ValueError(
                "unfreeze_last_n_layers must "
                "be between 0 and 12."
            )

        if not (
            0.0
            <= self.dropout
            < 1.0
        ):
            raise ValueError(
                "dropout must be in "
                "[0, 1)."
            )


class ReactionTransformerMultiLabelClassifier(
    nn.Module
):
    """RXNFP BERT with a multi-label classification head."""

    def __init__(
        self,
        config: (
            ReactionTransformerClassifierConfig
        ),
    ) -> None:
        super().__init__()

        self.config = config

        checkpoint = Path(
            config.checkpoint_directory
        )

        encoder, loading_info = (
            BertModel.from_pretrained(
                checkpoint,
                local_files_only=True,
                output_loading_info=True,
            )
        )

        self.loading_info = loading_info
        self._validate_loading_info()

        self.encoder = encoder

        for parameter in (
            self.encoder.parameters()
        ):
            parameter.requires_grad = False

        number_to_unfreeze = (
            config.unfreeze_last_n_layers
        )

        if number_to_unfreeze > 0:
            layers = (
                self.encoder.encoder.layer[
                    -number_to_unfreeze:
                ]
            )

            for layer in layers:
                for parameter in (
                    layer.parameters()
                ):
                    parameter.requires_grad = (
                        True
                    )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            config.number_of_labels,
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
                "Unexpected checkpoint keys: "
                f"{sorted(unexpected)}"
            )

        if mismatched:
            raise RuntimeError(
                "Mismatched weights: "
                f"{mismatched}"
            )

        if errors:
            raise RuntimeError(
                "Checkpoint errors: "
                f"{errors}"
            )

    @property
    def trainable_encoder_layer_indices(
        self,
    ) -> list[int]:
        indices = []

        for index, layer in enumerate(
            self.encoder.encoder.layer
        ):
            if any(
                parameter.requires_grad
                for parameter in (
                    layer.parameters()
                )
            ):
                indices.append(index)

        return indices

    def parameter_summary(self) -> dict:
        encoder_total = sum(
            parameter.numel()
            for parameter in (
                self.encoder.parameters()
            )
        )

        encoder_trainable = sum(
            parameter.numel()
            for parameter in (
                self.encoder.parameters()
            )
            if parameter.requires_grad
        )

        classifier_total = sum(
            parameter.numel()
            for parameter in (
                self.classifier.parameters()
            )
        )

        return {
            "encoder_total": (
                encoder_total
            ),
            "encoder_trainable": (
                encoder_trainable
            ),
            "classifier_trainable": (
                classifier_total
            ),
            "total_trainable": (
                encoder_trainable
                + classifier_total
            ),
            "trainable_encoder_rate": (
                encoder_trainable
                / encoder_total
            ),
            "trainable_encoder_layers": (
                self
                .trainable_encoder_layer_indices
            ),
        }

    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.pooling == "cls":
            return hidden_states[:, 0, :]

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: (
            torch.Tensor
            | None
        ) = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=(
                token_type_ids
            ),
        )

        pooled = self.pool(
            outputs.last_hidden_state,
            attention_mask,
        )

        pooled = self.dropout(pooled)

        return self.classifier(pooled)