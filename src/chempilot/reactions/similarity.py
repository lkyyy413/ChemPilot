"""End-to-end similar-reaction search with condition evidence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from chempilot.reactions.retrieval import (
    ReactionEmbeddingRetriever,
)
from chempilot.reactions.transformer import (
    ReactionTransformerConfig,
    ReactionTransformerEncoder,
)


DEFAULT_CHECKPOINT = Path(
    "artifacts/pretrained/day5/"
    "rxnfp_bert_pretrained"
)

DEFAULT_INDEX_ROOT = Path(
    "data/processed/reactions/"
    "retrieval/day5"
)

DEFAULT_CONDITION_PATH = Path(
    "data/processed/reactions/"
    "d9297630_condition_pairs.parquet"
)


def _normalize_smiles_input(
    value,
    name: str,
) -> list[str]:
    if isinstance(value, str):
        values = [value]

    elif isinstance(
        value,
        np.ndarray,
    ):
        values = value.tolist()

    elif isinstance(
        value,
        Sequence,
    ):
        values = list(value)

    else:
        raise TypeError(
            f"{name} must be a SMILES "
            "string or a sequence of "
            "SMILES strings; received "
            f"{type(value).__name__}"
        )

    normalized = []

    for index, item in enumerate(
        values
    ):
        if not isinstance(item, str):
            raise TypeError(
                f"{name} item {index} "
                "must be a string; "
                f"received "
                f"{type(item).__name__}"
            )

        item = item.strip()

        if not item:
            raise ValueError(
                f"{name} item {index} "
                "cannot be empty."
            )

        molecule = Chem.MolFromSmiles(
            item
        )

        if molecule is None:
            raise ValueError(
                f"Invalid SMILES in "
                f"{name} item {index}: "
                f"{item!r}"
            )

        for atom in molecule.GetAtoms():
            if atom.HasProp(
                "molAtomMapNumber"
            ):
                atom.ClearProp(
                    "molAtomMapNumber"
                )

        canonical = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )

        normalized.append(canonical)

    return sorted(normalized)


def _list_value(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            str(item)
            for item in value
            if item is not None
        ]

    return [str(value)]


def _finite_or_none(value):
    if value is None:
        return None

    try:
        numeric = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(numeric):
        return None

    return numeric


class SimilarReactionSearch:
    """Search RXNFP neighbors and return historical condition evidence."""

    def __init__(
        self,
        *,
        protocol: str = (
            "transformation"
        ),
        device: str | None = None,
        checkpoint_directory: Path = (
            DEFAULT_CHECKPOINT
        ),
        index_root: Path = (
            DEFAULT_INDEX_ROOT
        ),
        condition_path: Path = (
            DEFAULT_CONDITION_PATH
        ),
    ) -> None:
        if protocol not in {
            "transformation",
            "reaction_center",
        }:
            raise ValueError(
                "protocol must be "
                "'transformation' or "
                "'reaction_center'."
            )

        self.protocol = protocol
        self.pooling = "cls"

        index_directory = (
            Path(index_root)
            / protocol
        )

        index_path = (
            index_directory
            / "cls_index.npz"
        )

        metadata_path = (
            index_directory
            / "cls_metadata.parquet"
        )

        if not index_path.is_file():
            raise FileNotFoundError(
                f"Missing retrieval index: "
                f"{index_path}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Missing retrieval metadata: "
                f"{metadata_path}"
            )

        archive = np.load(
            index_path
        )

        self.index_embeddings = archive[
            "embeddings"
        ].astype(
            np.float32,
            copy=False,
        )

        self.index_feature_indices = (
            archive[
                "feature_indices"
            ].astype(
                np.int64,
                copy=False,
            )
        )

        self.metadata = (
            pd.read_parquet(
                metadata_path
            )
        )

        if (
            len(self.metadata)
            != len(
                self.index_embeddings
            )
        ):
            raise RuntimeError(
                "Retrieval embedding and "
                "metadata row mismatch."
            )

        expected_rows = np.arange(
            len(self.metadata)
        )

        if not np.array_equal(
            self.metadata[
                "retrieval_index_row"
            ].to_numpy(),
            expected_rows,
        ):
            raise RuntimeError(
                "Retrieval metadata rows "
                "are not aligned."
            )

        self.retriever = (
            ReactionEmbeddingRetriever(
                self.index_embeddings
            )
        )

        self.encoder = (
            ReactionTransformerEncoder(
                ReactionTransformerConfig(
                    checkpoint_directory=(
                        Path(
                            checkpoint_directory
                        )
                    ),
                    max_length=256,
                    batch_size=16,
                    pooling="cls",
                    normalize=False,
                    device=device,
                )
            )
        )

        self.condition_pairs = (
            pd.read_parquet(
                condition_path
            )
        )

        self.condition_groups = {
            transformation: group
            for transformation, group in (
                self.condition_pairs.groupby(
                    "transformation_signature",
                    sort=False,
                )
            )
        }

    @staticmethod
    def canonical_reaction(
        reactant_smiles,
        product_smiles,
    ) -> tuple[
        list[str],
        list[str],
        str,
    ]:
        reactants = (
            _normalize_smiles_input(
                reactant_smiles,
                "reactant_smiles",
            )
        )

        products = (
            _normalize_smiles_input(
                product_smiles,
                "product_smiles",
            )
        )

        reaction = (
            ".".join(reactants)
            + ">>"
            + ".".join(products)
        )

        return (
            reactants,
            products,
            reaction,
        )

    def _condition_evidence(
        self,
        transformation_signature,
    ) -> dict:
        group = self.condition_groups.get(
            transformation_signature
        )

        if (
            group is None
            or len(group) == 0
        ):
            return {
                "condition_pairs": 0,
                "best_historical_condition": (
                    None
                ),
            }

        ordered = group.sort_values(
            [
                "score_median",
                "replicate_count",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )

        best = ordered.iloc[0]

        return {
            "condition_pairs": int(
                len(group)
            ),
            "represented_experiments": int(
                group[
                    "replicate_count"
                ].sum()
            ),
            "best_historical_condition": {
                "solvents": _list_value(
                    best[
                        "solvent_labels"
                    ]
                ),
                "catalysts": _list_value(
                    best[
                        "catalyst_labels"
                    ]
                ),
                "reagents": _list_value(
                    best[
                        "reagent_labels"
                    ]
                ),
                "temperature_celsius": (
                    _finite_or_none(
                        best[
                            "temperature_celsius"
                        ]
                    )
                ),
                "reaction_time_hours": (
                    _finite_or_none(
                        best[
                            "reaction_time_hours"
                        ]
                    )
                ),
                "lc_area_percent": (
                    _finite_or_none(
                        best[
                            "score_median"
                        ]
                    )
                ),
                "replicate_count": int(
                    best[
                        "replicate_count"
                    ]
                ),
                "representative_reaction_id": (
                    str(
                        best[
                            "representative_reaction_id"
                        ]
                    )
                ),
            },
        }

    def search(
        self,
        *,
        reactant_smiles,
        product_smiles,
        top_k: int = 5,
    ) -> dict:
        (
            reactants,
            products,
            reaction,
        ) = self.canonical_reaction(
            reactant_smiles,
            product_smiles,
        )

        embedding = self.encoder.encode(
            reaction,
            pooling=self.pooling,
            normalize=False,
        )

        indices, similarities = (
            self.retriever.search(
                embedding,
                top_k=top_k,
            )
        )

        neighbors = []

        for rank, (
            local_index,
            similarity,
        ) in enumerate(
            zip(
                indices[0],
                similarities[0],
            ),
            start=1,
        ):
            metadata = self.metadata.iloc[
                int(local_index)
            ]

            transformation = metadata[
                "transformation_signature"
            ]

            neighbors.append(
                {
                    "rank": rank,
                    "similarity": float(
                        np.clip(
                            similarity,
                            -1.0,
                            1.0,
                        )
                    ),
                    "transformation_signature": (
                        transformation
                    ),
                    "canonical_reaction": (
                        metadata[
                            "canonical_reaction"
                        ]
                    ),
                    "reaction_type": (
                        metadata[
                            "reaction_type"
                        ]
                    ),
                    "reaction_center_signature": (
                        metadata[
                            "reaction_center_signature"
                        ]
                    ),
                    "condition_evidence": (
                        self._condition_evidence(
                            transformation
                        )
                    ),
                }
            )

        return {
            "protocol": self.protocol,
            "embedding": (
                "RXNFP bert_pretrained CLS"
            ),
            "input": {
                "reactant_smiles": (
                    reactants
                ),
                "product_smiles": (
                    products
                ),
                "canonical_reaction": (
                    reaction
                ),
            },
            "index": {
                "splits": [
                    "train",
                    "valid",
                ],
                "reactions": int(
                    len(self.metadata)
                ),
            },
            "neighbors": neighbors,
            "interpretation": {
                "similarity": (
                    "Cosine similarity in "
                    "the RXNFP embedding "
                    "space; it is not a "
                    "calibrated probability."
                ),
                "score": (
                    "lc_area_percent is a "
                    "semi-quantitative LC "
                    "area response at 280 nm, "
                    "not isolated yield."
                ),
            },
        }