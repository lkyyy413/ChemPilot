"""Build cached RXNFP Transformer reaction features."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chempilot.reactions.transformer import (
    ReactionTransformerConfig,
    ReactionTransformerEncoder,
)


DATA_PATH = Path(
    "data/processed/reactions/"
    "d9297630_condition_pairs.parquet"
)

TRANSFORMATION_SAMPLES_PATH = Path(
    "data/processed/reactions/targets/"
    "transformation/solvent/"
    "samples.parquet"
)

REACTION_CENTER_SAMPLES_PATH = Path(
    "data/processed/reactions/targets/"
    "reaction_center/solvent/"
    "samples.parquet"
)

CHECKPOINT_DIRECTORY = Path(
    "artifacts/pretrained/day5/"
    "rxnfp_bert_pretrained"
)

OUTPUT_DIRECTORY = Path(
    "data/processed/reactions/"
    "features/day5"
)

EMBEDDING_PATH = (
    OUTPUT_DIRECTORY
    / "rxnfp_reaction_embeddings.npz"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "rxnfp_reaction_metadata.parquet"
)

MANIFEST_PATH = Path(
    "reports/day5/embeddings/"
    "rxnfp_embedding_manifest.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device, for example "
            "cuda, cuda:0, or cpu."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalize_smiles(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if hasattr(value, "tolist"):
        value = value.tolist()

    return [
        str(item)
        for item in value
        if item is not None
        and str(item).strip()
    ]


def canonical_sequence(row) -> str:
    reactants = sorted(
        normalize_smiles(
            row["reactant_smiles"]
        )
    )

    products = sorted(
        normalize_smiles(
            row["product_smiles"]
        )
    )

    return (
        ".".join(reactants)
        + ">>"
        + ".".join(products)
    )


def norm_statistics(
    matrix: np.ndarray,
) -> dict:
    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    return {
        "minimum": float(
            norms.min()
        ),
        "mean": float(
            norms.mean()
        ),
        "median": float(
            np.median(norms)
        ),
        "maximum": float(
            norms.max()
        ),
    }


def main() -> None:
    arguments = parse_arguments()

    started = time.perf_counter()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading reaction data..."
    )

    dataframe = pd.read_parquet(
        DATA_PATH
    )

    transformation_samples = (
        pd.read_parquet(
            TRANSFORMATION_SAMPLES_PATH
        )
    )

    reaction_center_samples = (
        pd.read_parquet(
            REACTION_CENTER_SAMPLES_PATH
        )
    )

    transformation_set = set(
        transformation_samples[
            "transformation_signature"
        ]
    )

    reaction_center_set = set(
        reaction_center_samples[
            "transformation_signature"
        ]
    )

    assert (
        transformation_set
        == reaction_center_set
    )

    representative = (
        dataframe.sort_values(
            [
                "transformation_signature",
                "representative_reaction_id",
            ]
        )
        .drop_duplicates(
            "transformation_signature"
        )
        .loc[
            lambda frame: frame[
                "transformation_signature"
            ].isin(
                transformation_set
            )
        ]
        .copy()
        .sort_values(
            "transformation_signature"
        )
        .reset_index(drop=True)
    )

    representative[
        "canonical_reaction"
    ] = representative.apply(
        canonical_sequence,
        axis=1,
    )

    transformation_split = (
        transformation_samples.set_index(
            "transformation_signature"
        )["split"].to_dict()
    )

    reaction_center_split = (
        reaction_center_samples.set_index(
            "transformation_signature"
        )["split"].to_dict()
    )

    representative[
        "transformation_split"
    ] = representative[
        "transformation_signature"
    ].map(transformation_split)

    representative[
        "reaction_center_split"
    ] = representative[
        "transformation_signature"
    ].map(reaction_center_split)

    representative[
        "transformer_feature_row_index"
    ] = np.arange(
        len(representative),
        dtype=np.int64,
    )

    representative[
        "canonical_reaction_sha256"
    ] = representative[
        "canonical_reaction"
    ].map(sha256_text)

    assert len(representative) == 381

    assert representative[
        "transformation_signature"
    ].is_unique

    assert representative[
        "canonical_reaction"
    ].is_unique

    print(
        "Loading pretrained RXNFP encoder..."
    )

    encoder = ReactionTransformerEncoder(
        ReactionTransformerConfig(
            checkpoint_directory=(
                CHECKPOINT_DIRECTORY
            ),
            max_length=(
                arguments.max_length
            ),
            batch_size=(
                arguments.batch_size
            ),
            device=arguments.device,
        )
    )

    reactions = representative[
        "canonical_reaction"
    ].tolist()

    print(
        "Encoding CLS embeddings..."
    )

    cls_embeddings = encoder.encode(
        reactions,
        pooling="cls",
        normalize=False,
    )

    print(
        "Encoding masked-mean embeddings..."
    )

    mean_embeddings = encoder.encode(
        reactions,
        pooling="masked_mean",
        normalize=False,
    )

    assert cls_embeddings.shape == (
        381,
        256,
    )

    assert mean_embeddings.shape == (
        381,
        256,
    )

    assert cls_embeddings.dtype == (
        np.float32
    )

    assert mean_embeddings.dtype == (
        np.float32
    )

    assert np.isfinite(
        cls_embeddings
    ).all()

    assert np.isfinite(
        mean_embeddings
    ).all()

    np.savez_compressed(
        EMBEDDING_PATH,
        cls=cls_embeddings,
        masked_mean=mean_embeddings,
    )

    metadata_columns = [
        "transformer_feature_row_index",
        "transformation_signature",
        "reaction_center_signature",
        "reaction_type",
        "canonical_reaction",
        "canonical_reaction_sha256",
        "transformation_split",
        "reaction_center_split",
    ]

    representative[
        metadata_columns
    ].to_parquet(
        METADATA_PATH,
        index=False,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input": {
            "condition_pair_path": str(
                DATA_PATH
            ),
            "condition_pair_sha256": (
                sha256_file(DATA_PATH)
            ),
            "modeling_transformations": (
                len(representative)
            ),
            "sequence_definition": (
                "sorted canonical reactants"
                ">>"
                "sorted canonical products"
            ),
        },
        "checkpoint": {
            "directory": str(
                CHECKPOINT_DIRECTORY
            ),
            "config_sha256": (
                sha256_file(
                    CHECKPOINT_DIRECTORY
                    / "config.json"
                )
            ),
            "weights_sha256": (
                sha256_file(
                    CHECKPOINT_DIRECTORY
                    / "pytorch_model.bin"
                )
            ),
            "vocabulary_sha256": (
                sha256_file(
                    CHECKPOINT_DIRECTORY
                    / "vocab.txt"
                )
            ),
        },
        "encoder": {
            "model": (
                "RXNFP BERT pretrained"
            ),
            "hidden_size": (
                encoder.hidden_size
            ),
            "max_length": (
                arguments.max_length
            ),
            "batch_size": (
                arguments.batch_size
            ),
            "device": str(
                encoder.device
            ),
            "fine_tuned": False,
            "pooling_methods": [
                "cls",
                "masked_mean",
            ],
            "normalized_on_disk": False,
        },
        "outputs": {
            "embedding_path": str(
                EMBEDDING_PATH
            ),
            "embedding_sha256": (
                sha256_file(
                    EMBEDDING_PATH
                )
            ),
            "metadata_path": str(
                METADATA_PATH
            ),
            "metadata_sha256": (
                sha256_file(
                    METADATA_PATH
                )
            ),
            "shape": [
                381,
                256,
            ],
            "dtype": "float32",
            "cls_norm_statistics": (
                norm_statistics(
                    cls_embeddings
                )
            ),
            "masked_mean_norm_statistics": (
                norm_statistics(
                    mean_embeddings
                )
            ),
        },
        "elapsed_seconds": elapsed,
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    print("\nFeature cache completed")
    print("-----------------------")
    print(
        "Reactions:",
        len(representative),
    )
    print(
        "CLS shape:",
        cls_embeddings.shape,
    )
    print(
        "Masked-mean shape:",
        mean_embeddings.shape,
    )
    print(
        "Device:",
        encoder.device,
    )
    print(
        "Elapsed seconds:",
        round(elapsed, 2),
    )
    print("Saved:", EMBEDDING_PATH)
    print("Saved:", METADATA_PATH)
    print("Saved:", MANIFEST_PATH)


if __name__ == "__main__":
    main()