from __future__ import annotations

import hashlib
import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy
from scipy import sparse

from rdkit import rdBase

from chempilot.reactions.features import (
    ConditionFingerprintFeaturizer,
    ReactionFingerprintFeaturizer,
)


INPUT_PATH = Path(
    "data/processed/reactions/"
    "d9297630_condition_pairs.parquet"
)

OUTPUT_DIRECTORY = Path(
    "data/processed/reactions/features"
)

REACTION_FEATURE_PATH = (
    OUTPUT_DIRECTORY
    / "reaction_combined.npz"
)

CONDITION_FEATURE_PATH = (
    OUTPUT_DIRECTORY
    / "condition_combined.npz"
)

METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "feature_metadata.parquet"
)

MANIFEST_PATH = Path(
    "reports/day4/"
    "reaction_feature_manifest.json"
)

BATCH_SIZE = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def build_reaction_matrix(
    dataframe: pd.DataFrame,
    featurizer: (
        ReactionFingerprintFeaturizer
    ),
) -> tuple[
    sparse.csr_matrix,
    int,
]:
    # 每个 transformation 的反应物和产物相同，
    # 因此只计算一次，再映射回所有条件行。
    transformations = (
        dataframe[
            [
                "transformation_signature",
                "reactant_smiles",
                "product_smiles",
            ]
        ]
        .drop_duplicates(
            "transformation_signature"
        )
        .reset_index(drop=True)
    )

    print(
        "Unique transformations:",
        len(transformations),
    )

    features = featurizer.transform(
        reactant_smiles=(
            transformations[
                "reactant_smiles"
            ].tolist()
        ),
        product_smiles=(
            transformations[
                "product_smiles"
            ].tolist()
        ),
    )

    unique_matrix = sparse.csr_matrix(
        features["combined"],
        dtype=np.int8,
    )

    transformation_to_index = {
        transformation: index
        for index, transformation in enumerate(
            transformations[
                "transformation_signature"
            ]
        )
    }

    row_indices = np.asarray(
        [
            transformation_to_index[
                transformation
            ]
            for transformation in dataframe[
                "transformation_signature"
            ]
        ],
        dtype=np.int64,
    )

    matrix = unique_matrix[
        row_indices
    ].tocsr()

    return matrix, len(
        transformations
    )


def build_condition_matrix(
    dataframe: pd.DataFrame,
    featurizer: (
        ConditionFingerprintFeaturizer
    ),
) -> sparse.csr_matrix:
    batches = []

    number_of_rows = len(dataframe)

    for start in range(
        0,
        number_of_rows,
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            number_of_rows,
        )

        batch = dataframe.iloc[
            start:end
        ]

        dense_rows = []

        for row in batch.itertuples(
            index=False
        ):
            features = (
                featurizer.transform_one(
                    solvent_labels=(
                        row.solvent_labels
                    ),
                    catalyst_labels=(
                        row.catalyst_labels
                    ),
                    reagent_labels=(
                        row.reagent_labels
                    ),
                    temperature_celsius=(
                        row.temperature_celsius
                    ),
                    reaction_time_hours=(
                        row.reaction_time_hours
                    ),
                )
            )

            dense_rows.append(
                features["combined"]
            )

        dense_batch = np.stack(
            dense_rows
        ).astype(
            np.float32,
            copy=False,
        )

        sparse_batch = sparse.csr_matrix(
            dense_batch,
            dtype=np.float32,
        )

        batches.append(
            sparse_batch
        )

        if (
            end == number_of_rows
            or end % 2560 == 0
        ):
            print(
                "Condition rows:",
                f"{end:,}/{number_of_rows:,}",
            )

    return sparse.vstack(
        batches,
        format="csr",
        dtype=np.float32,
    )


def build_metadata(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "transformation_signature",
        "reaction_center_signature",
        "condition_signature",
        "representative_reaction_id",
        "source_dataset",
        "reaction_type",
        "solvent_labels",
        "catalyst_labels",
        "reagent_labels",
        "temperature_celsius",
        "reaction_time_hours",
        "score_median",
        "score_mean",
        "score_minimum",
        "score_maximum",
        "score_standard_deviation",
        "replicate_count",
        "transformation_condition_count",
        "transformation_experiment_count",
        "transformation_best_score",
        "score_rank_percentile",
        "condition_rank",
        "relative_to_best_score",
        "is_rankable_transformation",
        "is_top_quartile_condition",
        "is_best_condition",
    ]

    missing_columns = sorted(
        set(metadata_columns)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing metadata columns: "
            + ", ".join(missing_columns)
        )

    metadata = dataframe[
        metadata_columns
    ].copy()

    metadata.insert(
        0,
        "feature_row_index",
        np.arange(
            len(metadata),
            dtype=np.int64,
        ),
    )

    return metadata


def sparse_statistics(
    matrix: sparse.csr_matrix,
) -> dict:
    total_values = (
        matrix.shape[0]
        * matrix.shape[1]
    )

    density = (
        matrix.nnz / total_values
        if total_values
        else 0.0
    )

    return {
        "shape": [
            int(matrix.shape[0]),
            int(matrix.shape[1]),
        ],
        "dtype": str(
            matrix.dtype
        ),
        "nonzero_values": int(
            matrix.nnz
        ),
        "density": float(
            density
        ),
        "storage_format": "csr",
    }


def main() -> None:
    start_time = time.perf_counter()

    dataframe = pd.read_parquet(
        INPUT_PATH
    ).reset_index(drop=True)

    print(
        "Condition pairs:",
        f"{len(dataframe):,}",
    )

    reaction_featurizer = (
        ReactionFingerprintFeaturizer()
    )

    condition_featurizer = (
        ConditionFingerprintFeaturizer()
    )

    print(
        "\nBuilding reaction features..."
    )

    reaction_matrix, unique_reactions = (
        build_reaction_matrix(
            dataframe,
            reaction_featurizer,
        )
    )

    print(
        "Reaction matrix:",
        reaction_matrix.shape,
        "nnz=",
        reaction_matrix.nnz,
    )

    print(
        "\nBuilding condition features..."
    )

    condition_matrix = (
        build_condition_matrix(
            dataframe,
            condition_featurizer,
        )
    )

    print(
        "Condition matrix:",
        condition_matrix.shape,
        "nnz=",
        condition_matrix.nnz,
    )

    if (
        reaction_matrix.shape[0]
        != len(dataframe)
    ):
        raise RuntimeError(
            "Reaction feature row mismatch."
        )

    if (
        condition_matrix.shape[0]
        != len(dataframe)
    ):
        raise RuntimeError(
            "Condition feature row mismatch."
        )

    if reaction_matrix.shape[1] != 6144:
        raise RuntimeError(
            "Unexpected reaction dimension: "
            f"{reaction_matrix.shape[1]}"
        )

    if (
        condition_matrix.shape[1]
        != condition_featurizer
        .combined_dimension
    ):
        raise RuntimeError(
            "Unexpected condition dimension: "
            f"{condition_matrix.shape[1]}"
        )

    if not np.isfinite(
        reaction_matrix.data
    ).all():
        raise RuntimeError(
            "Non-finite reaction features."
        )

    if not np.isfinite(
        condition_matrix.data
    ).all():
        raise RuntimeError(
            "Non-finite condition features."
        )

    metadata = build_metadata(
        dataframe
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSaving sparse matrices...")

    sparse.save_npz(
        REACTION_FEATURE_PATH,
        reaction_matrix,
        compressed=True,
    )

    sparse.save_npz(
        CONDITION_FEATURE_PATH,
        condition_matrix,
        compressed=True,
    )

    metadata.to_parquet(
        METADATA_PATH,
        index=False,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    manifest = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "input": {
            "file": str(
                INPUT_PATH
            ),
            "sha256": sha256_file(
                INPUT_PATH
            ),
            "rows": int(
                len(dataframe)
            ),
            "unique_transformations": int(
                unique_reactions
            ),
        },
        "reaction_features": {
            "file": str(
                REACTION_FEATURE_PATH
            ),
            "sha256": sha256_file(
                REACTION_FEATURE_PATH
            ),
            **sparse_statistics(
                reaction_matrix
            ),
            "schema": (
                reaction_featurizer.schema
            ),
        },
        "condition_features": {
            "file": str(
                CONDITION_FEATURE_PATH
            ),
            "sha256": sha256_file(
                CONDITION_FEATURE_PATH
            ),
            **sparse_statistics(
                condition_matrix
            ),
            "schema": (
                condition_featurizer.schema
            ),
        },
        "metadata": {
            "file": str(
                METADATA_PATH
            ),
            "sha256": sha256_file(
                METADATA_PATH
            ),
            "rows": int(
                len(metadata)
            ),
            "columns": (
                metadata.columns.tolist()
            ),
            "row_alignment": (
                "feature_row_index equals the "
                "row index in both sparse matrices"
            ),
        },
        "construction": {
            "batch_size": BATCH_SIZE,
            "elapsed_seconds": float(
                elapsed_seconds
            ),
            "labels_used_as_features": False,
            "split_information_used": False,
        },
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
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

    print("\nFeature build completed")
    print("-----------------------")
    print(
        "Reaction features:",
        REACTION_FEATURE_PATH,
    )
    print(
        "Condition features:",
        CONDITION_FEATURE_PATH,
    )
    print(
        "Metadata:",
        METADATA_PATH,
    )
    print(
        "Manifest:",
        MANIFEST_PATH,
    )
    print(
        "Elapsed seconds:",
        round(
            elapsed_seconds,
            2,
        ),
    )


if __name__ == "__main__":
    main()