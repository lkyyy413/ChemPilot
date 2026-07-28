from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


METADATA_PATH = Path(
    "data/processed/reactions/features/"
    "feature_metadata.parquet"
)

SPLIT_ROOT = Path(
    "data/splits/reactions"
)

OUTPUT_ROOT = Path(
    "data/processed/reactions/targets"
)

REPORT_PATH = Path(
    "reports/day4/"
    "reaction_target_manifest.json"
)

SEED = 42

PROTOCOLS = {
    "transformation": (
        "transformation_signature"
    ),
    "reaction_center": (
        "reaction_center_signature"
    ),
}

TARGETS = {
    "solvent": {
        "column": "solvent_labels",
        "minimum_train_transformations": 5,
    },
    "catalyst": {
        "column": "catalyst_labels",
        "minimum_train_transformations": 2,
    },
}


def normalize_labels(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return sorted(
            {
                str(item)
                for item in value
                if item is not None
                and str(item).strip()
            }
        )

    return [str(value)]


def union_labels(
    values,
) -> list[str]:
    labels = set()

    for value in values:
        labels.update(
            normalize_labels(value)
        )

    return sorted(labels)


def load_split_mapping(
    protocol: str,
    group_column: str,
) -> dict:
    path = (
        SPLIT_ROOT
        / protocol
        / f"seed_{SEED}"
        / "split_assignments.csv"
    )

    assignments = pd.read_csv(path)

    return dict(
        zip(
            assignments[group_column],
            assignments["split"],
        )
    )


def encode_targets(
    label_lists: list[list[str]],
    vocabulary: list[str],
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
]:
    label_to_index = {
        label: index
        for index, label in enumerate(
            vocabulary
        )
    }

    row_indices = []
    column_indices = []

    unknown_counts = np.zeros(
        len(label_lists),
        dtype=np.int32,
    )

    known_counts = np.zeros(
        len(label_lists),
        dtype=np.int32,
    )

    for row_index, labels in enumerate(
        label_lists
    ):
        for label in labels:
            if label in label_to_index:
                row_indices.append(
                    row_index
                )

                column_indices.append(
                    label_to_index[label]
                )

                known_counts[
                    row_index
                ] += 1
            else:
                unknown_counts[
                    row_index
                ] += 1

    values = np.ones(
        len(row_indices),
        dtype=np.uint8,
    )

    matrix = sparse.csr_matrix(
        (
            values,
            (
                row_indices,
                column_indices,
            ),
        ),
        shape=(
            len(label_lists),
            len(vocabulary),
        ),
        dtype=np.uint8,
    )

    return (
        matrix,
        known_counts,
        unknown_counts,
    )


def split_statistics(
    samples: pd.DataFrame,
    matrix: sparse.csr_matrix,
    split_name: str,
) -> dict:
    mask = samples[
        "split"
    ].eq(split_name).to_numpy()

    indices = np.flatnonzero(mask)

    subset = matrix[indices]

    nonempty_target = samples.loc[
        mask,
        "original_target_count",
    ].gt(0)

    any_known = samples.loc[
        mask,
        "known_target_count",
    ].gt(0)

    all_known = (
        samples.loc[
            mask,
            "unknown_target_count",
        ].eq(0)
        & nonempty_target
    )

    denominator = int(
        nonempty_target.sum()
    )

    return {
        "transformations": int(
            mask.sum()
        ),
        "nonempty_target_transformations": (
            denominator
        ),
        "transformations_with_any_known_label": int(
            any_known.sum()
        ),
        "transformations_with_all_labels_known": int(
            all_known.sum()
        ),
        "any_known_transformation_coverage": (
            float(
                any_known.sum()
                / denominator
            )
            if denominator
            else 0.0
        ),
        "all_known_transformation_coverage": (
            float(
                all_known.sum()
                / denominator
            )
            if denominator
            else 0.0
        ),
        "known_positive_assignments": int(
            subset.nnz
        ),
    }


def build_target(
    metadata: pd.DataFrame,
    protocol: str,
    group_column: str,
    target: str,
    label_column: str,
    minimum_frequency: int,
) -> dict:
    split_mapping = load_split_mapping(
        protocol,
        group_column,
    )

    # 每个可排序 transformation 只保留一个样本。
    transformation_samples = (
        metadata.loc[
            metadata[
                "is_rankable_transformation"
            ],
            [
                "feature_row_index",
                "transformation_signature",
                "reaction_center_signature",
                "reaction_type",
            ],
        ]
        .drop_duplicates(
            "transformation_signature"
        )
        .reset_index(drop=True)
    )

    transformation_samples[
        "split"
    ] = transformation_samples[
        group_column
    ].map(split_mapping)

    if transformation_samples[
        "split"
    ].isna().any():
        raise RuntimeError(
            f"Missing split assignments: "
            f"{protocol}/{target}"
        )

    # 只把高分条件中的标签视为推荐正标签。
    positive_rows = metadata.loc[
        metadata[
            "is_rankable_transformation"
        ]
        & metadata[
            "is_top_quartile_condition"
        ],
        [
            "transformation_signature",
            label_column,
        ],
    ]

    target_lists = (
        positive_rows.groupby(
            "transformation_signature"
        )[label_column]
        .apply(union_labels)
        .rename("target_labels")
        .reset_index()
    )

    samples = (
        transformation_samples.merge(
            target_lists,
            on="transformation_signature",
            how="left",
            validate="one_to_one",
        )
    )

    samples["target_labels"] = (
        samples["target_labels"].apply(
            lambda value: (
                value
                if isinstance(value, list)
                else []
            )
        )
    )

    train = samples.loc[
        samples["split"].eq("train")
    ]

    frequency = Counter()

    for labels in train[
        "target_labels"
    ]:
        frequency.update(
            set(labels)
        )

    vocabulary = sorted(
        label
        for label, count in (
            frequency.items()
        )
        if count >= minimum_frequency
    )

    matrix, known_counts, unknown_counts = (
        encode_targets(
            samples[
                "target_labels"
            ].tolist(),
            vocabulary,
        )
    )

    samples.insert(
        0,
        "target_row_index",
        np.arange(
            len(samples),
            dtype=np.int64,
        ),
    )

    samples[
        "original_target_count"
    ] = samples[
        "target_labels"
    ].apply(len)

    samples[
        "known_target_count"
    ] = known_counts

    samples[
        "unknown_target_count"
    ] = unknown_counts

    samples[
        "target_available"
    ] = samples[
        "original_target_count"
    ].gt(0)

    samples[
        "has_any_known_target"
    ] = samples[
        "known_target_count"
    ].gt(0)

    samples[
        "all_targets_known"
    ] = (
        samples[
            "unknown_target_count"
        ].eq(0)
        & samples[
            "target_available"
        ]
    )

    output_directory = (
        OUTPUT_ROOT
        / protocol
        / target
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_path = (
        output_directory
        / "targets.npz"
    )

    samples_path = (
        output_directory
        / "samples.parquet"
    )

    vocabulary_path = (
        output_directory
        / "vocabulary.json"
    )

    sparse.save_npz(
        matrix_path,
        matrix,
        compressed=True,
    )

    samples.to_parquet(
        samples_path,
        index=False,
    )

    with vocabulary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "target": target,
                "protocol": protocol,
                "minimum_train_transformations": (
                    minimum_frequency
                ),
                "number_of_classes": len(
                    vocabulary
                ),
                "labels": vocabulary,
                "training_transformation_frequency": {
                    label: int(
                        frequency[label]
                    )
                    for label in vocabulary
                },
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "minimum_train_transformations": (
            minimum_frequency
        ),
        "classes": int(
            len(vocabulary)
        ),
        "target_matrix_shape": [
            int(matrix.shape[0]),
            int(matrix.shape[1]),
        ],
        "target_positive_assignments": int(
            matrix.nnz
        ),
        "target_matrix_file": str(
            matrix_path
        ),
        "samples_file": str(
            samples_path
        ),
        "vocabulary_file": str(
            vocabulary_path
        ),
        "split_statistics": {
            split_name: split_statistics(
                samples,
                matrix,
                split_name,
            )
            for split_name in (
                "train",
                "valid",
                "test",
            )
        },
    }


def main() -> None:
    metadata = pd.read_parquet(
        METADATA_PATH
    )

    report = {
        "metadata_file": str(
            METADATA_PATH
        ),
        "seed": SEED,
        "sample_unit": (
            "One sample per rankable "
            "transformation."
        ),
        "positive_label_policy": (
            "Union of labels appearing in "
            "top-quartile condition pairs."
        ),
        "missing_condition_policy": (
            "Missing solvent or catalyst is "
            "not treated as a negative or as "
            "a no-condition class."
        ),
        "protocols": {},
    }

    for protocol, group_column in (
        PROTOCOLS.items()
    ):
        report["protocols"][
            protocol
        ] = {}

        for target, configuration in (
            TARGETS.items()
        ):
            result = build_target(
                metadata=metadata,
                protocol=protocol,
                group_column=group_column,
                target=target,
                label_column=(
                    configuration["column"]
                ),
                minimum_frequency=(
                    configuration[
                        "minimum_train_transformations"
                    ]
                ),
            )

            report["protocols"][
                protocol
            ][target] = result

            print(
                f"{protocol:16s}",
                f"{target:10s}",
                (
                    f"classes="
                    f"{result['classes']}"
                ),
                (
                    f"matrix="
                    f"{result['target_matrix_shape']}"
                ),
            )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved:", REPORT_PATH)


if __name__ == "__main__":
    main()