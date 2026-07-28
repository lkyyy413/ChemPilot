from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_PATH = Path(
    "data/processed/reactions/features/"
    "feature_metadata.parquet"
)

SPLIT_ROOT = Path(
    "data/splits/reactions"
)

OUTPUT_DIRECTORY = Path(
    "reports/day4/label_space"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "label_space_audit.json"
)

SEED = 42

THRESHOLDS = (
    1,
    2,
    3,
    5,
    10,
    20,
)

PROTOCOLS = {
    "transformation": {
        "group_column": (
            "transformation_signature"
        ),
    },
    "reaction_center": {
        "group_column": (
            "reaction_center_signature"
        ),
    },
}

TARGETS = {
    "solvent": "solvent_labels",
    "catalyst": "catalyst_labels",
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

    if group_column not in (
        assignments.columns
    ):
        raise KeyError(
            f"{group_column!r} is missing "
            f"from {path}"
        )

    if "split" not in assignments:
        raise KeyError(
            f"'split' is missing from {path}"
        )

    mapping = dict(
        zip(
            assignments[group_column],
            assignments["split"],
        )
    )

    return mapping


def label_frequency_table(
    dataframe: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    label_rows = defaultdict(int)
    label_transformations = (
        defaultdict(set)
    )

    for row in dataframe.itertuples(
        index=False
    ):
        labels = normalize_labels(
            getattr(
                row,
                label_column,
            )
        )

        transformation = (
            row.transformation_signature
        )

        for label in labels:
            label_rows[label] += 1
            label_transformations[
                label
            ].add(transformation)

    records = []

    for label in sorted(label_rows):
        records.append(
            {
                "label": label,
                "condition_pair_count": int(
                    label_rows[label]
                ),
                "transformation_count": int(
                    len(
                        label_transformations[
                            label
                        ]
                    )
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "transformation_count",
                "condition_pair_count",
                "label",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def coverage_statistics(
    dataframe: pd.DataFrame,
    label_column: str,
    retained_labels: set[str],
) -> dict:
    number_of_rows = len(dataframe)

    nonempty_rows = 0
    any_known_rows = 0
    all_known_rows = 0

    total_assignments = 0
    known_assignments = 0
    unknown_labels = set()

    for value in dataframe[
        label_column
    ]:
        labels = normalize_labels(value)

        if not labels:
            continue

        nonempty_rows += 1
        total_assignments += len(labels)

        known = [
            label
            for label in labels
            if label in retained_labels
        ]

        unknown = [
            label
            for label in labels
            if label not in retained_labels
        ]

        known_assignments += len(known)
        unknown_labels.update(unknown)

        if known:
            any_known_rows += 1

        if not unknown:
            all_known_rows += 1

    return {
        "rows": int(number_of_rows),
        "nonempty_target_rows": int(
            nonempty_rows
        ),
        "rows_with_any_known_label": int(
            any_known_rows
        ),
        "rows_with_all_labels_known": int(
            all_known_rows
        ),
        "any_known_row_coverage": (
            any_known_rows
            / nonempty_rows
            if nonempty_rows
            else 0.0
        ),
        "all_known_row_coverage": (
            all_known_rows
            / nonempty_rows
            if nonempty_rows
            else 0.0
        ),
        "label_assignments": int(
            total_assignments
        ),
        "known_label_assignments": int(
            known_assignments
        ),
        "label_assignment_coverage": (
            known_assignments
            / total_assignments
            if total_assignments
            else 0.0
        ),
        "unique_unknown_labels": int(
            len(unknown_labels)
        ),
    }


def main() -> None:
    metadata = pd.read_parquet(
        METADATA_PATH
    )

    # 分类正标签只来源于同一 transformation
    # 内表现较好的候选条件。
    eligible = metadata.loc[
        metadata[
            "is_rankable_transformation"
        ]
        & metadata[
            "is_top_quartile_condition"
        ]
    ].copy()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "metadata_file": str(
            METADATA_PATH
        ),
        "seed": SEED,
        "positive_condition_policy": (
            "Rankable transformations and "
            "top-quartile condition pairs only."
        ),
        "frequency_definition": (
            "Number of unique training "
            "transformations containing the label."
        ),
        "thresholds": list(
            THRESHOLDS
        ),
        "protocols": {},
    }

    for protocol, configuration in (
        PROTOCOLS.items()
    ):
        group_column = configuration[
            "group_column"
        ]

        split_mapping = (
            load_split_mapping(
                protocol,
                group_column,
            )
        )

        protocol_data = eligible.copy()

        protocol_data["split"] = (
            protocol_data[
                group_column
            ].map(split_mapping)
        )

        if protocol_data[
            "split"
        ].isna().any():
            raise RuntimeError(
                f"Unassigned rows for {protocol}"
            )

        protocol_report = {
            "group_column": group_column,
            "eligible_condition_pairs": int(
                len(protocol_data)
            ),
            "targets": {},
        }

        for target, label_column in (
            TARGETS.items()
        ):
            train = protocol_data.loc[
                protocol_data[
                    "split"
                ].eq("train")
            ]

            frequency = (
                label_frequency_table(
                    train,
                    label_column,
                )
            )

            frequency_path = (
                OUTPUT_DIRECTORY
                / (
                    f"{protocol}_"
                    f"{target}_frequency.csv"
                )
            )

            frequency.to_csv(
                frequency_path,
                index=False,
            )

            threshold_report = {}

            for threshold in THRESHOLDS:
                retained = set(
                    frequency.loc[
                        frequency[
                            "transformation_count"
                        ].ge(threshold),
                        "label",
                    ]
                )

                split_coverage = {}

                for split_name in (
                    "train",
                    "valid",
                    "test",
                ):
                    split_data = (
                        protocol_data.loc[
                            protocol_data[
                                "split"
                            ].eq(split_name)
                        ]
                    )

                    split_coverage[
                        split_name
                    ] = coverage_statistics(
                        split_data,
                        label_column,
                        retained,
                    )

                threshold_report[
                    str(threshold)
                ] = {
                    "retained_classes": int(
                        len(retained)
                    ),
                    "coverage": (
                        split_coverage
                    ),
                }

            protocol_report[
                "targets"
            ][target] = {
                "training_unique_labels": int(
                    len(frequency)
                ),
                "frequency_file": str(
                    frequency_path
                ),
                "thresholds": (
                    threshold_report
                ),
            }

        report["protocols"][
            protocol
        ] = protocol_report

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

    print("Reaction label-space audit")
    print("--------------------------")

    for protocol, protocol_result in (
        report["protocols"].items()
    ):
        print(f"\n{protocol}")

        for target, target_result in (
            protocol_result[
                "targets"
            ].items()
        ):
            print(f"\n  {target}")
            print(
                "  training unique labels:",
                target_result[
                    "training_unique_labels"
                ],
            )

            print(
                "  threshold classes "
                "train_cov valid_cov test_cov"
            )

            for threshold in THRESHOLDS:
                result = target_result[
                    "thresholds"
                ][str(threshold)]

                coverage = result[
                    "coverage"
                ]

                print(
                    f"  {threshold:9d}",
                    f"{result['retained_classes']:7d}",
                    (
                        f"{coverage['train']['label_assignment_coverage']:.3f}"
                    ),
                    (
                        f"{coverage['valid']['label_assignment_coverage']:.3f}"
                    ),
                    (
                        f"{coverage['test']['label_assignment_coverage']:.3f}"
                    ),
                )

    print("\nSaved:", REPORT_PATH)


if __name__ == "__main__":
    main()