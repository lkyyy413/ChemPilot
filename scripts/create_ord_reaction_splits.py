from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/reactions/"
    "d9297630_condition_pairs.parquet"
)

OUTPUT_ROOT = Path(
    "data/splits/reactions"
)

REPORT_PATH = Path(
    "reports/day4/reaction_split_audit.json"
)

SEED = 42
NUMBER_OF_CANDIDATES = 20_000

SPLIT_NAMES = (
    "train",
    "valid",
    "test",
)

TARGET_PROPORTIONS = np.asarray(
    [0.70, 0.15, 0.15],
    dtype=np.float64,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def make_unit_table(
    dataframe: pd.DataFrame,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformation_table = (
        dataframe.groupby(
            "transformation_signature",
            as_index=False,
        )
        .agg(
            reaction_center_signature=(
                "reaction_center_signature",
                "first",
            ),
            reaction_type=(
                "reaction_type",
                "first",
            ),
            condition_pair_count=(
                "condition_signature",
                "size",
            ),
        )
    )

    if group_column == "transformation_signature":
        unit_table = transformation_table.copy()

        unit_table["split_unit"] = unit_table[
            "transformation_signature"
        ]

        unit_table["transformation_count"] = 1

    elif group_column == "reaction_center_signature":
        unit_table = (
            transformation_table.groupby(
                "reaction_center_signature",
                as_index=False,
            )
            .agg(
                condition_pair_count=(
                    "condition_pair_count",
                    "sum",
                ),
                transformation_count=(
                    "transformation_signature",
                    "size",
                ),
            )
        )

        unit_table["split_unit"] = unit_table[
            "reaction_center_signature"
        ]

    else:
        raise ValueError(
            f"Unsupported group column: {group_column}"
        )

    unit_table = unit_table.sort_values(
        "split_unit"
    ).reset_index(drop=True)

    return unit_table, transformation_table


def candidate_score(
    unit_table: pd.DataFrame,
    assignments: np.ndarray,
) -> float:
    row_counts = np.asarray(
        [
            unit_table.loc[
                assignments == split_index,
                "condition_pair_count",
            ].sum()
            for split_index in range(3)
        ],
        dtype=np.float64,
    )

    transformation_counts = np.asarray(
        [
            unit_table.loc[
                assignments == split_index,
                "transformation_count",
            ].sum()
            for split_index in range(3)
        ],
        dtype=np.float64,
    )

    unit_counts = np.asarray(
        [
            np.count_nonzero(
                assignments == split_index
            )
            for split_index in range(3)
        ],
        dtype=np.float64,
    )

    row_proportions = (
        row_counts / row_counts.sum()
    )

    transformation_proportions = (
        transformation_counts
        / transformation_counts.sum()
    )

    unit_proportions = (
        unit_counts / unit_counts.sum()
    )

    row_error = np.abs(
        row_proportions - TARGET_PROPORTIONS
    ).sum()

    transformation_error = np.abs(
        transformation_proportions
        - TARGET_PROPORTIONS
    ).sum()

    unit_error = np.abs(
        unit_proportions
        - TARGET_PROPORTIONS
    ).sum()

    return float(
        5.0 * row_error
        + 2.0 * transformation_error
        + unit_error
    )


def search_assignments(
    unit_table: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, float]:
    generator = np.random.default_rng(seed)

    number_of_units = len(unit_table)

    train_end = round(
        TARGET_PROPORTIONS[0]
        * number_of_units
    )

    valid_end = train_end + round(
        TARGET_PROPORTIONS[1]
        * number_of_units
    )

    best_assignments = None
    best_score = float("inf")

    for _ in range(NUMBER_OF_CANDIDATES):
        permutation = generator.permutation(
            number_of_units
        )

        assignments = np.full(
            number_of_units,
            2,
            dtype=np.int8,
        )

        assignments[
            permutation[:train_end]
        ] = 0

        assignments[
            permutation[
                train_end:valid_end
            ]
        ] = 1

        score = candidate_score(
            unit_table,
            assignments,
        )

        if score < best_score:
            best_score = score
            best_assignments = (
                assignments.copy()
            )

    if best_assignments is None:
        raise RuntimeError(
            "No split assignment was generated."
        )

    return best_assignments, best_score


def overlap_count(
    left: pd.Series,
    right: pd.Series,
) -> int:
    return len(
        set(left.dropna())
        & set(right.dropna())
    )


def create_protocol(
    dataframe: pd.DataFrame,
    protocol: str,
    group_column: str,
) -> dict:
    unit_table, _ = make_unit_table(
        dataframe,
        group_column,
    )

    assignments, search_score = (
        search_assignments(
            unit_table,
            seed=SEED,
        )
    )

    unit_to_split = dict(
        zip(
            unit_table["split_unit"],
            [
                SPLIT_NAMES[index]
                for index in assignments
            ],
        )
    )

    result = dataframe.copy()

    result["split"] = result[
        group_column
    ].map(unit_to_split)

    if result["split"].isna().any():
        raise RuntimeError(
            f"Unassigned rows in {protocol}."
        )

    protocol_directory = (
        OUTPUT_ROOT
        / protocol
        / f"seed_{SEED}"
    )

    protocol_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_tables = {}
    split_report = {}

    for split_name in SPLIT_NAMES:
        split_table = (
            result.loc[
                result["split"].eq(
                    split_name
                )
            ]
            .drop(columns="split")
            .reset_index(drop=True)
        )

        output_path = (
            protocol_directory
            / f"{split_name}.parquet"
        )

        split_table.to_parquet(
            output_path,
            index=False,
        )

        split_tables[split_name] = (
            split_table
        )

        split_report[split_name] = {
            "condition_pairs": int(
                len(split_table)
            ),
            "condition_pair_proportion": (
                len(split_table)
                / len(result)
            ),
            "transformations": int(
                split_table[
                    "transformation_signature"
                ].nunique()
            ),
            "reaction_centers": int(
                split_table[
                    "reaction_center_signature"
                ].nunique()
            ),
            "reaction_types": int(
                split_table[
                    "reaction_type"
                ].nunique()
            ),
            "represented_experiments": int(
                split_table[
                    "replicate_count"
                ].sum()
            ),
            "rankable_transformations": int(
                split_table.loc[
                    split_table[
                        "is_rankable_transformation"
                    ],
                    "transformation_signature",
                ].nunique()
            ),
            "all_zero_transformations": int(
                split_table.groupby(
                    "transformation_signature"
                )["score_median"]
                .max()
                .eq(0.0)
                .sum()
            ),
            "file": str(output_path),
            "sha256": sha256_file(
                output_path
            ),
        }

    overlap_report = {}

    for left_name, right_name in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        left = split_tables[left_name]
        right = split_tables[right_name]

        overlap_report[
            f"{left_name}_{right_name}"
        ] = {
            "transformation_overlap": (
                overlap_count(
                    left[
                        "transformation_signature"
                    ],
                    right[
                        "transformation_signature"
                    ],
                )
            ),
            "reaction_center_overlap": (
                overlap_count(
                    left[
                        "reaction_center_signature"
                    ],
                    right[
                        "reaction_center_signature"
                    ],
                )
            ),
            "condition_signature_overlap": (
                overlap_count(
                    left[
                        "condition_signature"
                    ],
                    right[
                        "condition_signature"
                    ],
                )
            ),
        }

    # 两种协议都必须禁止 transformation 泄漏。
    for values in overlap_report.values():
        assert (
            values[
                "transformation_overlap"
            ]
            == 0
        )

    # 困难协议额外禁止反应中心泄漏。
    if protocol == "reaction_center":
        for values in (
            overlap_report.values()
        ):
            assert (
                values[
                    "reaction_center_overlap"
                ]
                == 0
            )

    manifest_path = (
        protocol_directory
        / "split_assignments.csv"
    )

    unit_table = unit_table.copy()

    unit_table["split"] = [
        SPLIT_NAMES[index]
        for index in assignments
    ]

    unit_table.to_csv(
        manifest_path,
        index=False,
    )

    return {
        "group_column": group_column,
        "seed": SEED,
        "candidate_assignments_evaluated": (
            NUMBER_OF_CANDIDATES
        ),
        "search_score": search_score,
        "splits": split_report,
        "overlap_audit": overlap_report,
        "assignment_file": str(
            manifest_path
        ),
        "assignment_sha256": (
            sha256_file(manifest_path)
        ),
    }


def main() -> None:
    dataframe = pd.read_parquet(
        INPUT_PATH
    )

    required_columns = {
        "transformation_signature",
        "reaction_center_signature",
        "condition_signature",
        "reaction_type",
        "replicate_count",
        "score_median",
        "is_rankable_transformation",
    }

    missing_columns = sorted(
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    report = {
        "input_file": str(INPUT_PATH),
        "input_sha256": sha256_file(
            INPUT_PATH
        ),
        "seed": SEED,
        "target_proportions": {
            "train": 0.70,
            "valid": 0.15,
            "test": 0.15,
        },
        "selection_policy": (
            "Candidate assignments are selected "
            "using only group sizes and row counts. "
            "Reaction scores and condition ranks are "
            "not used to select the split."
        ),
        "protocols": {},
    }

    report["protocols"][
        "transformation"
    ] = create_protocol(
        dataframe=dataframe,
        protocol="transformation",
        group_column=(
            "transformation_signature"
        ),
    )

    report["protocols"][
        "reaction_center"
    ] = create_protocol(
        dataframe=dataframe,
        protocol="reaction_center",
        group_column=(
            "reaction_center_signature"
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

    print("Reaction split generation")
    print("-------------------------")

    for protocol, result in (
        report["protocols"].items()
    ):
        print(f"\n{protocol}")

        for split_name, values in (
            result["splits"].items()
        ):
            print(
                f"  {split_name:5s}",
                f"pairs={values['condition_pairs']:6,d}",
                (
                    "proportion="
                    f"{values['condition_pair_proportion']:.2%}"
                ),
                (
                    "transformations="
                    f"{values['transformations']:3,d}"
                ),
                (
                    "centers="
                    f"{values['reaction_centers']:3,d}"
                ),
            )

        print(
            "  search score:",
            round(
                result["search_score"],
                6,
            ),
        )

    print("\nSaved:", REPORT_PATH)
    print("Reaction split audit passed.")


if __name__ == "__main__":
    main()