"""Build standardized and replicate-aggregated ORD reaction data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger

from chempilot.reactions import (
    ReactionDataset,
    ScorePolicy,
    reaction_center_elements,
    reaction_center_signature,
)


SOURCE_PATH = Path(
    "/tmp/ord-data-source/data/d9/"
    "ord_dataset-d92976309c3a48a3a64a4cf5e7048086.parquet"
)

SOURCE_REPOSITORY = (
    "https://github.com/"
    "open-reaction-database/ord-data"
)

SOURCE_REPOSITORY_COMMIT = (
    "ad4a2e12efacc9641ec14e7b2403acfd"
    "882bfe31"
)

SOURCE_LICENSE = "CC-BY-SA-4.0"

SOURCE_LFS_OID = (
    "sha256:"
    "78c17145099d29458960ffcb6cec7a898"
    "7efeae06b100004be2255ff28e54994"
)

SOURCE_SIZE_BYTES = 3_784_674

STANDARDIZED_PATH = Path(
    "data/interim/reactions/"
    "d9297630_standardized_experiments.parquet"
)

AGGREGATED_PATH = Path(
    "data/processed/reactions/"
    "d9297630_condition_pairs.parquet"
)

REPORT_PATH = Path(
    "reports/day4/"
    "reaction_dataset_build_report.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def write_parquet(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pandas(
        dataframe,
        preserve_index=False,
    )

    pq.write_table(
        table,
        path,
        compression="zstd",
    )


def flatten_unique(
    series: pd.Series,
) -> set[str]:
    labels = set()

    for values in series:
        labels.update(values)

    return labels

def assign_reaction_center_signatures(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Assign one consensus center signature per transformation."""

    dataframe = dataframe.copy()

    unique_reaction_smiles = (
        dataframe[
            "reaction_smiles_mapped"
        ]
        .dropna()
        .unique()
    )

    signature_cache = {
        reaction_smiles: (
            reaction_center_signature(
                reaction_smiles
            )
        )
        for reaction_smiles in (
            unique_reaction_smiles
        )
    }

    feature_count_cache = {
        reaction_smiles: len(
            reaction_center_elements(
                reaction_smiles
            )
        )
        for reaction_smiles in (
            unique_reaction_smiles
        )
    }

    dataframe[
        "reaction_center_signature_raw"
    ] = dataframe[
        "reaction_smiles_mapped"
    ].map(signature_cache)

    dataframe[
        "reaction_center_feature_count_raw"
    ] = dataframe[
        "reaction_smiles_mapped"
    ].map(feature_count_cache)

    preference = (
        dataframe.groupby(
            [
                "transformation_signature",
                "reaction_center_signature_raw",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            experiment_weight=(
                "replicate_count",
                "sum",
            ),
            condition_rows=(
                "condition_signature",
                "size",
            ),
            feature_count=(
                "reaction_center_feature_count_raw",
                "first",
            ),
        )
        .reset_index()
    )

    preference = preference.sort_values(
        [
            "transformation_signature",
            "experiment_weight",
            "condition_rows",
            "reaction_center_signature_raw",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )

    selected = (
        preference.drop_duplicates(
            "transformation_signature",
            keep="first",
        )
        .set_index(
            "transformation_signature"
        )
    )

    selected_signatures = selected[
        "reaction_center_signature_raw"
    ]

    selected_feature_counts = selected[
        "feature_count"
    ]

    variant_counts = (
        preference.groupby(
            "transformation_signature"
        )[
            "reaction_center_signature_raw"
        ]
        .nunique()
    )

    dataframe[
        "reaction_center_signature"
    ] = dataframe[
        "transformation_signature"
    ].map(selected_signatures)

    dataframe[
        "reaction_center_feature_count"
    ] = dataframe[
        "transformation_signature"
    ].map(selected_feature_counts)

    dataframe[
        "reaction_center_variant_count"
    ] = dataframe[
        "transformation_signature"
    ].map(variant_counts).astype(int)

    dataframe[
        "reaction_center_conflict"
    ] = (
        dataframe[
            "reaction_center_variant_count"
        ]
        > 1
    )

    return dataframe

def build_standardized_table() -> pd.DataFrame:
    dataset = ReactionDataset(
        parquet_path=SOURCE_PATH,
        source_dataset="d9297630",
        score_policy=ScorePolicy(
            measurement_type="CUSTOM",
            details="LC area percent",
            score_name="lc_area_percent",
        ),
    )

    records = [
        record.to_dict()
        for record in dataset.iter_records()
    ]

    dataframe = pd.DataFrame(records)

    if len(dataframe) != len(dataset):
        raise ValueError(
            "Not every raw reaction produced "
            "a standardized record."
        )

    if not dataframe[
        "reaction_id"
    ].is_unique:
        raise ValueError(
            "Reaction IDs are not unique."
        )

    if dataframe["score"].isna().any():
        raise ValueError(
            "Missing LC-area score detected."
        )

    if not dataframe[
        "score"
    ].between(0.0, 100.0).all():
        raise ValueError(
            "LC-area score outside [0, 100]."
        )

    return dataframe


def aggregate_replicates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    grouping_columns = [
        "transformation_signature",
        "condition_signature",
    ]

    aggregated = (
        dataframe.groupby(
            grouping_columns,
            sort=True,
            dropna=False,
        )
        .agg(
            representative_reaction_id=(
                "reaction_id",
                "first",
            ),
            source_dataset=(
                "source_dataset",
                "first",
            ),
            reaction_type=(
                "reaction_type",
                "first",
            ),
            reaction_smiles_mapped=(
                "reaction_smiles_mapped",
                "first",
            ),
            reactant_labels=(
                "reactant_labels",
                "first",
            ),
            product_labels=(
                "product_labels",
                "first",
            ),
            reagent_labels=(
                "reagent_labels",
                "first",
            ),
            solvent_labels=(
                "solvent_labels",
                "first",
            ),
            catalyst_labels=(
                "catalyst_labels",
                "first",
            ),
            reactant_smiles=(
                "reactant_smiles",
                "first",
            ),
            product_smiles=(
                "product_smiles",
                "first",
            ),
            reagent_smiles=(
                "reagent_smiles",
                "first",
            ),
            solvent_smiles=(
                "solvent_smiles",
                "first",
            ),
            catalyst_smiles=(
                "catalyst_smiles",
                "first",
            ),
            temperature_celsius=(
                "temperature_celsius",
                "first",
            ),
            reaction_time_hours=(
                "reaction_time_hours",
                "first",
            ),
            score_median=(
                "score",
                "median",
            ),
            score_mean=(
                "score",
                "mean",
            ),
            score_minimum=(
                "score",
                "min",
            ),
            score_maximum=(
                "score",
                "max",
            ),
            score_standard_deviation=(
                "score",
                "std",
            ),
            replicate_count=(
                "reaction_id",
                "size",
            ),
        )
        .reset_index()
    )

    aggregated = (
        assign_reaction_center_signatures(
            aggregated
        )
    )

    aggregated[
        "score_standard_deviation"
    ] = (
        aggregated[
            "score_standard_deviation"
        ].fillna(0.0)
    )

    grouped = aggregated.groupby(
        "transformation_signature",
        sort=False,
    )

    aggregated[
        "transformation_condition_count"
    ] = grouped[
        "condition_signature"
    ].transform("size")

    aggregated[
        "transformation_experiment_count"
    ] = grouped[
        "replicate_count"
    ].transform("sum")

    aggregated[
        "transformation_best_score"
    ] = grouped[
        "score_median"
    ].transform("max")

    aggregated[
        "score_rank_percentile"
    ] = grouped[
        "score_median"
    ].rank(
        method="average",
        pct=True,
        ascending=True,
    )

    aggregated[
        "condition_rank"
    ] = grouped[
        "score_median"
    ].rank(
        method="min",
        ascending=False,
    ).astype(int)

    best_score = aggregated[
        "transformation_best_score"
    ]

    aggregated[
        "relative_to_best_score"
    ] = np.where(
        best_score > 0.0,
        aggregated["score_median"]
        / best_score,
        0.0,
    )

    aggregated[
        "is_rankable_transformation"
    ] = (
        (
            aggregated[
                "transformation_condition_count"
            ]
            >= 2
        )
        & (best_score > 0.0)
    )

    aggregated[
        "is_top_quartile_condition"
    ] = (
        aggregated[
            "is_rankable_transformation"
        ]
        & (
            aggregated[
                "score_rank_percentile"
            ]
            >= 0.75
        )
        & (
            aggregated[
                "score_median"
            ]
            > 0.0
        )
    )

    aggregated[
        "is_best_condition"
    ] = (
        aggregated[
            "is_rankable_transformation"
        ]
        & (
            aggregated[
                "condition_rank"
            ]
            == 1
        )
        & (
            aggregated[
                "score_median"
            ]
            > 0.0
        )
    )

    return aggregated


def build_report(
    standardized: pd.DataFrame,
    aggregated: pd.DataFrame,
) -> dict:
    solvent_labels = flatten_unique(
        standardized["solvent_labels"]
    )

    catalyst_labels = flatten_unique(
        standardized["catalyst_labels"]
    )

    reagent_labels = flatten_unique(
        standardized["reagent_labels"]
    )

    transformations = aggregated[
        "transformation_signature"
    ].nunique()

    all_zero_transformations = (
        aggregated.groupby(
            "transformation_signature"
        )["score_median"]
        .max()
        .eq(0.0)
        .sum()
    )

    return {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "dataset": {
            "name": "d9297630",
            "source": (
                "Open Reaction Database"
            ),
            "source_repository": (
                SOURCE_REPOSITORY
            ),
            "source_repository_commit": (
                SOURCE_REPOSITORY_COMMIT
            ),
            "source_license": (
                SOURCE_LICENSE
            ),
            "source_article": (
                "Probing the chemical "
                "'reactome' with "
                "high-throughput "
                "experimentation data"
            ),
            "source_doi": (
                "10.1038/"
                "s41557-023-01393-w"
            ),
            "score_definition": (
                "LC area percent at 280 nm; "
                "a semi-quantitative analytical "
                "response, not isolated yield"
            ),
        },
        "files": {
            "source": str(SOURCE_PATH),
            "source_lfs_oid": (
                SOURCE_LFS_OID
            ),
            "source_size_bytes": (
                SOURCE_SIZE_BYTES
            ),
            "standardized": str(
                STANDARDIZED_PATH
            ),
            "aggregated": str(
                AGGREGATED_PATH
            ),
            "source_sha256": (
                sha256_file(SOURCE_PATH)
            ),
            "standardized_sha256": (
                sha256_file(
                    STANDARDIZED_PATH
                )
            ),
            "aggregated_sha256": (
                sha256_file(
                    AGGREGATED_PATH
                )
            ),
        },
        "standardized_experiments": {
            "rows": len(standardized),
            "unique_reaction_ids": (
                standardized[
                    "reaction_id"
                ].nunique()
            ),
            "unique_transformations": (
                standardized[
                    "transformation_signature"
                ].nunique()
            ),
            "missing_temperature": int(
                standardized[
                    "temperature_celsius"
                ].isna().sum()
            ),
            "missing_reaction_time": int(
                standardized[
                    "reaction_time_hours"
                ].isna().sum()
            ),
            "zero_score_rows": int(
                standardized[
                    "score"
                ].eq(0.0).sum()
            ),
            "unique_solvent_labels": len(
                solvent_labels
            ),
            "unique_catalyst_labels": len(
                catalyst_labels
            ),
            "unique_reagent_labels": len(
                reagent_labels
            ),
            "name_only_solvent_labels": sum(
                label.startswith("NAME:")
                for label in solvent_labels
            ),
            "name_only_catalyst_labels": sum(
                label.startswith("NAME:")
                for label in catalyst_labels
            ),
        },
        "aggregated_condition_pairs": {
            "rows": len(aggregated),
            "unique_transformations": (
                transformations
            ),
            "removed_replicate_rows": (
                len(standardized)
                - len(aggregated)
            ),
            "maximum_replicate_count": int(
                aggregated[
                    "replicate_count"
                ].max()
            ),
            "rankable_transformations": int(
                aggregated.loc[
                    aggregated[
                        "is_rankable_transformation"
                    ],
                    "transformation_signature",
                ].nunique()
            ),
            "all_zero_transformations": int(
                all_zero_transformations
            ),
            "top_quartile_pairs": int(
                aggregated[
                    "is_top_quartile_condition"
                ].sum()
            ),
            "best_condition_pairs": int(
                aggregated[
                    "is_best_condition"
                ].sum()
            ),
            "reaction_centers": {
                "unique_raw_signatures": int(
                    aggregated[
                        "reaction_center_signature_raw"
                    ].nunique()
                ),
                "unique_consensus_signatures": int(
                    aggregated[
                        "reaction_center_signature"
                    ].nunique()
                ),
                "missing_raw_signatures": int(
                    aggregated[
                        "reaction_center_signature_raw"
                    ].isna().sum()
                ),
                "missing_consensus_signatures": int(
                    aggregated[
                        "reaction_center_signature"
                    ].isna().sum()
                ),
                "conflicting_transformations": int(
                    aggregated.loc[
                        aggregated[
                            "reaction_center_conflict"
                        ],
                        "transformation_signature",
                    ].nunique()
                ),
                "condition_pairs_in_conflicts": int(
                    aggregated[
                        "reaction_center_conflict"
                    ].sum()
                ),
                "condition_pairs_changed_by_consensus": int(
                    (
                        aggregated[
                            "reaction_center_signature_raw"
                        ]
                        != aggregated[
                            "reaction_center_signature"
                        ]
                    ).sum()
                ),
                "zero_feature_condition_pairs": int(
                    aggregated[
                        "reaction_center_feature_count"
                    ].eq(0).sum()
                ),
                "maximum_variants_per_transformation": int(
                    aggregated[
                        "reaction_center_variant_count"
                    ].max()
                ),
            },
        },
        "leakage_control": {
            "primary_split_unit": (
                "transformation_signature"
            ),
            "hard_split_unit": (
                "reaction_center_signature"
            ),
            "reaction_center_policy": (
                "Weighted-majority consensus within "
                "each standardized transformation; "
                "weights are raw experiment replicate "
                "counts."
            ),
            "condition_fields": [
                "solvent",
                "catalyst",
                "reagent",
                "temperature_celsius",
                "reaction_time_hours",
            ],
            "test_labels_used_for_selection": (
                False
            ),
        },
    }


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")

    print(
        "Building standardized experiment table..."
    )

    actual_source_size = (
        SOURCE_PATH.stat().st_size
    )

    if (
        actual_source_size
        != SOURCE_SIZE_BYTES
    ):
        raise RuntimeError(
            "ORD source size mismatch: "
            f"expected {SOURCE_SIZE_BYTES}, "
            f"received {actual_source_size}."
        )

    actual_source_sha256 = (
        sha256_file(SOURCE_PATH)
    )

    expected_source_sha256 = (
        SOURCE_LFS_OID.removeprefix(
            "sha256:"
        )
    )

    if (
        actual_source_sha256
        != expected_source_sha256
    ):
        raise RuntimeError(
            "ORD source SHA256 mismatch: "
            f"expected "
            f"{expected_source_sha256}, "
            f"received "
            f"{actual_source_sha256}."
        )

    standardized = (
        build_standardized_table()
    )

    write_parquet(
        standardized,
        STANDARDIZED_PATH,
    )

    print(
        "Standardized rows:",
        len(standardized),
    )

    print(
        "Aggregating exact "
        "transformation-condition replicates..."
    )

    aggregated = aggregate_replicates(
        standardized
    )

    write_parquet(
        aggregated,
        AGGREGATED_PATH,
    )

    report = build_report(
        standardized,
        aggregated,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Aggregated rows:",
        len(aggregated),
    )

    print(
        "Unique transformations:",
        aggregated[
            "transformation_signature"
        ].nunique(),
    )

    print("Saved:", STANDARDIZED_PATH)
    print("Saved:", AGGREGATED_PATH)
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()