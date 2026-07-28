"""Analyze reaction-condition applicability domains."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from chempilot.evaluation.applicability import (
    leave_one_out_nearest_tanimoto,
    nearest_binary_tanimoto,
)


FEATURE_PATH = Path(
    "data/processed/reactions/features/"
    "reaction_combined.npz"
)

TARGET_ROOT = Path(
    "data/processed/reactions/targets"
)

PREDICTION_ROOT = Path(
    "reports/day4/classification"
)

OUTPUT_ROOT = Path(
    "reports/day4/applicability"
)

AD_PERCENTILE = 5.0

# reaction_combined =
# reactant 2048 + product 2048 + difference 2048.
# AD uses only the first two binary blocks.
BINARY_STRUCTURE_DIMENSION = 4096

TASKS = [
    ("transformation", "solvent"),
    ("transformation", "catalyst"),
    ("reaction_center", "solvent"),
    ("reaction_center", "catalyst"),
]


def parse_json_list(
    value,
) -> list:
    """Parse a JSON-encoded list from a CSV field."""

    if isinstance(value, list):
        return value

    if value is None or pd.isna(value):
        return []

    parsed = json.loads(value)

    if not isinstance(parsed, list):
        raise ValueError(
            "Expected a JSON list."
        )

    return parsed


def ranking_statistics(
    dataframe: pd.DataFrame,
) -> dict:
    """Calculate ranking metrics from Top-5 predictions."""

    if dataframe.empty:
        return {
            "samples": 0,
            "hit_rate_at_1": None,
            "hit_rate_at_3": None,
            "hit_rate_at_5": None,
            "recall_at_5": None,
            "mean_reciprocal_rank": None,
            "mean_top1_confidence": None,
        }

    hits_at_1 = []
    hits_at_3 = []
    hits_at_5 = []
    recalls_at_5 = []
    reciprocal_ranks = []
    top1_confidences = []

    for row in dataframe.itertuples(
        index=False
    ):
        true_labels = set(
            parse_json_list(
                row.true_known_labels
            )
        )

        top_labels = parse_json_list(
            row.top_labels
        )

        confidences = parse_json_list(
            row.top_confidences
        )

        top1 = set(top_labels[:1])
        top3 = set(top_labels[:3])
        top5 = set(top_labels[:5])

        hits_at_1.append(
            bool(true_labels & top1)
        )

        hits_at_3.append(
            bool(true_labels & top3)
        )

        hits_at_5.append(
            bool(true_labels & top5)
        )

        recalls_at_5.append(
            (
                len(true_labels & top5)
                / len(true_labels)
            )
            if true_labels
            else 0.0
        )

        reciprocal_rank = 0.0

        for rank, label in enumerate(
            top_labels,
            start=1,
        ):
            if label in true_labels:
                reciprocal_rank = (
                    1.0 / rank
                )
                break

        reciprocal_ranks.append(
            reciprocal_rank
        )

        top1_confidences.append(
            float(confidences[0])
            if confidences
            else np.nan
        )

    return {
        "samples": int(len(dataframe)),
        "hit_rate_at_1": float(
            np.mean(hits_at_1)
        ),
        "hit_rate_at_3": float(
            np.mean(hits_at_3)
        ),
        "hit_rate_at_5": float(
            np.mean(hits_at_5)
        ),
        "recall_at_5": float(
            np.mean(recalls_at_5)
        ),
        "mean_reciprocal_rank": float(
            np.mean(reciprocal_ranks)
        ),
        "mean_top1_confidence": float(
            np.nanmean(top1_confidences)
        ),
    }


def analyze_task(
    reaction_features: sparse.csr_matrix,
    protocol: str,
    target: str,
) -> dict:
    """Analyze one protocol and prediction target."""

    target_directory = (
        TARGET_ROOT / protocol / target
    )

    samples_path = (
        target_directory / "samples.parquet"
    )

    prediction_path = (
        PREDICTION_ROOT
        / protocol
        / f"{target}_final_test_top5.csv"
    )

    samples = pd.read_parquet(
        samples_path
    )

    predictions = pd.read_csv(
        prediction_path
    )

    training_mask = (
        samples["split"].isin(
            ["train", "valid"]
        )
        & samples["has_any_known_target"]
    )

    test_mask = samples["split"].eq(
        "test"
    )

    training_samples = (
        samples.loc[
            training_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_samples = (
        samples.loc[
            test_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    training_feature_indices = (
        training_samples[
            "feature_row_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    test_feature_indices = (
        test_samples[
            "feature_row_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    training_features = (
        reaction_features[
            training_feature_indices,
            :BINARY_STRUCTURE_DIMENSION,
        ]
    )

    test_features = (
        reaction_features[
            test_feature_indices,
            :BINARY_STRUCTURE_DIMENSION,
        ]
    )

    training_loo_similarities = (
        leave_one_out_nearest_tanimoto(
            training_features
        )
    )

    ad_threshold = float(
        np.percentile(
            training_loo_similarities,
            AD_PERCENTILE,
        )
    )

    (
        nearest_similarities,
        nearest_indices,
    ) = nearest_binary_tanimoto(
        test_features,
        training_features,
    )

    nearest_training_samples = (
        training_samples.iloc[
            nearest_indices
        ]
        .reset_index(drop=True)
    )

    test_samples[
        "nearest_train_transformation"
    ] = nearest_training_samples[
        "transformation_signature"
    ].to_numpy()

    test_samples[
        "nearest_train_reaction_center"
    ] = nearest_training_samples[
        "reaction_center_signature"
    ].to_numpy()

    test_samples[
        "nearest_train_reaction_type"
    ] = nearest_training_samples[
        "reaction_type"
    ].to_numpy()

    test_samples[
        "nearest_similarity"
    ] = nearest_similarities

    test_samples[
        "ad_threshold"
    ] = ad_threshold

    test_samples["in_domain"] = (
        test_samples[
            "nearest_similarity"
        ]
        >= ad_threshold
    )

    test_samples[
        "same_reaction_type_as_nearest"
    ] = (
        test_samples[
            "reaction_type"
        ].to_numpy()
        == test_samples[
            "nearest_train_reaction_type"
        ].to_numpy()
    )

    prediction_columns = [
        "transformation_signature",
        "true_known_labels",
        "top_labels",
        "top_confidences",
    ]

    predictions = predictions[
        prediction_columns
    ].copy()

    predictions[
        "has_final_prediction"
    ] = True

    result = test_samples.merge(
        predictions,
        on="transformation_signature",
        how="left",
        validate="one_to_one",
    )

    result[
        "has_final_prediction"
    ] = result[
        "has_final_prediction"
    ].fillna(False)

    evaluated = result.loc[
        result["has_final_prediction"]
    ].copy()

    in_domain_evaluated = evaluated.loc[
        evaluated["in_domain"]
    ].copy()

    out_of_domain_evaluated = (
        evaluated.loc[
            ~evaluated["in_domain"]
        ].copy()
    )

    output_path = (
        OUTPUT_ROOT
        / f"{protocol}_{target}_test_ad.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    summary = {
        "protocol": protocol,
        "target": target,
        "threshold_definition": (
            "5th percentile of final "
            "training-set leave-one-out "
            "nearest-neighbor binary "
            "Tanimoto similarity"
        ),
        "ad_percentile": AD_PERCENTILE,
        "binary_structure_dimension": (
            BINARY_STRUCTURE_DIMENSION
        ),
        "final_training_samples": int(
            len(training_samples)
        ),
        "complete_test_samples": int(
            len(test_samples)
        ),
        "evaluated_test_samples": int(
            len(evaluated)
        ),
        "ad_threshold": ad_threshold,
        "training_loo_similarity": {
            "minimum": float(
                training_loo_similarities.min()
            ),
            "median": float(
                np.median(
                    training_loo_similarities
                )
            ),
            "mean": float(
                training_loo_similarities.mean()
            ),
            "maximum": float(
                training_loo_similarities.max()
            ),
        },
        "complete_test_applicability": {
            "in_domain_samples": int(
                result["in_domain"].sum()
            ),
            "out_of_domain_samples": int(
                (~result["in_domain"]).sum()
            ),
            "in_domain_rate": float(
                result["in_domain"].mean()
            ),
            "mean_nearest_similarity": float(
                result[
                    "nearest_similarity"
                ].mean()
            ),
            "median_nearest_similarity": float(
                result[
                    "nearest_similarity"
                ].median()
            ),
            "same_reaction_type_rate": float(
                result[
                    "same_reaction_type_as_nearest"
                ].mean()
            ),
        },
        "evaluated_test_metrics": (
            ranking_statistics(
                evaluated
            )
        ),
        "in_domain_test_metrics": (
            ranking_statistics(
                in_domain_evaluated
            )
        ),
        "out_of_domain_test_metrics": (
            ranking_statistics(
                out_of_domain_evaluated
            )
        ),
        "unknown_target_audit": {
            "evaluated_with_unknown_targets": (
                int(
                    evaluated[
                        "unknown_target_count"
                    ].gt(0).sum()
                )
            ),
            "in_domain_with_unknown_targets": (
                int(
                    in_domain_evaluated[
                        "unknown_target_count"
                    ].gt(0).sum()
                )
            ),
            "out_of_domain_with_unknown_targets": (
                int(
                    out_of_domain_evaluated[
                        "unknown_target_count"
                    ].gt(0).sum()
                )
            ),
        },
        "output_file": str(output_path),
    }

    print(
        f"\n{protocol} | {target}"
    )
    print(
        "  final training samples:",
        len(training_samples),
    )
    print(
        "  complete/evaluated test:",
        len(test_samples),
        "/",
        len(evaluated),
    )
    print(
        "  AD threshold:",
        round(ad_threshold, 4),
    )
    print(
        "  in-domain complete test:",
        (
            f"{result['in_domain'].mean():.2%}"
        ),
    )
    print(
        "  in-domain evaluated:",
        len(in_domain_evaluated),
    )
    print(
        "  out-of-domain evaluated:",
        len(out_of_domain_evaluated),
    )

    return summary


def main() -> None:
    """Run applicability analysis for all tasks."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    reaction_features = sparse.load_npz(
        FEATURE_PATH
    ).tocsr()

    if (
        reaction_features.shape[1]
        < BINARY_STRUCTURE_DIMENSION
    ):
        raise ValueError(
            "Reaction feature cache has fewer "
            "than 4096 structural columns."
        )

    summaries = {}

    for protocol, target in TASKS:
        key = f"{protocol}|{target}"

        summaries[key] = analyze_task(
            reaction_features,
            protocol,
            target,
        )

    report = {
        "method": {
            "representation": (
                "Concatenated binary Morgan "
                "fingerprints of reactants "
                "and products"
            ),
            "similarity": (
                "Binary Tanimoto"
            ),
            "threshold_source": (
                "Final train+validation subset; "
                "test labels were not used"
            ),
            "threshold_percentile": (
                AD_PERCENTILE
            ),
            "confidence_note": (
                "Logistic-regression scores are "
                "ranking confidence proxies and "
                "are not calibrated probabilities."
            ),
        },
        "tasks": summaries,
    }

    report_path = (
        OUTPUT_ROOT
        / "applicability_summary.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nSaved:",
        report_path,
    )


if __name__ == "__main__":
    main()