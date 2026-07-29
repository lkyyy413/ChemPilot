"""Final evaluation of validation-selected frozen RXNFP models."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.multiclass import (
    OneVsRestClassifier,
)

from chempilot.evaluation.multilabel import (
    multilabel_metrics,
)


TARGET_ROOT = Path(
    "data/processed/reactions/targets"
)

FEATURE_PATH = Path(
    "data/processed/reactions/"
    "features/day5/"
    "rxnfp_reaction_embeddings.npz"
)

METADATA_PATH = Path(
    "data/processed/reactions/"
    "features/day5/"
    "rxnfp_reaction_metadata.parquet"
)

SELECTION_PATH = Path(
    "reports/day5/classification/"
    "transformer_pooling_c_search.json"
)

MODEL_ROOT = Path(
    "artifacts/models/day5/"
    "frozen_final"
)

REPORT_ROOT = Path(
    "reports/day5/frozen_final"
)

SUMMARY_JSON_PATH = (
    REPORT_ROOT
    / "final_test_summary.json"
)

SUMMARY_CSV_PATH = (
    REPORT_ROOT
    / "final_test_results.csv"
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
    )

    return parser.parse_args()


def load_vocabulary(path):
    with path.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)[
            "labels"
        ]


def probability_matrix(
    model,
    features,
):
    probabilities = np.asarray(
        model.predict_proba(
            features
        ),
        dtype=np.float64,
    )

    if probabilities.ndim != 2:
        raise RuntimeError(
            "Expected a two-dimensional "
            "probability matrix."
        )

    return probabilities


def popularity_scores(
    y_train,
    number_of_samples,
):
    prevalence = np.asarray(
        y_train.mean(axis=0)
    ).reshape(-1)

    return np.broadcast_to(
        prevalence,
        (
            number_of_samples,
            len(prevalence),
        ),
    ).copy()


def top_prediction_table(
    samples,
    y_true,
    y_score,
    vocabulary,
    k=5,
):
    y_true = (
        y_true.toarray()
        if sparse.issparse(y_true)
        else np.asarray(y_true)
    )

    order = np.argsort(
        -y_score,
        axis=1,
    )[:, :k]

    records = []

    for row_index in range(
        len(samples)
    ):
        sample = samples.iloc[
            row_index
        ]

        true_indices = np.flatnonzero(
            y_true[row_index]
        )

        predicted_indices = order[
            row_index
        ]

        records.append(
            {
                "transformation_signature": (
                    sample[
                        "transformation_signature"
                    ]
                ),
                "reaction_center_signature": (
                    sample[
                        "reaction_center_signature"
                    ]
                ),
                "reaction_type": (
                    sample["reaction_type"]
                ),
                "split": "test",
                "true_known_labels": (
                    json.dumps(
                        [
                            vocabulary[index]
                            for index in (
                                true_indices
                            )
                        ],
                        ensure_ascii=False,
                    )
                ),
                "top_labels": (
                    json.dumps(
                        [
                            vocabulary[index]
                            for index in (
                                predicted_indices
                            )
                        ],
                        ensure_ascii=False,
                    )
                ),
                "top_scores": (
                    json.dumps(
                        [
                            float(
                                y_score[
                                    row_index,
                                    index,
                                ]
                            )
                            for index in (
                                predicted_indices
                            )
                        ]
                    )
                ),
                "known_target_count": int(
                    sample[
                        "known_target_count"
                    ]
                ),
                "unknown_target_count": int(
                    sample[
                        "unknown_target_count"
                    ]
                ),
                "all_targets_known": bool(
                    sample[
                        "all_targets_known"
                    ]
                ),
            }
        )

    return pd.DataFrame(records)


def evaluate_task(
    configuration,
    feature_archive,
    metadata,
    arguments,
):
    protocol = configuration[
        "protocol"
    ]

    target = configuration[
        "target"
    ]

    pooling = configuration[
        "pooling"
    ]

    regularization_c = (
        configuration[
            "regularization_c"
        ]
    )

    target_directory = (
        TARGET_ROOT
        / protocol
        / target
    )

    samples = pd.read_parquet(
        target_directory
        / "samples.parquet"
    )

    targets = sparse.load_npz(
        target_directory
        / "targets.npz"
    ).tocsr()

    vocabulary = load_vocabulary(
        target_directory
        / "vocabulary.json"
    )

    feature_index = (
        metadata.set_index(
            "transformation_signature"
        )[
            "transformer_feature_row_index"
        ]
    )

    sample_feature_indices = (
        samples[
            "transformation_signature"
        ].map(feature_index)
    )

    if sample_feature_indices.isna().any():
        raise RuntimeError(
            "Missing frozen RXNFP feature."
        )

    sample_feature_indices = (
        sample_feature_indices.to_numpy(
            dtype=np.int64
        )
    )

    sample_features = (
        feature_archive[
            pooling
        ][sample_feature_indices]
    )

    train_valid_mask = (
        samples["split"].isin(
            [
                "train",
                "valid",
            ]
        )
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    test_mask = (
        samples["split"].eq("test")
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    train_valid_indices = (
        np.flatnonzero(
            train_valid_mask
        )
    )

    test_indices = np.flatnonzero(
        test_mask
    )

    x_train_valid = sample_features[
        train_valid_indices
    ]

    y_train_valid = targets[
        train_valid_indices
    ]

    x_test = sample_features[
        test_indices
    ]

    y_test = targets[
        test_indices
    ]

    positive_counts = np.asarray(
        y_train_valid.sum(axis=0)
    ).reshape(-1)

    if np.any(
        positive_counts == 0
    ):
        raise RuntimeError(
            "A vocabulary class has no "
            "train+valid positives."
        )

    estimator = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        solver="liblinear",
        max_iter=(
            arguments.max_iterations
        ),
        random_state=42,
    )

    model = OneVsRestClassifier(
        estimator,
        n_jobs=arguments.n_jobs,
    )

    started = time.perf_counter()

    model.fit(
        x_train_valid,
        y_train_valid,
    )

    training_seconds = (
        time.perf_counter()
        - started
    )

    # The configuration and model are fixed
    # before this single test prediction.
    test_scores = probability_matrix(
        model,
        x_test,
    )

    test_metrics = multilabel_metrics(
        y_test,
        test_scores,
        top_k_values=(
            1,
            3,
            5,
        ),
    )

    frequency_scores = (
        popularity_scores(
            y_train_valid,
            len(test_indices),
        )
    )

    frequency_metrics = (
        multilabel_metrics(
            y_test,
            frequency_scores,
            top_k_values=(
                1,
                3,
                5,
            ),
        )
    )

    maximum_iterations_used = max(
        int(
            np.max(
                estimator.n_iter_
            )
        )
        for estimator in (
            model.estimators_
        )
    )

    model_directory = (
        MODEL_ROOT / protocol
    )

    report_directory = (
        REPORT_ROOT / protocol
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_directory
        / f"{target}_final.joblib"
    )

    metrics_path = (
        report_directory
        / f"{target}_test_metrics.json"
    )

    predictions_path = (
        report_directory
        / f"{target}_test_top5.csv"
    )

    joblib.dump(
        {
            "model": model,
            "protocol": protocol,
            "target": target,
            "vocabulary": vocabulary,
            "stage": "final",
            "training_splits": [
                "train",
                "valid",
            ],
            "pooling": pooling,
            "regularization_c": (
                regularization_c
            ),
            "feature_representation": (
                "frozen_rxnfp_"
                f"{pooling}"
            ),
        },
        model_path,
    )

    test_samples = (
        samples.iloc[
            test_indices
        ].reset_index(drop=True)
    )

    top_prediction_table(
        test_samples,
        y_test,
        test_scores,
        vocabulary,
    ).to_csv(
        predictions_path,
        index=False,
    )

    result = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol": protocol,
        "target": target,
        "stage": "final",
        "model": (
            "frozen_rxnfp_plus_"
            "one_vs_rest_logistic"
        ),
        "pooling": pooling,
        "regularization_c": (
            regularization_c
        ),
        "selection_source": str(
            SELECTION_PATH
        ),
        "test_labels_used_for_selection": (
            False
        ),
        "test_evaluations": 1,
        "classes": len(vocabulary),
        "train_valid_samples": len(
            train_valid_indices
        ),
        "complete_test_samples": int(
            samples["split"].eq(
                "test"
            ).sum()
        ),
        "test_evaluated_samples": len(
            test_indices
        ),
        "all_targets_known": int(
            test_samples[
                "all_targets_known"
            ].sum()
        ),
        "contains_unknown_targets": int(
            (
                test_samples[
                    "unknown_target_count"
                ]
                > 0
            ).sum()
        ),
        "test_metrics": test_metrics,
        "frequency_baseline_metrics": (
            frequency_metrics
        ),
        "training_seconds": (
            training_seconds
        ),
        "maximum_iterations": (
            arguments.max_iterations
        ),
        "maximum_iterations_used": (
            maximum_iterations_used
        ),
        "converged": bool(
            maximum_iterations_used
            < arguments.max_iterations
        ),
        "model_path": str(model_path),
        "prediction_path": str(
            predictions_path
        ),
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    print(
        "\n"
        + f"{protocol} | {target}"
    )
    print(
        "  pooling:",
        pooling,
    )
    print(
        "  C:",
        regularization_c,
    )
    print(
        "  test micro AP:",
        round(
            test_metrics[
                "micro_average_precision"
            ],
            4,
        ),
    )
    print(
        "  test MRR:",
        round(
            test_metrics[
                "mean_reciprocal_rank"
            ],
            4,
        ),
    )
    print(
        "  test HitRate@5:",
        round(
            test_metrics[
                "top_k"
            ]["5"]["hit_rate"],
            4,
        ),
    )

    return result


def main():
    arguments = parse_arguments()

    if SUMMARY_JSON_PATH.exists():
        raise FileExistsError(
            "Frozen RXNFP final summary "
            "already exists; refusing to "
            "repeat test evaluation."
        )

    with SELECTION_PATH.open(
        encoding="utf-8"
    ) as file:
        selection = json.load(file)

    assert (
        selection[
            "test_labels_used_for_selection"
        ]
        is False
    )

    selected = selection[
        "selected_configurations"
    ]

    if len(selected) != 4:
        raise RuntimeError(
            "Expected four selected "
            "configurations."
        )

    feature_archive = np.load(
        FEATURE_PATH
    )

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Frozen RXNFP final evaluation"
    )
    print("-----------------------------")
    print(
        "Training splits: train+valid"
    )
    print(
        "Evaluation split: test"
    )
    print(
        "Test evaluations per task: 1"
    )

    results = []

    for configuration in selected:
        results.append(
            evaluate_task(
                configuration,
                feature_archive,
                metadata,
                arguments,
            )
        )

    summary = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "selection_source": str(
            SELECTION_PATH
        ),
        "test_labels_used_for_selection": (
            False
        ),
        "test_evaluations_per_task": 1,
        "results": results,
    }

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    rows = []

    for result in results:
        metrics = result[
            "test_metrics"
        ]

        rows.append(
            {
                "protocol": (
                    result["protocol"]
                ),
                "target": (
                    result["target"]
                ),
                "pooling": (
                    result["pooling"]
                ),
                "regularization_c": (
                    result[
                        "regularization_c"
                    ]
                ),
                "classes": (
                    result["classes"]
                ),
                "train_valid_samples": (
                    result[
                        "train_valid_samples"
                    ]
                ),
                "test_evaluated_samples": (
                    result[
                        "test_evaluated_samples"
                    ]
                ),
                "test_micro_ap": (
                    metrics[
                        "micro_average_precision"
                    ]
                ),
                "test_macro_ap_observed": (
                    metrics[
                        "macro_average_precision_"
                        "observed"
                    ]
                ),
                "test_mrr": (
                    metrics[
                        "mean_reciprocal_rank"
                    ]
                ),
                "test_hit_rate_at_1": (
                    metrics[
                        "top_k"
                    ]["1"]["hit_rate"]
                ),
                "test_hit_rate_at_3": (
                    metrics[
                        "top_k"
                    ]["3"]["hit_rate"]
                ),
                "test_hit_rate_at_5": (
                    metrics[
                        "top_k"
                    ]["5"]["hit_rate"]
                ),
                "test_recall_at_5": (
                    metrics[
                        "top_k"
                    ]["5"]["recall"]
                ),
                (
                    "frequency_hit_rate_at_5"
                ): (
                    result[
                        "frequency_baseline_metrics"
                    ][
                        "top_k"
                    ]["5"]["hit_rate"]
                ),
                "all_targets_known": (
                    result[
                        "all_targets_known"
                    ]
                ),
                (
                    "contains_unknown_targets"
                ): (
                    result[
                        "contains_unknown_targets"
                    ]
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        SUMMARY_CSV_PATH,
        index=False,
    )

    print("\nSaved:", SUMMARY_JSON_PATH)
    print("Saved:", SUMMARY_CSV_PATH)


if __name__ == "__main__":
    main()