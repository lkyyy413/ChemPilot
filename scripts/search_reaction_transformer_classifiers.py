"""Validation search for frozen RXNFP reaction classifiers."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

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


PROTOCOLS = (
    "transformation",
    "reaction_center",
)

TARGETS = (
    "solvent",
    "catalyst",
)

POOLING_METHODS = (
    "cls",
    "masked_mean",
)

REGULARIZATION_VALUES = (
    0.01,
    0.1,
    1.0,
    10.0,
)

TOP_K_VALUES = (
    1,
    3,
    5,
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

REPORT_DIRECTORY = Path(
    "reports/day5/classification"
)

CSV_PATH = (
    REPORT_DIRECTORY
    / "transformer_pooling_c_search.csv"
)

JSON_PATH = (
    REPORT_DIRECTORY
    / "transformer_pooling_c_search.json"
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


def probability_matrix(
    model,
    features,
) -> np.ndarray:
    probabilities = model.predict_proba(
        features
    )

    probabilities = np.asarray(
        probabilities,
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
    number_of_samples: int,
) -> np.ndarray:
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


def load_task(
    protocol: str,
    target: str,
    feature_matrix: np.ndarray,
    feature_metadata: pd.DataFrame,
):
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

    with (
        target_directory
        / "vocabulary.json"
    ).open(
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(
            file
        )["labels"]

    if len(samples) != targets.shape[0]:
        raise RuntimeError(
            "Sample and target row mismatch."
        )

    feature_index_by_transformation = (
        feature_metadata.set_index(
            "transformation_signature"
        )[
            "transformer_feature_row_index"
        ]
    )

    feature_indices = samples[
        "transformation_signature"
    ].map(
        feature_index_by_transformation
    )

    if feature_indices.isna().any():
        missing = samples.loc[
            feature_indices.isna(),
            "transformation_signature",
        ].tolist()

        raise RuntimeError(
            "Missing Transformer features "
            f"for transformations: {missing}"
        )

    feature_indices = (
        feature_indices.to_numpy(
            dtype=np.int64
        )
    )

    sample_features = feature_matrix[
        feature_indices
    ]

    train_mask = (
        samples["split"].eq("train")
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    valid_mask = (
        samples["split"].eq("valid")
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    train_indices = np.flatnonzero(
        train_mask
    )

    valid_indices = np.flatnonzero(
        valid_mask
    )

    if len(train_indices) == 0:
        raise RuntimeError(
            "No training samples."
        )

    if len(valid_indices) == 0:
        raise RuntimeError(
            "No validation samples."
        )

    x_train = sample_features[
        train_indices
    ]

    y_train = targets[
        train_indices
    ]

    x_valid = sample_features[
        valid_indices
    ]

    y_valid = targets[
        valid_indices
    ]

    positive_counts = np.asarray(
        y_train.sum(axis=0)
    ).reshape(-1)

    if np.any(positive_counts == 0):
        raise RuntimeError(
            "At least one retained class "
            "has no training positives."
        )

    return {
        "samples": samples,
        "vocabulary": vocabulary,
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "train_samples": len(
            train_indices
        ),
        "valid_samples": len(
            valid_indices
        ),
    }


def train_configuration(
    task: dict,
    protocol: str,
    target: str,
    pooling: str,
    regularization_c: float,
    max_iterations: int,
    n_jobs: int,
) -> dict:
    estimator = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        solver="liblinear",
        max_iter=max_iterations,
        random_state=42,
    )

    model = OneVsRestClassifier(
        estimator,
        n_jobs=n_jobs,
    )

    started = time.perf_counter()

    model.fit(
        task["x_train"],
        task["y_train"],
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    scores = probability_matrix(
        model,
        task["x_valid"],
    )

    frequency_scores = popularity_scores(
        task["y_train"],
        task["valid_samples"],
    )

    metrics = multilabel_metrics(
        task["y_valid"],
        scores,
        top_k_values=TOP_K_VALUES,
    )

    frequency_metrics = (
        multilabel_metrics(
            task["y_valid"],
            frequency_scores,
            top_k_values=(
                TOP_K_VALUES
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

    result = {
        "protocol": protocol,
        "target": target,
        "pooling": pooling,
        "regularization_c": float(
            regularization_c
        ),
        "classes": len(
            task["vocabulary"]
        ),
        "train_samples": int(
            task["train_samples"]
        ),
        "valid_samples": int(
            task["valid_samples"]
        ),
        "training_seconds": float(
            elapsed
        ),
        "maximum_iterations": int(
            max_iterations
        ),
        "maximum_iterations_used": int(
            maximum_iterations_used
        ),
        "converged": bool(
            maximum_iterations_used
            < max_iterations
        ),
        "micro_average_precision": float(
            metrics[
                "micro_average_precision"
            ]
        ),
        (
            "macro_average_precision_"
            "observed"
        ): float(
            metrics[
                "macro_average_precision_"
                "observed"
            ]
        ),
        "mean_reciprocal_rank": float(
            metrics[
                "mean_reciprocal_rank"
            ]
        ),
        "hit_rate_at_1": float(
            metrics[
                "top_k"
            ]["1"]["hit_rate"]
        ),
        "hit_rate_at_3": float(
            metrics[
                "top_k"
            ]["3"]["hit_rate"]
        ),
        "hit_rate_at_5": float(
            metrics[
                "top_k"
            ]["5"]["hit_rate"]
        ),
        "recall_at_5": float(
            metrics[
                "top_k"
            ]["5"]["recall"]
        ),
        (
            "frequency_hit_rate_at_5"
        ): float(
            frequency_metrics[
                "top_k"
            ]["5"]["hit_rate"]
        ),
    }

    return result


def select_configuration(
    task_results: list[dict],
) -> dict:
    # Primary criterion: validation micro AP.
    # Tie breakers: MRR, HitRate@5,
    # then smaller C.
    return max(
        task_results,
        key=lambda result: (
            result[
                "micro_average_precision"
            ],
            result[
                "mean_reciprocal_rank"
            ],
            result[
                "hit_rate_at_5"
            ],
            -result[
                "regularization_c"
            ],
        ),
    )


def main() -> None:
    arguments = parse_arguments()

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_archive = np.load(
        FEATURE_PATH
    )

    feature_metadata = (
        pd.read_parquet(
            METADATA_PATH
        )
    )

    if len(feature_metadata) != 381:
        raise RuntimeError(
            "Unexpected feature metadata "
            "row count."
        )

    results = []
    selected_results = []

    print(
        "Frozen RXNFP classifier search"
    )
    print("------------------------------")
    print(
        "Evaluation split: valid"
    )
    print(
        "Test labels used: False"
    )

    for protocol in PROTOCOLS:
        for target in TARGETS:
            task_results = []

            for pooling in (
                POOLING_METHODS
            ):
                feature_matrix = (
                    feature_archive[
                        pooling
                    ]
                )

                task = load_task(
                    protocol,
                    target,
                    feature_matrix,
                    feature_metadata,
                )

                for regularization_c in (
                    REGULARIZATION_VALUES
                ):
                    print(
                        "\n"
                        f"{protocol} | "
                        f"{target} | "
                        f"{pooling} | "
                        f"C={regularization_c}"
                    )

                    result = (
                        train_configuration(
                            task=task,
                            protocol=protocol,
                            target=target,
                            pooling=pooling,
                            regularization_c=(
                                regularization_c
                            ),
                            max_iterations=(
                                arguments
                                .max_iterations
                            ),
                            n_jobs=(
                                arguments.n_jobs
                            ),
                        )
                    )

                    results.append(result)
                    task_results.append(
                        result
                    )

                    print(
                        "  micro AP:",
                        round(
                            result[
                                "micro_average_precision"
                            ],
                            4,
                        ),
                    )

                    print(
                        "  MRR:",
                        round(
                            result[
                                "mean_reciprocal_rank"
                            ],
                            4,
                        ),
                    )

                    print(
                        "  HitRate@5:",
                        round(
                            result[
                                "hit_rate_at_5"
                            ],
                            4,
                        ),
                    )

                    print(
                        "  frequency HitRate@5:",
                        round(
                            result[
                                "frequency_hit_rate_at_5"
                            ],
                            4,
                        ),
                    )

            selected = (
                select_configuration(
                    task_results
                )
            )

            selected_results.append(
                selected
            )

    results_frame = pd.DataFrame(
        results
    )

    selected_frame = pd.DataFrame(
        selected_results
    )

    selected_keys = {
        (
            row["protocol"],
            row["target"],
            row["pooling"],
            row["regularization_c"],
        )
        for row in selected_results
    }

    results_frame["selected"] = [
        (
            row.protocol,
            row.target,
            row.pooling,
            row.regularization_c,
        )
        in selected_keys
        for row in (
            results_frame.itertuples()
        )
    ]

    results_frame.to_csv(
        CSV_PATH,
        index=False,
    )

    payload = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "stage": "development",
        "encoder": (
            "frozen_rxnfp_bert_pretrained"
        ),
        "feature_dimension": 256,
        "feature_preprocessing": (
            "raw frozen embeddings; "
            "no target-dependent scaling"
        ),
        "evaluation_split": "valid",
        "test_labels_used_for_selection": (
            False
        ),
        "selection_metric": (
            "micro_average_precision"
        ),
        "pooling_methods": list(
            POOLING_METHODS
        ),
        "regularization_values": list(
            REGULARIZATION_VALUES
        ),
        "selected_configurations": (
            selected_results
        ),
        "all_results": results,
    }

    with JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    print(
        "\nSelected configurations"
    )
    print("-----------------------")

    columns = [
        "protocol",
        "target",
        "pooling",
        "regularization_c",
        "micro_average_precision",
        "mean_reciprocal_rank",
        "hit_rate_at_5",
        "frequency_hit_rate_at_5",
    ]

    print(
        selected_frame[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nSaved:", CSV_PATH)
    print("Saved:", JSON_PATH)


if __name__ == "__main__":
    main()