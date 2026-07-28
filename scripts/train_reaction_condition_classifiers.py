from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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


FEATURE_PATH = Path(
    "data/processed/reactions/features/"
    "reaction_combined.npz"
)

TARGET_ROOT = Path(
    "data/processed/reactions/targets"
)

MODEL_ROOT = Path(
    "artifacts/models/day4"
)

REPORT_ROOT = Path(
    "reports/day4/classification"
)

PROTOCOLS = (
    "transformation",
    "reaction_center",
)

TARGETS = (
    "solvent",
    "catalyst",
)

TOP_K_VALUES = (
    1,
    3,
    5,
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        choices=[
            "all",
            *PROTOCOLS,
        ],
        default="all",
    )

    parser.add_argument(
        "--target",
        choices=[
            "all",
            *TARGETS,
        ],
        default="all",
    )

    parser.add_argument(
        "--evaluation-split",
        choices=[
            "valid",
            "test",
        ],
        default="valid",
    )

    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--include-valid-in-training",
        action="store_true",
        help=(
            "Fit the final model on train "
            "and validation splits."
        ),
    )

    return parser.parse_args()


def load_vocabulary(
    path: Path,
) -> list[str]:
    with path.open(
        encoding="utf-8"
    ) as file:
        payload = json.load(file)

    return payload["labels"]


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


def top_prediction_table(
    samples: pd.DataFrame,
    y_true,
    y_score: np.ndarray,
    vocabulary: list[str],
    k: int = 5,
) -> pd.DataFrame:
    y_true = (
        y_true.toarray()
        if sparse.issparse(y_true)
        else np.asarray(y_true)
    )

    k = min(
        k,
        y_score.shape[1],
    )

    order = np.argsort(
        -y_score,
        axis=1,
    )[:, :k]

    records = []

    for row_index in range(
        len(samples)
    ):
        predicted_indices = (
            order[row_index]
        )

        true_indices = np.flatnonzero(
            y_true[row_index]
        )

        records.append(
            {
                "transformation_signature": (
                    samples.iloc[
                        row_index
                    ][
                        "transformation_signature"
                    ]
                ),
                "reaction_center_signature": (
                    samples.iloc[
                        row_index
                    ][
                        "reaction_center_signature"
                    ]
                ),
                "reaction_type": (
                    samples.iloc[
                        row_index
                    ]["reaction_type"]
                ),
                "split": (
                    samples.iloc[
                        row_index
                    ]["split"]
                ),
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
                "top_confidences": (
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
                    samples.iloc[
                        row_index
                    ][
                        "known_target_count"
                    ]
                ),
                "unknown_target_count": int(
                    samples.iloc[
                        row_index
                    ][
                        "unknown_target_count"
                    ]
                ),
                "all_targets_known": bool(
                    samples.iloc[
                        row_index
                    ][
                        "all_targets_known"
                    ]
                ),
            }
        )

    return pd.DataFrame(records)


def train_one(
    protocol: str,
    target: str,
    evaluation_split: str,
    regularization_c: float,
    max_iterations: int,
    n_jobs: int,
    reaction_features,
    include_valid_in_training: bool = False,
) -> dict:
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

    if len(samples) != targets.shape[0]:
        raise RuntimeError(
            "Sample and target row mismatch."
        )

    feature_indices = samples[
        "feature_row_index"
    ].to_numpy(
        dtype=np.int64
    )

    sample_features = (
        reaction_features[
            feature_indices
        ].tocsr()
    )

    training_splits = (
        [
            "train",
            "valid",
        ]
        if include_valid_in_training
        else [
            "train",
        ]
    )

    if (
        evaluation_split
        in training_splits
    ):
        raise ValueError(
            "Evaluation split cannot also "
            "be used for training."
        )

    train_mask = (
        samples["split"].isin(
            training_splits
        )
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    evaluation_mask = (
        samples["split"].eq(
            evaluation_split
        )
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    train_indices = np.flatnonzero(
        train_mask
    )

    evaluation_indices = (
        np.flatnonzero(
            evaluation_mask
        )
    )

    x_train = sample_features[
        train_indices
    ]

    y_train = targets[
        train_indices
    ]

    x_evaluation = sample_features[
        evaluation_indices
    ]

    y_evaluation = targets[
        evaluation_indices
    ]

    evaluation_samples = (
        samples.iloc[
            evaluation_indices
        ].reset_index(drop=True)
    )

    if len(train_indices) == 0:
        raise RuntimeError(
            "No training samples."
        )

    if len(evaluation_indices) == 0:
        raise RuntimeError(
            "No evaluation samples."
        )

    positive_counts = np.asarray(
        y_train.sum(axis=0)
    ).reshape(-1)

    if np.any(positive_counts == 0):
        raise RuntimeError(
            "At least one retained class "
            "has no training positives."
        )

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

    start = time.perf_counter()

    model.fit(
        x_train,
        y_train,
    )

    training_seconds = (
        time.perf_counter()
        - start
    )

    model_scores = probability_matrix(
        model,
        x_evaluation,
    )

    frequency_scores = popularity_scores(
        y_train,
        len(evaluation_indices),
    )

    model_metrics = multilabel_metrics(
        y_evaluation,
        model_scores,
        top_k_values=TOP_K_VALUES,
    )

    frequency_metrics = (
        multilabel_metrics(
            y_evaluation,
            frequency_scores,
            top_k_values=(
                TOP_K_VALUES
            ),
        )
    )

    convergence_iterations = [
        int(
            np.max(
                estimator.n_iter_
            )
        )
        for estimator in (
            model.estimators_
        )
    ]

    model_directory = (
        MODEL_ROOT
        / protocol
    )

    report_directory = (
        REPORT_ROOT
        / protocol
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage = (
        "final"
        if include_valid_in_training
        else "development"
    )

    model_path = (
        model_directory
        / (
            f"{target}_logistic_"
            f"{stage}.joblib"
        )
    )

    prediction_path = (
        report_directory
        / (
            f"{target}_"
            f"{stage}_"
            f"{evaluation_split}_"
            "top5.csv"
        )
    )

    result_path = (
        report_directory
        / (
            f"{target}_"
            f"{stage}_"
            f"{evaluation_split}_"
            "metrics.json"
        )
    )

    joblib.dump(
        {
            "model": model,
            "protocol": protocol,
            "target": target,
            "vocabulary": vocabulary,
            "stage": stage,
            "training_splits": (
                training_splits
            ),
            "regularization_c": (
                regularization_c
            ),
            "feature_representation": (
                "reaction_combined"
            ),
        },
        model_path,
    )

    prediction_table = (
        top_prediction_table(
            evaluation_samples,
            y_evaluation,
            model_scores,
            vocabulary,
            k=5,
        )
    )

    prediction_table.to_csv(
        prediction_path,
        index=False,
    )

    result = {
        "protocol": protocol,
        "target": target,
        "stage": stage,
        "training_splits": (
            training_splits
        ),
        "label_vocabulary_policy": (
            "Vocabulary remains fixed from "
            "the original training split."
        ),
        "evaluation_split": (
            evaluation_split
        ),
        "model": (
            "one_vs_rest_logistic_regression"
        ),
        "feature_representation": (
            "reaction_combined"
        ),
        "classes": int(
            len(vocabulary)
        ),
        "train_samples": int(
            len(train_indices)
        ),
        "evaluation_samples": int(
            len(evaluation_indices)
        ),
        "training_seconds": float(
            training_seconds
        ),
        "regularization_c": float(
            regularization_c
        ),
        "class_weight": "balanced",
        "maximum_iterations": int(
            max_iterations
        ),
        "maximum_iterations_used": int(
            max(
                convergence_iterations
            )
        ),
        "converged": bool(
            max(
                convergence_iterations
            )
            < max_iterations
        ),
        "model_metrics": model_metrics,
        "frequency_baseline_metrics": (
            frequency_metrics
        ),
        "target_coverage": {
            "evaluation_transformations": int(
                samples[
                    "split"
                ].eq(
                    evaluation_split
                ).sum()
            ),
            "evaluated_with_known_target": int(
                len(evaluation_indices)
            ),
            "all_targets_known": int(
                evaluation_samples[
                    "all_targets_known"
                ].sum()
            ),
            "contains_unknown_targets": int(
                (
                    evaluation_samples[
                        "unknown_target_count"
                    ]
                    > 0
                ).sum()
            ),
        },
        "model_file": str(
            model_path
        ),
        "prediction_file": str(
            prediction_path
        ),
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return result


def main() -> None:
    arguments = parse_arguments()

    protocols = (
        PROTOCOLS
        if arguments.protocol == "all"
        else (
            arguments.protocol,
        )
    )

    targets = (
        TARGETS
        if arguments.target == "all"
        else (
            arguments.target,
        )
    )

    reaction_features = (
        sparse.load_npz(
            FEATURE_PATH
        ).tocsr()
    )

    print("Condition classifiers")
    print("---------------------")
    print(
        "Evaluation split:",
        arguments.evaluation_split,
    )

    summary = []

    for protocol in protocols:
        for target in targets:
            print(
                f"\nTraining "
                f"{protocol} | {target}"
            )

            result = train_one(
                protocol=protocol,
                target=target,
                evaluation_split=(
                    arguments.evaluation_split
                ),
                regularization_c=(
                    arguments.c
                ),
                max_iterations=(
                    arguments.max_iterations
                ),
                n_jobs=arguments.n_jobs,
                reaction_features=(
                    reaction_features
                ),
                include_valid_in_training=(
                    arguments
                    .include_valid_in_training
                ),
            )

            top_k = result[
                "model_metrics"
            ]["top_k"]

            baseline_top_k = result[
                "frequency_baseline_metrics"
            ]["top_k"]

            print(
                "  classes:",
                result["classes"],
            )
            print(
                "  train/evaluation:",
                result["train_samples"],
                "/",
                result[
                    "evaluation_samples"
                ],
            )
            print(
                "  micro AP:",
                round(
                    result[
                        "model_metrics"
                    ][
                        "micro_average_precision"
                    ],
                    4,
                ),
            )
            print(
                "  MRR:",
                round(
                    result[
                        "model_metrics"
                    ][
                        "mean_reciprocal_rank"
                    ],
                    4,
                ),
            )
            print(
                "  model HitRate@5:",
                round(
                    top_k["5"][
                        "hit_rate"
                    ],
                    4,
                ),
            )
            print(
                "  frequency HitRate@5:",
                round(
                    baseline_top_k["5"][
                        "hit_rate"
                    ],
                    4,
                ),
            )
            print(
                "  seconds:",
                round(
                    result[
                        "training_seconds"
                    ],
                    2,
                ),
            )

            summary.append(result)

    stage = (
        "final"
        if arguments
        .include_valid_in_training
        else "development"
    )

    summary_path = (
        REPORT_ROOT
        / (
            "classification_"
            f"{stage}_"
            f"{arguments.evaluation_split}_"
            f"{arguments.protocol}_"
            f"{arguments.target}_"
            "summary.json"
        )
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()