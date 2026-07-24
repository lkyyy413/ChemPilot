#!/usr/bin/env python
"""Train Mean and Ridge baselines on random and scaffold splits."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chempilot.data.feature_store import (
    FeatureBatch,
    MolecularFeatureStore,
)
from chempilot.evaluation.regression import (
    prediction_table,
    regression_metrics,
)


ROOT = Path(__file__).resolve().parents[1]

CACHE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "solubility_aqsoldb_features.npz"
)

REPORT_ROOT = ROOT / "reports" / "day2"
MODEL_ROOT = ROOT / "artifacts" / "models" / "day2"

ALPHA_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
]
REPRESENTATIONS = ["descriptors", "ecfp", "combined"]
SPLIT_TYPES = ["random", "scaffold"]


def split_paths(
    split_type: str,
) -> dict[str, Path]:
    if split_type == "random":
        root = ROOT / "data" / "splits" / "random" / "seed_42"

        return {
            "train": root / "train.csv",
            "valid": root / "valid.csv",
            "test": root / "test.csv",
        }

    if split_type == "scaffold":
        root = ROOT / "data" / "splits" / "scaffold"

        return {
            "train": root / "seed_42" / "train.csv",
            "valid": root / "seed_42" / "valid.csv",
            "test": root / "test.csv",
        }

    raise ValueError(f"Unsupported split type: {split_type}")


def build_ridge_pipeline(
    representation: str,
    n_descriptor_features: int = 10,
    alpha: float = 1.0,
) -> Pipeline:
    """Build preprocessing and Ridge as one leakage-safe pipeline."""

    if representation == "descriptors":
        preprocessing = StandardScaler()

    elif representation == "ecfp":
        preprocessing = "passthrough"

    elif representation == "combined":
        preprocessing = ColumnTransformer(
            transformers=[
                (
                    "descriptor_scaler",
                    StandardScaler(),
                    list(range(n_descriptor_features)),
                ),
                (
                    "ecfp",
                    "passthrough",
                    list(
                        range(
                            n_descriptor_features,
                            n_descriptor_features + 2048,
                        )
                    ),
                ),
            ],
            remainder="drop",
        )

    else:
        raise ValueError(
            f"Unsupported representation: {representation}"
        )

    model = Ridge(
        alpha=alpha,
        solver="lsqr",
        max_iter=10000,
        tol=1e-4,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def merge_batches(
    train: FeatureBatch,
    valid: FeatureBatch,
) -> FeatureBatch:
    if train.feature_names != valid.feature_names:
        raise ValueError(
            "Train and validation feature names do not match."
        )

    return FeatureBatch(
        sample_ids=np.concatenate(
            [train.sample_ids, valid.sample_ids]
        ),
        smiles=np.concatenate(
            [train.smiles, valid.smiles]
        ),
        x=np.concatenate(
            [train.x, valid.x],
            axis=0,
        ),
        y=np.concatenate(
            [train.y, valid.y]
        ),
        in_druglike_scope=np.concatenate(
            [
                train.in_druglike_scope,
                valid.in_druglike_scope,
            ]
        ),
        feature_names=train.feature_names,
    )


def evaluate_predictions(
    model,
    batch: FeatureBatch,
) -> tuple[dict, pd.DataFrame, float]:
    start = time.perf_counter()
    predictions = model.predict(batch.x)
    prediction_seconds = time.perf_counter() - start

    metrics = regression_metrics(
        batch.y,
        predictions,
    )

    metrics["prediction_seconds"] = prediction_seconds
    metrics["milliseconds_per_sample"] = (
        1000.0 * prediction_seconds / len(batch.y)
    )

    table = prediction_table(
        sample_ids=batch.sample_ids,
        smiles=batch.smiles,
        y_true=batch.y,
        y_pred=predictions,
        in_druglike_scope=batch.in_druglike_scope,
    )

    return metrics, table, prediction_seconds


def train_mean_baseline(
    split_type: str,
    train: FeatureBatch,
    valid: FeatureBatch,
    test: FeatureBatch,
) -> dict:
    """Train-only validation, then train+valid final test."""

    validation_model = DummyRegressor(strategy="mean")

    start = time.perf_counter()
    validation_model.fit(train.x[:, :1], train.y)
    validation_train_seconds = time.perf_counter() - start

    validation_metrics, _, _ = evaluate_predictions(
        validation_model,
        FeatureBatch(
            sample_ids=valid.sample_ids,
            smiles=valid.smiles,
            x=valid.x[:, :1],
            y=valid.y,
            in_druglike_scope=valid.in_druglike_scope,
            feature_names=valid.feature_names[:1],
        ),
    )

    train_valid = merge_batches(train, valid)
    final_model = DummyRegressor(strategy="mean")

    start = time.perf_counter()
    final_model.fit(
        train_valid.x[:, :1],
        train_valid.y,
    )
    final_train_seconds = time.perf_counter() - start

    test_batch = FeatureBatch(
        sample_ids=test.sample_ids,
        smiles=test.smiles,
        x=test.x[:, :1],
        y=test.y,
        in_druglike_scope=test.in_druglike_scope,
        feature_names=test.feature_names[:1],
    )

    test_metrics, test_predictions, _ = evaluate_predictions(
        final_model,
        test_batch,
    )

    prediction_path = (
        REPORT_ROOT
        / "predictions"
        / f"{split_type}_mean_test.csv"
    )
    test_predictions.to_csv(prediction_path, index=False)

    model_path = MODEL_ROOT / f"{split_type}_mean.joblib"
    joblib.dump(final_model, model_path)

    return {
        "split_type": split_type,
        "model": "mean",
        "representation": "none",
        "selected_alpha": None,
        "validation_train_seconds": validation_train_seconds,
        "final_train_seconds": final_train_seconds,
        "valid": validation_metrics,
        "test": test_metrics,
        "prediction_file": str(
            prediction_path.relative_to(ROOT)
        ),
        "model_file": str(model_path.relative_to(ROOT)),
    }


def train_ridge_baseline(
    split_type: str,
    representation: str,
    train: FeatureBatch,
    valid: FeatureBatch,
    test: FeatureBatch,
) -> dict:
    """Select alpha on validation, then refit and evaluate test."""

    search_rows = []

    for alpha in ALPHA_GRID:
        model = build_ridge_pipeline(
            representation=representation,
            alpha=alpha,
        )

        start = time.perf_counter()
        model.fit(train.x, train.y)
        train_seconds = time.perf_counter() - start

        valid_metrics, _, _ = evaluate_predictions(
            model,
            valid,
        )

        search_rows.append(
            {
                "split_type": split_type,
                "model": "ridge",
                "representation": representation,
                "alpha": alpha,
                "train_seconds": train_seconds,
                **{
                    f"valid_{key}": value
                    for key, value in valid_metrics.items()
                },
            }
        )

        logging.info(
            "%s | %s | alpha=%s | valid MAE=%.4f",
            split_type,
            representation,
            alpha,
            valid_metrics["mae"],
        )

    search_df = pd.DataFrame(search_rows).sort_values(
        ["valid_mae", "alpha"],
        ascending=[True, True],
    )

    selected_alpha = float(search_df.iloc[0]["alpha"])

    search_path = (
        REPORT_ROOT
        / "search"
        / f"{split_type}_ridge_{representation}.csv"
    )
    search_df.to_csv(search_path, index=False)

    validation_model = build_ridge_pipeline(
        representation=representation,
        alpha=selected_alpha,
    )

    start = time.perf_counter()
    validation_model.fit(train.x, train.y)
    validation_train_seconds = time.perf_counter() - start

    valid_metrics, valid_predictions, _ = (
        evaluate_predictions(
            validation_model,
            valid,
        )
    )

    valid_prediction_path = (
        REPORT_ROOT
        / "predictions"
        / f"{split_type}_ridge_{representation}_valid.csv"
    )
    valid_predictions.to_csv(
        valid_prediction_path,
        index=False,
    )

    train_valid = merge_batches(train, valid)

    final_model = build_ridge_pipeline(
        representation=representation,
        alpha=selected_alpha,
    )

    start = time.perf_counter()
    final_model.fit(train_valid.x, train_valid.y)
    final_train_seconds = time.perf_counter() - start

    test_metrics, test_predictions, _ = (
        evaluate_predictions(
            final_model,
            test,
        )
    )

    test_prediction_path = (
        REPORT_ROOT
        / "predictions"
        / f"{split_type}_ridge_{representation}_test.csv"
    )
    test_predictions.to_csv(
        test_prediction_path,
        index=False,
    )

    model_path = (
        MODEL_ROOT
        / f"{split_type}_ridge_{representation}.joblib"
    )
    joblib.dump(final_model, model_path)

    return {
        "split_type": split_type,
        "model": "ridge",
        "representation": representation,
        "selected_alpha": selected_alpha,
        "validation_train_seconds": (
            validation_train_seconds
        ),
        "final_train_seconds": final_train_seconds,
        "valid": valid_metrics,
        "test": test_metrics,
        "search_file": str(search_path.relative_to(ROOT)),
        "valid_prediction_file": str(
            valid_prediction_path.relative_to(ROOT)
        ),
        "test_prediction_file": str(
            test_prediction_path.relative_to(ROOT)
        ),
        "model_file": str(model_path.relative_to(ROOT)),
    }


def flatten_result(result: dict) -> dict:
    row = {
        "split_type": result["split_type"],
        "model": result["model"],
        "representation": result["representation"],
        "selected_alpha": result["selected_alpha"],
        "validation_train_seconds": result[
            "validation_train_seconds"
        ],
        "final_train_seconds": result[
            "final_train_seconds"
        ],
    }

    for evaluation_split in ["valid", "test"]:
        for key, value in result[evaluation_split].items():
            row[f"{evaluation_split}_{key}"] = value

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["random", "scaffold", "all"],
        default="all",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "predictions").mkdir(
        parents=True,
        exist_ok=True,
    )
    (REPORT_ROOT / "search").mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    store = MolecularFeatureStore(CACHE_PATH)

    selected_splits = (
        SPLIT_TYPES
        if args.split == "all"
        else [args.split]
    )

    full_results = []

    for split_type in selected_splits:
        paths = split_paths(split_type)

        descriptor_train = store.load_split(
            paths["train"],
            representation="descriptors",
        )
        descriptor_valid = store.load_split(
            paths["valid"],
            representation="descriptors",
        )
        descriptor_test = store.load_split(
            paths["test"],
            representation="descriptors",
        )

        full_results.append(
            train_mean_baseline(
                split_type=split_type,
                train=descriptor_train,
                valid=descriptor_valid,
                test=descriptor_test,
            )
        )

        for representation in REPRESENTATIONS:
            train = store.load_split(
                paths["train"],
                representation=representation,
            )
            valid = store.load_split(
                paths["valid"],
                representation=representation,
            )
            test = store.load_split(
                paths["test"],
                representation=representation,
            )

            full_results.append(
                train_ridge_baseline(
                    split_type=split_type,
                    representation=representation,
                    train=train,
                    valid=valid,
                    test=test,
                )
            )

    result_rows = [
        flatten_result(result)
        for result in full_results
    ]

    result_df = pd.DataFrame(result_rows).sort_values(
        ["split_type", "test_mae"]
    )

    result_csv_path = (
        REPORT_ROOT / "linear_baseline_results.csv"
    )
    result_df.to_csv(result_csv_path, index=False)

    result_json_path = (
        REPORT_ROOT / "linear_baseline_results.json"
    )

    payload = {
        "alpha_grid": ALPHA_GRID,
        "selection_metric": "validation MAE",
        "refit_for_test": "train plus validation",
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "results": full_results,
    }

    with result_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nFinal baseline results:")
    columns = [
        "split_type",
        "model",
        "representation",
        "selected_alpha",
        "valid_mae",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_spearman",
    ]
    print(
        result_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    logging.info(
        "Saved results to %s",
        result_csv_path,
    )


if __name__ == "__main__":
    main()