#!/usr/bin/env python
"""Train Random Forest baselines for AqSolDB."""

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
from sklearn.ensemble import RandomForestRegressor

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
    ROOT / "data" / "processed"
    / "solubility_aqsoldb_features.npz"
)
REPORT_ROOT = ROOT / "reports" / "day2"
MODEL_ROOT = ROOT / "artifacts" / "models" / "day2"

REPRESENTATIONS = ["descriptors", "ecfp", "combined"]
SPLIT_TYPES = ["random", "scaffold"]

PARAMETER_CANDIDATES = [
    {
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    {
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.5,
    },
    {
        "max_depth": None,
        "min_samples_leaf": 4,
        "max_features": 0.5,
    },
    {
        "max_depth": 20,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": 0.5,
    },
    {
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": 0.5,
    },
]


def get_split_paths(split_type: str) -> dict[str, Path]:
    if split_type == "random":
        root = (
            ROOT / "data" / "splits"
            / "random" / "seed_42"
        )
        return {
            "train": root / "train.csv",
            "valid": root / "valid.csv",
            "test": root / "test.csv",
        }

    root = ROOT / "data" / "splits" / "scaffold"
    return {
        "train": root / "seed_42" / "train.csv",
        "valid": root / "seed_42" / "valid.csv",
        "test": root / "test.csv",
    }


def build_model(parameters: dict) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        max_features=parameters["max_features"],
        random_state=42,
        n_jobs=-1,
    )


def evaluate(
    model: RandomForestRegressor,
    batch: FeatureBatch,
) -> tuple[dict, pd.DataFrame]:
    start = time.perf_counter()
    y_pred = model.predict(batch.x)
    prediction_seconds = time.perf_counter() - start

    metrics = regression_metrics(batch.y, y_pred)
    metrics["prediction_seconds"] = prediction_seconds
    metrics["milliseconds_per_sample"] = (
        1000.0 * prediction_seconds / len(batch.y)
    )

    predictions = prediction_table(
        sample_ids=batch.sample_ids,
        smiles=batch.smiles,
        y_true=batch.y,
        y_pred=y_pred,
        in_druglike_scope=batch.in_druglike_scope,
    )

    return metrics, predictions


def train_one_experiment(
    split_type: str,
    representation: str,
    train: FeatureBatch,
    valid: FeatureBatch,
    test: FeatureBatch,
) -> dict:
    search_rows = []

    for candidate_id, parameters in enumerate(
        PARAMETER_CANDIDATES,
        start=1,
    ):
        model = build_model(parameters)

        start = time.perf_counter()
        model.fit(train.x, train.y)
        train_seconds = time.perf_counter() - start

        valid_metrics, _ = evaluate(model, valid)

        row = {
            "candidate_id": candidate_id,
            "split_type": split_type,
            "representation": representation,
            "n_estimators": 300,
            "max_depth": parameters["max_depth"],
            "min_samples_leaf": (
                parameters["min_samples_leaf"]
            ),
            "max_features": parameters["max_features"],
            "train_seconds": train_seconds,
        }

        row.update(
            {
                f"valid_{key}": value
                for key, value in valid_metrics.items()
            }
        )
        search_rows.append(row)

        logging.info(
            "%s | %s | candidate=%d | "
            "depth=%s | leaf=%s | features=%s | MAE=%.4f",
            split_type,
            representation,
            candidate_id,
            parameters["max_depth"],
            parameters["min_samples_leaf"],
            parameters["max_features"],
            valid_metrics["mae"],
        )

    search_df = pd.DataFrame(search_rows).sort_values(
        ["valid_mae", "candidate_id"]
    )

    search_path = (
        REPORT_ROOT / "search"
        / f"{split_type}_random_forest_{representation}.csv"
    )
    search_df.to_csv(search_path, index=False)

    best_id = int(search_df.iloc[0]["candidate_id"])
    best_parameters = PARAMETER_CANDIDATES[best_id - 1]

    validation_model = build_model(best_parameters)

    start = time.perf_counter()
    validation_model.fit(train.x, train.y)
    validation_train_seconds = time.perf_counter() - start

    valid_metrics, valid_predictions = evaluate(
        validation_model,
        valid,
    )

    valid_path = (
        REPORT_ROOT / "predictions"
        / f"{split_type}_random_forest_"
          f"{representation}_valid.csv"
    )
    valid_predictions.to_csv(valid_path, index=False)

    x_train_valid = np.concatenate(
        [train.x, valid.x],
        axis=0,
    )
    y_train_valid = np.concatenate(
        [train.y, valid.y],
    )

    final_model = build_model(best_parameters)

    start = time.perf_counter()
    final_model.fit(x_train_valid, y_train_valid)
    final_train_seconds = time.perf_counter() - start

    test_metrics, test_predictions = evaluate(
        final_model,
        test,
    )

    test_path = (
        REPORT_ROOT / "predictions"
        / f"{split_type}_random_forest_"
          f"{representation}_test.csv"
    )
    test_predictions.to_csv(test_path, index=False)

    model_path = (
        MODEL_ROOT
        / f"{split_type}_random_forest_"
          f"{representation}.joblib"
    )
    joblib.dump(final_model, model_path)

    importance_df = pd.DataFrame(
        {
            "feature": train.feature_names,
            "importance": final_model.feature_importances_,
        }
    )

    importance_df["feature_group"] = np.where(
        importance_df["feature"].str.startswith("ecfp_"),
        "ecfp",
        "descriptor",
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_path = (
        REPORT_ROOT
        / "feature_importance"
        / f"{split_type}_random_forest_"
          f"{representation}.csv"
    )
    importance_df.head(100).to_csv(
       importance_path,
       index=False,
    )

    failure_path = (
        REPORT_ROOT
        / "failures"
        / f"{split_type}_random_forest_"
          f"{representation}_top50.csv"
    )
    test_predictions.head(50).to_csv(
        failure_path,
        index=False,
    )

    return {
        "split_type": split_type,
        "model": "random_forest",
        "representation": representation,
        "selected_candidate": best_id,
        "selected_parameters": best_parameters,
        "validation_train_seconds": validation_train_seconds,
        "final_train_seconds": final_train_seconds,
        "valid": valid_metrics,
        "test": test_metrics,
        "search_file": str(search_path.relative_to(ROOT)),
        "importance_file": str(
            importance_path.relative_to(ROOT)
        ),
        "failure_file": str(
            failure_path.relative_to(ROOT)
        ),
        "model_file": str(model_path.relative_to(ROOT)),
    }


def flatten_result(result: dict) -> dict:
    row = {
        "split_type": result["split_type"],
        "model": result["model"],
        "representation": result["representation"],
        "selected_candidate": result[
            "selected_candidate"
        ],
        "selected_parameters": json.dumps(
            result["selected_parameters"]
        ),
        "validation_train_seconds": result[
            "validation_train_seconds"
        ],
        "final_train_seconds": result[
            "final_train_seconds"
        ],
    }

    for split_name in ["valid", "test"]:
        for key, value in result[split_name].items():
            row[f"{split_name}_{key}"] = value

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["random", "scaffold", "all"],
        default="all",
    )
    parser.add_argument(
        "--representation",
        choices=[
            "descriptors",
            "ecfp",
            "combined",
            "all",
        ],
        default="all",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    for directory in [
        REPORT_ROOT / "search",
        REPORT_ROOT / "predictions",
        REPORT_ROOT / "feature_importance",
        REPORT_ROOT / "failures",
        MODEL_ROOT,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    split_types = (
        SPLIT_TYPES
        if args.split == "all"
        else [args.split]
    )
    representations = (
        REPRESENTATIONS
        if args.representation == "all"
        else [args.representation]
    )

    store = MolecularFeatureStore(CACHE_PATH)
    results = []

    for split_type in split_types:
        paths = get_split_paths(split_type)

        for representation in representations:
            logging.info(
                "Starting %s | %s",
                split_type,
                representation,
            )

            train = store.load_split(
                paths["train"],
                representation,
            )
            valid = store.load_split(
                paths["valid"],
                representation,
            )
            test = store.load_split(
                paths["test"],
                representation,
            )

            results.append(
                train_one_experiment(
                    split_type,
                    representation,
                    train,
                    valid,
                    test,
                )
            )

    result_df = pd.DataFrame(
        [flatten_result(result) for result in results]
    ).sort_values(["split_type", "test_mae"])

    csv_path = (
        REPORT_ROOT / "random_forest_results.csv"
    )
    result_df.to_csv(csv_path, index=False)

    json_path = (
        REPORT_ROOT / "random_forest_results.json"
    )
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_metric": "validation MAE",
                "n_estimators": 300,
                "software_versions": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                },
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    columns = [
        "split_type",
        "model",
        "representation",
        "selected_candidate",
        "valid_mae",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_spearman",
        "final_train_seconds",
    ]

    print("\nRandom Forest results:")
    print(
        result_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()