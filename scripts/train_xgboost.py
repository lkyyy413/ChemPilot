#!/usr/bin/env python
"""Train XGBoost baselines for AqSolDB."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

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
        "max_depth": 3,
        "learning_rate": 0.10,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 8,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
    },
    {
        "max_depth": 6,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
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


def build_search_model(
    parameters: dict,
    device: str,
) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=2000,
        early_stopping_rounds=75,
        tree_method="hist",
        device=device,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **parameters,
    )


def build_final_model(
    parameters: dict,
    n_estimators: int,
    device: str,
) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=n_estimators,
        tree_method="hist",
        device=device,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **parameters,
    )


def evaluate(
    model: XGBRegressor,
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


def extract_gain_importance(
    model: XGBRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    score = model.get_booster().get_score(
        importance_type="gain"
    )

    importance = np.zeros(
        len(feature_names),
        dtype=np.float64,
    )

    for key, value in score.items():
        feature_index = int(key.removeprefix("f"))
        importance[feature_index] = float(value)

    total = importance.sum()
    if total > 0:
        importance = importance / total

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "gain_importance": importance,
        }
    )

    result["feature_group"] = np.where(
        result["feature"].str.startswith("ecfp_"),
        "ecfp",
        "descriptor",
    )

    return result.sort_values(
        "gain_importance",
        ascending=False,
    ).reset_index(drop=True)


def train_one(
    split_type: str,
    representation: str,
    train: FeatureBatch,
    valid: FeatureBatch,
    test: FeatureBatch,
    device: str,
) -> dict:
    search_rows = []

    for candidate_id, parameters in enumerate(
        PARAMETER_CANDIDATES,
        start=1,
    ):
        model = build_search_model(parameters, device)

        start = time.perf_counter()
        model.fit(
            train.x,
            train.y,
            eval_set=[(valid.x, valid.y)],
            verbose=False,
        )
        train_seconds = time.perf_counter() - start

        valid_metrics, _ = evaluate(model, valid)

        best_iteration = int(model.best_iteration)
        selected_rounds = best_iteration + 1

        row = {
            "candidate_id": candidate_id,
            "split_type": split_type,
            "representation": representation,
            **parameters,
            "best_iteration": best_iteration,
            "selected_rounds": selected_rounds,
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
            "rounds=%d | valid MAE=%.4f",
            split_type,
            representation,
            candidate_id,
            selected_rounds,
            valid_metrics["mae"],
        )

    search_df = pd.DataFrame(search_rows).sort_values(
        ["valid_mae", "candidate_id"]
    )

    search_path = (
        REPORT_ROOT / "search"
        / f"{split_type}_xgboost_{representation}.csv"
    )
    search_df.to_csv(search_path, index=False)

    best_row = search_df.iloc[0]
    best_id = int(best_row["candidate_id"])
    best_parameters = PARAMETER_CANDIDATES[best_id - 1]
    selected_rounds = int(best_row["selected_rounds"])

    validation_model = build_search_model(
        best_parameters,
        device,
    )

    start = time.perf_counter()
    validation_model.fit(
        train.x,
        train.y,
        eval_set=[(valid.x, valid.y)],
        verbose=False,
    )
    validation_train_seconds = time.perf_counter() - start

    valid_metrics, valid_predictions = evaluate(
        validation_model,
        valid,
    )

    valid_path = (
        REPORT_ROOT / "predictions"
        / f"{split_type}_xgboost_"
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

    final_model = build_final_model(
        parameters=best_parameters,
        n_estimators=selected_rounds,
        device=device,
    )

    start = time.perf_counter()
    final_model.fit(x_train_valid, y_train_valid)
    final_train_seconds = time.perf_counter() - start

    test_metrics, test_predictions = evaluate(
        final_model,
        test,
    )

    test_path = (
        REPORT_ROOT / "predictions"
        / f"{split_type}_xgboost_"
          f"{representation}_test.csv"
    )
    test_predictions.to_csv(test_path, index=False)

    model_path = (
        MODEL_ROOT
        / f"{split_type}_xgboost_"
          f"{representation}.joblib"
    )
    joblib.dump(final_model, model_path)

    importance = extract_gain_importance(
        final_model,
        train.feature_names,
    )

    importance_path = (
        REPORT_ROOT / "feature_importance"
        / f"{split_type}_xgboost_"
          f"{representation}.csv"
    )
    importance.head(100).to_csv(
       importance_path,
       index=False,
    )

    failure_path = (
        REPORT_ROOT / "failures"
        / f"{split_type}_xgboost_"
          f"{representation}_top50.csv"
    )
    test_predictions.head(50).to_csv(
        failure_path,
        index=False,
    )

    return {
        "split_type": split_type,
        "model": "xgboost",
        "representation": representation,
        "selected_candidate": best_id,
        "selected_parameters": best_parameters,
        "selected_rounds": selected_rounds,
        "device": device,
        "validation_train_seconds": (
            validation_train_seconds
        ),
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
        "selected_rounds": result["selected_rounds"],
        "selected_parameters": json.dumps(
            result["selected_parameters"]
        ),
        "device": result["device"],
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
    parser.add_argument(
        "--device",
        default="cpu",
        help="Use cpu or cuda:0.",
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
                "Starting %s | %s | device=%s",
                split_type,
                representation,
                args.device,
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
                train_one(
                    split_type,
                    representation,
                    train,
                    valid,
                    test,
                    args.device,
                )
            )

    result_df = pd.DataFrame(
        [flatten_result(result) for result in results]
    ).sort_values(["split_type", "test_mae"])

    csv_path = REPORT_ROOT / "xgboost_results.csv"
    result_df.to_csv(csv_path, index=False)

    json_path = REPORT_ROOT / "xgboost_results.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selection_metric": "validation MAE",
                "early_stopping_rounds": 75,
                "software_versions": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "xgboost": xgboost.__version__,
                },
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    columns = [
        "split_type",
        "representation",
        "selected_candidate",
        "selected_rounds",
        "valid_mae",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_spearman",
        "final_train_seconds",
    ]

    print("\nXGBoost results:")
    print(
        result_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()