#!/usr/bin/env python
"""Generate final Day 2 tables, figures, and failure analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chempilot.evaluation.regression import (
    regression_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "day2"
FIGURE_ROOT = REPORT_ROOT / "figures"


def load_results() -> pd.DataFrame:
    paths = [
        REPORT_ROOT / "linear_baseline_results.csv",
        REPORT_ROOT / "random_forest_results.csv",
        REPORT_ROOT / "xgboost_results.csv",
    ]

    frames = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file: {path}"
            )

        frame = pd.read_csv(path)
        frames.append(frame)

    results = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    results["representation"] = (
        results["representation"].fillna("none")
    )

    results["model_label"] = (
        results["model"]
        .replace(
            {
                "mean": "Mean",
                "ridge": "Ridge",
                "random_forest": "Random Forest",
                "xgboost": "XGBoost",
            }
        )
        + " + "
        + results["representation"].replace(
            {
                "none": "none",
                "descriptors": "descriptors",
                "ecfp": "ECFP",
                "combined": "combined",
            }
        )
    )

    results.loc[
        results["model"] == "mean",
        "model_label",
    ] = "Mean"

    return results


def save_final_table(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split_type",
        "model",
        "representation",
        "valid_mae",
        "valid_rmse",
        "valid_r2",
        "valid_spearman",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_spearman",
        "final_train_seconds",
    ]

    table = (
        results[columns]
        .sort_values(["split_type", "test_mae"])
        .reset_index(drop=True)
    )

    table.to_csv(
        REPORT_ROOT / "baseline_results.csv",
        index=False,
    )

    return table


def plot_baseline_mae(results: pd.DataFrame) -> None:
    model_order = [
        "Mean",
        "Ridge + descriptors",
        "Ridge + ECFP",
        "Ridge + combined",
        "Random Forest + descriptors",
        "Random Forest + ECFP",
        "Random Forest + combined",
        "XGBoost + descriptors",
        "XGBoost + ECFP",
        "XGBoost + combined",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        sharey=True,
    )

    colors = {
        "Mean": "#9CA3AF",
        "Ridge": "#60A5FA",
        "Random Forest": "#34D399",
        "XGBoost": "#F59E0B",
    }

    for axis, split_type in zip(
        axes,
        ["random", "scaffold"],
    ):
        subset = (
            results[results["split_type"] == split_type]
            .set_index("model_label")
            .reindex(model_order)
            .dropna(subset=["test_mae"])
            .reset_index()
        )

        bar_colors = [
            colors[
                "Mean"
                if label == "Mean"
                else label.split(" + ")[0]
            ]
            for label in subset["model_label"]
        ]

        positions = np.arange(len(subset))

        axis.barh(
            positions,
            subset["test_mae"],
            color=bar_colors,
            edgecolor="white",
        )

        axis.set_yticks(positions)
        axis.set_yticklabels(subset["model_label"])
        axis.invert_yaxis()
        axis.set_xlabel("Test MAE (lower is better)")
        axis.set_title(f"{split_type.capitalize()} split")
        axis.grid(axis="x", alpha=0.25)

        for position, value in zip(
            positions,
            subset["test_mae"],
        ):
            axis.text(
                value + 0.02,
                position,
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )

    fig.suptitle(
        "AqSolDB baseline comparison",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()

    fig.savefig(
        FIGURE_ROOT / "baseline_test_mae.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_xgboost_importance() -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        sharex=True,
    )

    for axis, split_type in zip(
        axes,
        ["random", "scaffold"],
    ):
        path = (
            REPORT_ROOT
            / "feature_importance"
            / f"{split_type}_xgboost_descriptors.csv"
        )

        importance = pd.read_csv(path).head(10)
        importance = importance.sort_values(
            "gain_importance"
        )

        axis.barh(
            importance["feature"],
            importance["gain_importance"],
            color="#2563EB",
        )
        axis.set_title(f"{split_type.capitalize()} split")
        axis.set_xlabel("Normalized gain importance")
        axis.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "XGBoost descriptor importance",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()

    fig.savefig(
        FIGURE_ROOT / "xgboost_descriptor_importance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_best_predictions() -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
    )

    for axis, split_type in zip(
        axes,
        ["random", "scaffold"],
    ):
        path = (
            REPORT_ROOT
            / "predictions"
            / f"{split_type}_xgboost_combined_test.csv"
        )
        predictions = pd.read_csv(path)

        metrics = regression_metrics(
            predictions["y_true"].to_numpy(),
            predictions["y_pred"].to_numpy(),
        )

        axis.scatter(
            predictions["y_true"],
            predictions["y_pred"],
            s=10,
            alpha=0.35,
            color="#2563EB",
            edgecolors="none",
        )

        minimum = min(
            predictions["y_true"].min(),
            predictions["y_pred"].min(),
        )
        maximum = max(
            predictions["y_true"].max(),
            predictions["y_pred"].max(),
        )

        axis.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
            color="#DC2626",
            linewidth=1.2,
        )

        axis.set_xlabel("Observed LogS")
        axis.set_ylabel("Predicted LogS")
        axis.set_title(
            f"{split_type.capitalize()} split\n"
            f"MAE={metrics['mae']:.3f}, "
            f"R²={metrics['r2']:.3f}"
        )
        axis.grid(alpha=0.2)

    fig.suptitle(
        "Best baseline: XGBoost with combined features",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    fig.savefig(
        FIGURE_ROOT / "xgboost_combined_predictions.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def analyze_failures() -> tuple[pd.DataFrame, pd.DataFrame]:
    top_failure_frames = []
    group_rows = []

    label_bins = [
        -np.inf,
        -8.0,
        -4.0,
        0.0,
        np.inf,
    ]
    label_names = [
        "LogS <= -8",
        "-8 < LogS <= -4",
        "-4 < LogS <= 0",
        "LogS > 0",
    ]

    for split_type in ["random", "scaffold"]:
        path = (
            REPORT_ROOT
            / "predictions"
            / f"{split_type}_xgboost_combined_test.csv"
        )

        predictions = pd.read_csv(path)
        predictions = predictions.sort_values(
            "absolute_error",
            ascending=False,
        )

        top_failures = predictions.head(20).copy()
        top_failures.insert(0, "split_type", split_type)
        top_failure_frames.append(top_failures)

        predictions["scope"] = np.where(
            predictions["in_druglike_scope"],
            "druglike",
            "outside_druglike_scope",
        )

        predictions["label_band"] = pd.cut(
            predictions["y_true"],
            bins=label_bins,
            labels=label_names,
            include_lowest=True,
        )

        for group_type, column in [
            ("scope", "scope"),
            ("label_band", "label_band"),
        ]:
            for group_name, group in predictions.groupby(
                column,
                observed=True,
            ):
                metrics = regression_metrics(
                    group["y_true"].to_numpy(),
                    group["y_pred"].to_numpy(),
                )

                group_rows.append(
                    {
                        "split_type": split_type,
                        "group_type": group_type,
                        "group": str(group_name),
                        **metrics,
                    }
                )

    top_failure_table = pd.concat(
        top_failure_frames,
        ignore_index=True,
    )
    group_table = pd.DataFrame(group_rows)

    top_failure_table.to_csv(
        REPORT_ROOT / "top_failure_samples.csv",
        index=False,
    )
    group_table.to_csv(
        REPORT_ROOT / "subgroup_error_analysis.csv",
        index=False,
    )

    return top_failure_table, group_table


def markdown_table(
    frame: pd.DataFrame,
    float_digits: int = 4,
) -> str:
    headers = [str(column) for column in frame.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in frame.iterrows():
        values = []

        for value in row:
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    values.append("NA")
                else:
                    values.append(f"{value:.{float_digits}f}")
            else:
                values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_report(
    results: pd.DataFrame,
    final_table: pd.DataFrame,
    failures: pd.DataFrame,
    subgroup: pd.DataFrame,
) -> None:
    best_rows = (
        results.sort_values("test_mae")
        .groupby("split_type", as_index=False)
        .first()
    )

    compact_results = final_table[
        [
            "split_type",
            "model",
            "representation",
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_spearman",
        ]
    ].copy()

    compact_results["model"] = compact_results[
        "model"
    ].replace(
        {
            "mean": "Mean",
            "ridge": "Ridge",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
        }
    )

    compact_results["test_spearman"] = (
        compact_results["test_spearman"].fillna(
            "undefined"
        )
    )

    failure_preview = failures[
        [
            "split_type",
            "sample_id",
            "y_true",
            "y_pred",
            "absolute_error",
            "in_druglike_scope",
        ]
    ].groupby("split_type").head(5)

    report = f"""# Day 2: AqSolDB property-prediction baselines

## Objective

The objective of Day 2 was to establish reproducible baselines for aqueous solubility regression using global RDKit descriptors, ECFP4 fingerprints, and their concatenation. Mean prediction, Ridge regression, Random Forest, and XGBoost were compared under fixed random and scaffold protocols. Hyperparameters were selected using validation MAE, after which the selected configuration was refitted on train plus validation data and evaluated on the held-out test set.

## Experimental controls

All models within one split protocol used the same sample IDs, labels, canonical SMILES, and cached molecular features. Feature generation did not use labels. Scaling for Ridge descriptors was fitted inside the training pipeline. Test labels were not used to select model configurations. Random and scaffold scores are reported separately because the two protocols do not contain the same test samples and therefore should not be interpreted as a controlled one-variable comparison.

## Final results

{markdown_table(compact_results)}

The strongest model under both protocols was XGBoost with combined RDKit descriptors and ECFP features. Random-split test MAE was {best_rows.loc[best_rows['split_type'] == 'random', 'test_mae'].iloc[0]:.4f}; scaffold-split test MAE was {best_rows.loc[best_rows['split_type'] == 'scaffold', 'test_mae'].iloc[0]:.4f}. Combining global descriptors with local substructure fingerprints consistently outperformed either representation alone.

## Representation findings

RDKit descriptors alone formed a strong and computationally inexpensive baseline. ECFP alone improved on the linear descriptor model under the random protocol but transferred poorly to the scaffold protocol. The combined representation performed best because it captures both transferable physicochemical trends and local structural patterns. The poor scaffold performance of ECFP-only tree models shows that substructure memorization does not automatically provide reliable extrapolation to new chemical frameworks.

## Feature importance

LogP was the most important descriptor for both Random Forest and XGBoost, followed by molecular size, TPSA, fraction Csp3, and hydrogen-bond variables. This agrees with the exploratory negative association between lipophilicity and aqueous solubility. Gain and impurity importance describe model usage rather than causal contribution, and correlated descriptors can divide or exchange importance.

![Baseline test MAE](figures/baseline_test_mae.png)

![XGBoost descriptor importance](figures/xgboost_descriptor_importance.png)

![Prediction comparison](figures/xgboost_combined_predictions.png)

## Failure analysis

The largest errors were enriched for uncommon elements, charged dyes, salts, mixtures, highly halogenated structures, extreme LogS values, and structures outside the defined drug-like analysis scope. These samples expose the applicability boundary of a model trained on heterogeneous public solubility measurements. They should not be silently deleted from the official benchmark, but predictions for such compounds should be accompanied by an out-of-domain warning.

{markdown_table(failure_preview)}

Detailed sample-level failures are stored in `top_failure_samples.csv`, and subgroup metrics are stored in `subgroup_error_analysis.csv`.
Negative R² values within narrow LogS bands should not be interpreted as overall model failure because restricting labels to a narrow interval substantially reduces the denominator variance used by R². MAE, RMSE, and sample count are more informative for these subgroup comparisons.

## Main conclusion

XGBoost with combined RDKit and ECFP features is the Day 2 reference baseline. It should be treated as the model that subsequent GNN and Transformer experiments must exceed under the same split and evaluation protocol. Random Forest remains a competitive, stable alternative, while descriptor-only tree models provide the best speed–interpretability trade-off.

## Known limitations

For seed 42, all 2,940 samples in the official scaffold validation split have an empty Bemis–Murcko scaffold, and the validation labels show a strong distribution shift
"""

    with (
        REPORT_ROOT / "day2_baseline_report.md"
    ).open("w", encoding="utf-8") as file:
        file.write(report)


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    results = load_results()
    final_table = save_final_table(results)

    plot_baseline_mae(results)
    plot_xgboost_importance()
    plot_best_predictions()

    failures, subgroup = analyze_failures()

    write_report(
        results,
        final_table,
        failures,
        subgroup,
    )

    print(
        final_table[
            [
                "split_type",
                "model",
                "representation",
                "test_mae",
                "test_rmse",
                "test_r2",
                "test_spearman",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved Day 2 report to:")
    print(REPORT_ROOT / "day2_baseline_report.md")


if __name__ == "__main__":
    main()