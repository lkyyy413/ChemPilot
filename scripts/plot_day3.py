"""Generate figures for the Day 3 GINE report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports/day3"
FIGURE_ROOT = REPORT_ROOT / "figures"

PROTOCOLS = ["random", "scaffold"]

sns.set_theme(
    style="whitegrid",
    context="talk",
)


def plot_learning_curves() -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5.5),
        sharey=False,
    )

    for axis, protocol in zip(
        axes,
        PROTOCOLS,
    ):
        history = pd.read_csv(
            REPORT_ROOT
            / "history/"
              f"{protocol}_gine_development.csv"
        )

        best_index = history[
            "valid_mae"
        ].idxmin()
        best = history.loc[best_index]

        axis.plot(
            history["epoch"],
            history["train_mae"],
            label="Train MAE",
            linewidth=2,
        )
        axis.plot(
            history["epoch"],
            history["valid_mae"],
            label="Validation MAE",
            linewidth=2,
        )
        axis.scatter(
            [best["epoch"]],
            [best["valid_mae"]],
            color="black",
            marker="*",
            s=180,
            zorder=5,
            label=(
                f"Best epoch {int(best['epoch'])}"
            ),
        )
        axis.axvline(
            best["epoch"],
            color="black",
            linestyle="--",
            alpha=0.4,
        )

        axis.set_title(
            f"{protocol.capitalize()} development"
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MAE (LogS)")
        axis.legend(fontsize=10)

    figure.suptitle(
        "GINE development learning curves",
        fontsize=17,
    )
    figure.tight_layout()

    figure.savefig(
        FIGURE_ROOT
        / "gine_learning_curves.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_test_mae_comparison() -> None:
    comparison = pd.read_csv(
        REPORT_ROOT
        / "gine_xgboost_comparison.csv"
    )

    rows = []

    for _, row in comparison.iterrows():
        rows.append({
            "protocol": row["protocol"],
            "model": row["model"],
            "mae": row["test_mae"],
            "ci95": (
                0.0
                if pd.isna(
                    row.get("test_mae_ci95")
                )
                else row["test_mae_ci95"]
            ),
        })

    table = pd.DataFrame(rows)

    figure, axis = plt.subplots(
        figsize=(9, 6),
    )

    x_positions = np.arange(
        len(PROTOCOLS)
    )
    width = 0.34

    colors = {
        "XGBoost": "#2E86AB",
        "GINE": "#E07A5F",
    }

    for offset, model in [
        (-width / 2, "XGBoost"),
        (width / 2, "GINE"),
    ]:
        subset = (
            table[
                table["model"] == model
            ]
            .set_index("protocol")
            .loc[PROTOCOLS]
        )

        bars = axis.bar(
            x_positions + offset,
            subset["mae"],
            width=width,
            yerr=subset["ci95"],
            capsize=5,
            label=model,
            color=colors[model],
        )

        axis.bar_label(
            bars,
            labels=[
                f"{value:.3f}"
                for value in subset["mae"]
            ],
            padding=4,
            fontsize=10,
        )

    axis.set_xticks(
        x_positions,
        ["Random", "Scaffold"],
    )
    axis.set_ylabel("Test MAE (LogS)")
    axis.set_title(
        "XGBoost versus GINE test performance"
    )
    axis.legend()
    axis.set_ylim(
        0,
        table["mae"].max() * 1.25,
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_ROOT
        / "gine_xgboost_test_mae.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_subgroup_differences() -> None:
    table = pd.read_csv(
        REPORT_ROOT
        / "subgroups/"
          "gine_xgboost_subgroup_metrics.csv"
    )

    dimensions = [
        "similarity_bin",
        "molecular_weight_bin",
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 11),
    )

    for row_index, dimension in enumerate(
        dimensions
    ):
        for column_index, protocol in enumerate(
            PROTOCOLS
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            subset = table[
                (table["protocol"] == protocol)
                & (
                    table["dimension"]
                    == dimension
                )
            ].copy()

            values = subset[
                "gine_minus_xgboost_mae"
            ]

            colors = [
                "#C84C4C"
                if value > 0
                else "#3A9679"
                for value in values
            ]

            bars = axis.bar(
                subset["subgroup"],
                values,
                color=colors,
            )

            axis.axhline(
                0,
                color="black",
                linewidth=1,
            )
            axis.bar_label(
                bars,
                labels=[
                    f"{value:+.3f}"
                    for value in values
                ],
                padding=3,
                fontsize=9,
            )
            axis.tick_params(
                axis="x",
                rotation=25,
            )
            axis.set_ylabel(
                "GINE MAE − XGBoost MAE"
            )
            axis.set_title(
                f"{protocol.capitalize()} — "
                f"{dimension.replace('_', ' ')}"
            )

    figure.suptitle(
        "Subgroup error differences "
        "(positive values favor XGBoost)",
        fontsize=17,
    )
    figure.tight_layout()

    figure.savefig(
        FIGURE_ROOT
        / "gine_xgboost_subgroup_differences.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_uncertainty_analysis() -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5.5),
        sharey=True,
    )

    for axis, protocol in zip(
        axes,
        PROTOCOLS,
    ):
        table = pd.read_csv(
            REPORT_ROOT
            / "subgroups/"
              f"{protocol}_uncertainty_analysis.csv"
        )

        positions = np.arange(len(table))
        width = 0.34

        axis.bar(
            positions - width / 2,
            table["gine_mae"],
            width=width,
            label="GINE ensemble",
            color="#E07A5F",
        )
        axis.bar(
            positions + width / 2,
            table["xgboost_mae"],
            width=width,
            label="XGBoost",
            color="#2E86AB",
        )

        axis.set_xticks(
            positions,
            table["uncertainty_quartile"],
        )
        axis.set_xlabel(
            "GINE seed-disagreement quartile"
        )
        axis.set_ylabel("MAE (LogS)")
        axis.set_title(protocol.capitalize())
        axis.legend(fontsize=10)

    figure.suptitle(
        "Error by GINE ensemble disagreement",
        fontsize=17,
    )
    figure.tight_layout()

    figure.savefig(
        FIGURE_ROOT
        / "gine_uncertainty_analysis.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_prediction_comparison() -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 12),
        sharex=True,
        sharey=True,
    )

    for row_index, protocol in enumerate(
        PROTOCOLS
    ):
        table = pd.read_csv(
            REPORT_ROOT
            / "subgroups/"
              f"{protocol}_sample_analysis.csv"
        )

        configurations = [
            (
                "XGBoost",
                "xgboost_prediction",
                "#2E86AB",
            ),
            (
                "GINE ensemble",
                "gine_prediction",
                "#E07A5F",
            ),
        ]

        for column_index, (
            model_name,
            prediction_column,
            color,
        ) in enumerate(configurations):
            axis = axes[
                row_index,
                column_index,
            ]

            axis.scatter(
                table["y_true"],
                table[prediction_column],
                s=10,
                alpha=0.35,
                color=color,
                edgecolors="none",
            )

            minimum = min(
                table["y_true"].min(),
                table[prediction_column].min(),
            )
            maximum = max(
                table["y_true"].max(),
                table[prediction_column].max(),
            )

            axis.plot(
                [minimum, maximum],
                [minimum, maximum],
                linestyle="--",
                color="black",
                linewidth=1,
            )

            axis.set_title(
                f"{protocol.capitalize()} — "
                f"{model_name}"
            )
            axis.set_xlabel("Observed LogS")
            axis.set_ylabel("Predicted LogS")

    figure.suptitle(
        "Observed versus predicted aqueous solubility",
        fontsize=17,
    )
    figure.tight_layout()

    figure.savefig(
        FIGURE_ROOT
        / "gine_xgboost_predictions.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_learning_curves()
    plot_test_mae_comparison()
    plot_subgroup_differences()
    plot_uncertainty_analysis()
    plot_prediction_comparison()

    print("Saved Day 3 figures:")

    for path in sorted(
        FIGURE_ROOT.glob("gine_*.png")
    ):
        print(path)


if __name__ == "__main__":
    main()