"""Generate summary figures for Day 5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_ROOT = Path("reports/day5")
FIGURE_ROOT = REPORT_ROOT / "figures"

MODEL_COMPARISON_PATH = (
    REPORT_ROOT
    / "classification"
    / "three_model_final_test_comparison.csv"
)

DEVELOPMENT_SUMMARY_PATH = (
    REPORT_ROOT
    / "fine_tuning"
    / "development_summary.json"
)

RETRIEVAL_RESULTS_PATH = (
    REPORT_ROOT
    / "retrieval"
    / "final"
    / "retrieval_test_results.csv"
)

MODEL_LABELS = {
    "morgan_logistic": "Morgan + Logistic",
    "frozen_rxnfp": "Frozen RXNFP",
    "fine_tuned_rxnfp": "Fine-tuned RXNFP",
}

MODEL_COLORS = {
    "morgan_logistic": "#4C78A8",
    "frozen_rxnfp": "#F58518",
    "fine_tuned_rxnfp": "#54A24B",
}

TASK_LABELS = {
    ("transformation", "solvent"): (
        "Transformation\nSolvent"
    ),
    ("transformation", "catalyst"): (
        "Transformation\nCatalyst"
    ),
    ("reaction_center", "solvent"): (
        "Reaction center\nSolvent"
    ),
    ("reaction_center", "catalyst"): (
        "Reaction center\nCatalyst"
    ),
}

TASK_ORDER = [
    ("transformation", "solvent"),
    ("transformation", "catalyst"),
    ("reaction_center", "solvent"),
    ("reaction_center", "catalyst"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_bar_labels(
    axis,
    bars,
) -> None:
    for bar in bars:
        height = bar.get_height()

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.012,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )


def plot_model_comparison() -> None:
    dataframe = pd.read_csv(
        MODEL_COMPARISON_PATH
    )

    models = [
        "morgan_logistic",
        "frozen_rxnfp",
        "fine_tuned_rxnfp",
    ]

    metrics = [
        (
            "test_micro_ap",
            "Test micro average precision",
        ),
        (
            "test_hit_rate_at_5",
            "Test HitRate@5",
        ),
    ]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        constrained_layout=True,
    )

    x_positions = np.arange(
        len(TASK_ORDER)
    )

    width = 0.24

    for axis, (
        metric,
        title,
    ) in zip(axes, metrics):
        for model_index, model in enumerate(
            models
        ):
            values = []

            for protocol, target in TASK_ORDER:
                row = dataframe.loc[
                    dataframe["model"].eq(model)
                    & dataframe["protocol"].eq(
                        protocol
                    )
                    & dataframe["target"].eq(
                        target
                    )
                ]

                assert len(row) == 1

                values.append(
                    float(row.iloc[0][metric])
                )

            offset = (
                model_index
                - (len(models) - 1) / 2
            ) * width

            bars = axis.bar(
                x_positions + offset,
                values,
                width=width,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )

            add_bar_labels(
                axis,
                bars,
            )

        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.04)
        axis.set_xticks(
            x_positions,
            [
                TASK_LABELS[task]
                for task in TASK_ORDER
            ],
        )
        axis.grid(
            axis="y",
            alpha=0.25,
        )
        axis.legend(
            loc="upper right",
        )

    output_path = (
        FIGURE_ROOT
        / "day5_model_comparison.png"
    )

    figure.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)


def plot_fine_tuning_curves() -> None:
    with DEVELOPMENT_SUMMARY_PATH.open(
        encoding="utf-8"
    ) as file:
        summary = json.load(file)

    results_by_task = {
        (
            result["protocol"],
            result["target"],
        ): result
        for result in summary["results"]
    }

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        constrained_layout=True,
    )

    for axis, task in zip(
        axes.flat,
        TASK_ORDER,
    ):
        result = results_by_task[task]

        history = pd.read_csv(
            result["history_path"]
        )

        best_epoch = int(
            result["best_epoch"]
        )

        axis.plot(
            history["epoch"],
            history["train_loss"],
            color="#4C78A8",
            linewidth=1.8,
            label="Training loss",
        )

        axis.plot(
            history["epoch"],
            history["valid_loss"],
            color="#E45756",
            linewidth=1.8,
            label="Validation loss",
        )

        axis.set_xlabel("Epoch")
        axis.set_ylabel("BCE loss")
        axis.grid(alpha=0.2)

        metric_axis = axis.twinx()

        metric_axis.plot(
            history["epoch"],
            history["valid_micro_ap"],
            color="#54A24B",
            linewidth=2.0,
            label="Validation micro AP",
        )

        metric_axis.set_ylabel(
            "Validation micro AP"
        )

        metric_axis.set_ylim(
            0.0,
            min(
                1.0,
                max(
                    0.55,
                    history[
                        "valid_micro_ap"
                    ].max()
                    + 0.08,
                ),
            ),
        )

        axis.axvline(
            best_epoch,
            color="#7F3C8D",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Selected epoch "
                f"{best_epoch}"
            ),
        )

        handles_left, labels_left = (
            axis.get_legend_handles_labels()
        )

        handles_right, labels_right = (
            metric_axis
            .get_legend_handles_labels()
        )

        axis.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            loc="best",
            fontsize=8,
        )

        axis.set_title(
            TASK_LABELS[task].replace(
                "\n",
                " | ",
            )
        )

    output_path = (
        FIGURE_ROOT
        / "day5_fine_tuning_curves.png"
    )

    figure.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)


def plot_retrieval_metrics() -> None:
    dataframe = pd.read_csv(
        RETRIEVAL_RESULTS_PATH
    )

    metric_columns = [
        (
            "reaction_type_hit_at_5",
            "Reaction-type Hit@5",
        ),
        (
            "reaction_type_mrr_at_10",
            "Reaction-type MRR@10",
        ),
        (
            "solvent_recall_at_5",
            "Solvent recall@5",
        ),
        (
            "catalyst_recall_at_5",
            "Catalyst recall@5",
        ),
    ]

    protocols = [
        "transformation",
        "reaction_center",
    ]

    protocol_labels = {
        "transformation": (
            "Transformation split"
        ),
        "reaction_center": (
            "Reaction-center split"
        ),
    }

    colors = {
        "transformation": "#4C78A8",
        "reaction_center": "#F58518",
    }

    x_positions = np.arange(
        len(metric_columns)
    )

    width = 0.34

    figure, axis = plt.subplots(
        figsize=(11, 5.5),
        constrained_layout=True,
    )

    for protocol_index, protocol in enumerate(
        protocols
    ):
        row = dataframe.loc[
            dataframe["protocol"].eq(
                protocol
            )
        ]

        assert len(row) == 1

        values = [
            float(row.iloc[0][column])
            for column, _ in metric_columns
        ]

        offset = (
            protocol_index
            - (len(protocols) - 1) / 2
        ) * width

        bars = axis.bar(
            x_positions + offset,
            values,
            width=width,
            label=protocol_labels[
                protocol
            ],
            color=colors[protocol],
        )

        add_bar_labels(
            axis,
            bars,
        )

    axis.set_title(
        "Final RXNFP similar-reaction retrieval"
    )

    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)

    axis.set_xticks(
        x_positions,
        [
            label
            for _, label in metric_columns
        ],
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    output_path = (
        FIGURE_ROOT
        / "day5_retrieval_metrics.png"
    )

    figure.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)


def main() -> None:
    configure_style()

    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_model_comparison()
    plot_fine_tuning_curves()
    plot_retrieval_metrics()

    print(
        "\nDay 5 figures completed."
    )


if __name__ == "__main__":
    main()