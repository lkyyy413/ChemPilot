"""Generate Day 4 reaction-condition figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_PATH = Path(
    "reports/day4/classification/"
    "final_test_results.csv"
)

AD_PATH = Path(
    "reports/day4/applicability/"
    "applicability_summary.json"
)

OUTPUT_ROOT = Path(
    "reports/day4/figures"
)


PROTOCOL_LABELS = {
    "transformation": "Transformation split",
    "reaction_center": "Reaction-center split",
}

TARGET_LABELS = {
    "solvent": "Solvent",
    "catalyst": "Catalyst",
}


def add_bar_labels(
    axis,
    bars,
    *,
    digits: int = 3,
) -> None:
    for bar in bars:
        value = bar.get_height()

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_hit_rate_comparison(
    results: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
    )

    protocol_order = [
        "transformation",
        "reaction_center",
    ]

    for axis, target in zip(
        axes,
        ["solvent", "catalyst"],
    ):
        subset = (
            results.loc[
                results["target"].eq(target)
            ]
            .set_index("protocol")
            .loc[protocol_order]
            .reset_index()
        )

        positions = np.arange(
            len(subset)
        )

        width = 0.34

        model_bars = axis.bar(
            positions - width / 2,
            subset[
                "test_hit_rate_at_5"
            ],
            width,
            label="Logistic model",
        )

        baseline_bars = axis.bar(
            positions + width / 2,
            subset[
                "frequency_hit_rate_at_5"
            ],
            width,
            label="Frequency baseline",
        )

        add_bar_labels(
            axis,
            model_bars,
        )

        add_bar_labels(
            axis,
            baseline_bars,
        )

        axis.set_title(
            TARGET_LABELS[target]
        )

        axis.set_xticks(
            positions,
            [
                "Transformation",
                "Reaction center",
            ],
        )

        axis.set_ylim(0.0, 1.08)
        axis.set_ylabel("Test HitRate@5")
        axis.grid(
            axis="y",
            alpha=0.3,
        )

    axes[0].legend(
        loc="lower right"
    )

    figure.suptitle(
        "Reaction-condition Top-5 performance"
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / "condition_classifier_hit_rate.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)


def plot_test_metrics(
    results: pd.DataFrame,
) -> None:
    order = [
        ("transformation", "solvent"),
        ("reaction_center", "solvent"),
        ("transformation", "catalyst"),
        ("reaction_center", "catalyst"),
    ]

    ordered_rows = []

    for protocol, target in order:
        row = results.loc[
            results["protocol"].eq(protocol)
            & results["target"].eq(target)
        ].iloc[0]

        ordered_rows.append(row)

    ordered = pd.DataFrame(
        ordered_rows
    ).reset_index(drop=True)

    labels = [
        "Transform.\nsolvent",
        "Center\nsolvent",
        "Transform.\ncatalyst",
        "Center\ncatalyst",
    ]

    metrics = [
        (
            "test_micro_ap",
            "Micro AP",
        ),
        (
            "test_mrr",
            "MRR",
        ),
        (
            "test_recall_at_5",
            "Recall@5",
        ),
    ]

    positions = np.arange(
        len(ordered)
    )

    width = 0.24

    figure, axis = plt.subplots(
        figsize=(11, 5.5)
    )

    for metric_index, (
        column,
        label,
    ) in enumerate(metrics):
        offset = (
            metric_index - 1
        ) * width

        bars = axis.bar(
            positions + offset,
            ordered[column],
            width,
            label=label,
        )

        add_bar_labels(
            axis,
            bars,
        )

    axis.set_xticks(
        positions,
        labels,
    )

    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Test metric")
    axis.set_title(
        "Final reaction-condition "
        "classification metrics"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend(
        loc="upper right"
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / "condition_classifier_metrics.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)


def plot_coverage(
    results: pd.DataFrame,
    ad_report: dict,
) -> None:
    order = [
        ("transformation", "solvent"),
        ("reaction_center", "solvent"),
        ("transformation", "catalyst"),
        ("reaction_center", "catalyst"),
    ]

    records = []

    for protocol, target in order:
        result_row = results.loc[
            results["protocol"].eq(protocol)
            & results["target"].eq(target)
        ].iloc[0]

        task = ad_report[
            "tasks"
        ][f"{protocol}|{target}"]

        complete_test = task[
            "complete_test_samples"
        ]

        evaluated_test = task[
            "evaluated_test_samples"
        ]

        in_domain_rate = task[
            "complete_test_applicability"
        ]["in_domain_rate"]

        all_known_rate = (
            result_row[
                "all_targets_known"
            ]
            / evaluated_test
        )

        records.append(
            {
                "label": (
                    (
                        "Transform."
                        if protocol
                        == "transformation"
                        else "Center"
                    )
                    + "\n"
                    + target
                ),
                "evaluable_rate": (
                    evaluated_test
                    / complete_test
                ),
                "all_known_rate": (
                    all_known_rate
                ),
                "in_domain_rate": (
                    in_domain_rate
                ),
            }
        )

    coverage = pd.DataFrame(records)

    positions = np.arange(
        len(coverage)
    )

    width = 0.24

    figure, axis = plt.subplots(
        figsize=(11, 5.5)
    )

    series = [
        (
            "evaluable_rate",
            "At least one known target",
        ),
        (
            "all_known_rate",
            "All targets known",
        ),
        (
            "in_domain_rate",
            "Structurally in domain",
        ),
    ]

    for series_index, (
        column,
        label,
    ) in enumerate(series):
        offset = (
            series_index - 1
        ) * width

        bars = axis.bar(
            positions + offset,
            coverage[column],
            width,
            label=label,
        )

        add_bar_labels(
            axis,
            bars,
        )

    axis.set_xticks(
        positions,
        coverage["label"],
    )

    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Fraction of test samples")
    axis.set_title(
        "Test-label and applicability-domain "
        "coverage"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend(
        loc="lower left"
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_ROOT
        / "condition_classifier_coverage.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.read_csv(
        RESULT_PATH
    )

    with AD_PATH.open(
        encoding="utf-8"
    ) as file:
        ad_report = json.load(file)

    expected_tasks = {
        (
            "transformation",
            "solvent",
        ),
        (
            "transformation",
            "catalyst",
        ),
        (
            "reaction_center",
            "solvent",
        ),
        (
            "reaction_center",
            "catalyst",
        ),
    }

    observed_tasks = set(
        zip(
            results["protocol"],
            results["target"],
        )
    )

    if observed_tasks != expected_tasks:
        raise ValueError(
            "Final result tasks are incomplete."
        )

    plot_hit_rate_comparison(
        results
    )

    plot_test_metrics(
        results
    )

    plot_coverage(
        results,
        ad_report,
    )


if __name__ == "__main__":
    main()