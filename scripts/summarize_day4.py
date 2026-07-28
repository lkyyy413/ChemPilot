"""Generate the Day 4 reaction-condition report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPORT_ROOT = Path("reports/day4")

BUILD_REPORT = (
    REPORT_ROOT
    / "reaction_dataset_build_report.json"
)

SPLIT_REPORT = (
    REPORT_ROOT
    / "reaction_split_audit.json"
)

FEATURE_REPORT = (
    REPORT_ROOT
    / "reaction_feature_manifest.json"
)

TARGET_REPORT = (
    REPORT_ROOT
    / "reaction_target_manifest.json"
)

AD_REPORT = (
    REPORT_ROOT
    / "applicability"
    / "applicability_summary.json"
)

FINAL_RESULTS = (
    REPORT_ROOT
    / "classification"
    / "final_test_results.csv"
)

SEARCH_RESULTS = (
    REPORT_ROOT
    / "classification"
    / "logistic_c_search.csv"
)

OUTPUT_PATH = (
    REPORT_ROOT
    / "day4_reaction_condition_report.md"
)


def load_json(
    path: Path,
):
    with path.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


def format_float(
    value,
    digits: int = 4,
) -> str:
    return f"{float(value):.{digits}f}"


def format_percent(
    value,
    digits: int = 1,
) -> str:
    return (
        f"{100.0 * float(value):.{digits}f}%"
    )


def protocol_name(
    value: str,
) -> str:
    names = {
        "transformation": "Transformation",
        "reaction_center": (
            "Reaction center"
        ),
    }

    return names[value]


def target_name(
    value: str,
) -> str:
    names = {
        "solvent": "Solvent",
        "catalyst": "Catalyst",
    }

    return names[value]


def add_table(
    lines: list[str],
    headers: list[str],
    rows: list[list],
) -> None:
    lines.append(
        "| "
        + " | ".join(headers)
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(
            ["---"] * len(headers)
        )
        + " |"
    )

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(value)
                for value in row
            )
            + " |"
        )

    lines.append("")


def main() -> None:
    build = load_json(
        BUILD_REPORT
    )

    split = load_json(
        SPLIT_REPORT
    )

    features = load_json(
        FEATURE_REPORT
    )

    targets = load_json(
        TARGET_REPORT
    )

    applicability = load_json(
        AD_REPORT
    )

    final_results = pd.read_csv(
        FINAL_RESULTS
    )

    search_results = pd.read_csv(
        SEARCH_RESULTS
    )

    standardized = build[
        "standardized_experiments"
    ]

    aggregated = build[
        "aggregated_condition_pairs"
    ]

    reaction_features = features[
        "reaction_features"
    ]

    condition_features = features[
        "condition_features"
    ]

    lines: list[str] = []

    lines.extend(
        [
            "# Day 4: Reaction-condition prediction",
            "",
            "Day 4 constructs a reproducible "
            "reaction-condition benchmark from "
            "the Open Reaction Database (ORD) "
            "and trains reaction-only classifiers "
            "for solvent and catalyst recommendation.",
            "",
            "The source data are distributed under "
            "[CC BY-SA 4.0]"
            "(https://github.com/"
            "open-reaction-database/ord-data). "
            "Derived data products must preserve "
            "the applicable attribution and "
            "share-alike requirements.",
            "",
            "## Scope and scientific interpretation",
            "",
            "The selected d9297630 dataset contains "
            "high-throughput reaction-screening "
            "experiments. Its response variable is "
            "**LC area percent at 280 nm**, a "
            "semi-quantitative analytical response. "
            "It is not treated or reported as "
            "isolated reaction yield.",
            "",
            "The primary Day 4 tasks are multi-label "
            "solvent and catalyst recommendation "
            "from reactant and product structures. "
            "Reagent and temperature prediction are "
            "deferred until their label and missing-"
            "value behavior are modeled explicitly.",
            "",
            "## Dataset construction",
            "",
        ]
    )

    dataset_rows = [
        [
            "Raw standardized experiments",
            f"{standardized['rows']:,}",
        ],
        [
            "Unique reaction IDs",
            (
                f"{standardized['unique_reaction_ids']:,}"
            ),
        ],
        [
            "Aggregated condition pairs",
            f"{aggregated['rows']:,}",
        ],
        [
            "Removed exact replicate rows",
            (
                f"{aggregated['removed_replicate_rows']:,}"
            ),
        ],
        [
            "Unique transformations",
            (
                f"{aggregated['unique_transformations']:,}"
            ),
        ],
        [
            "Rankable transformations",
            (
                f"{aggregated['rankable_transformations']:,}"
            ),
        ],
        [
            "All-zero transformations",
            (
                f"{aggregated['all_zero_transformations']:,}"
            ),
        ],
        [
            "Maximum replicate count",
            (
                f"{aggregated['maximum_replicate_count']:,}"
            ),
        ],
    ]

    add_table(
        lines,
        ["Quantity", "Value"],
        dataset_rows,
    )

    temperature_missing = (
        standardized[
            "missing_temperature"
        ]
    )

    time_missing = standardized[
        "missing_reaction_time"
    ]

    lines.extend(
        [
            "Missing numeric conditions are retained "
            "as missing values rather than filled "
            "with global averages:",
            "",
            f"- Missing temperature: "
            f"{temperature_missing:,} "
            f"({temperature_missing / standardized['rows']:.2%})",
            f"- Missing reaction time: "
            f"{time_missing:,} "
            f"({time_missing / standardized['rows']:.2%})",
            f"- Zero-score experiments: "
            f"{standardized['zero_score_rows']:,}",
            "",
            "Exact transformation-condition "
            "replicates are aggregated, while "
            "different conditions for the same "
            "transformation are retained because "
            "they contain screening information.",
            "",
            "## Reaction centers",
            "",
            "Atom-mapped reaction SMILES are used "
            "to derive deterministic reaction-center "
            "signatures. Conflicting center variants "
            "within the same standardized "
            "transformation are resolved using a "
            "replicate-weighted consensus.",
            "",
        ]
    )

    center_fields = [
        (
            "unique_raw_signatures",
            "Unique raw center signatures",
        ),
        (
            "unique_consensus_signatures",
            "Unique consensus center signatures",
        ),
        (
            "conflicting_transformations",
            "Transformations with center conflicts",
        ),
        (
            "condition_pairs_in_conflicts",
            "Condition pairs in conflicts",
        ),
        (
            "condition_pairs_changed_by_consensus",
            "Pairs changed by consensus",
        ),
        (
            "zero_feature_condition_pairs",
            "Zero-feature center pairs",
        ),
        (
            "maximum_variants_per_transformation",
            "Maximum center variants",
        ),
    ]

    center_rows = []

    for key, label in center_fields:
        if key in aggregated:
            center_rows.append(
                [
                    label,
                    aggregated[key],
                ]
            )

    if center_rows:
        add_table(
            lines,
            ["Reaction-center audit", "Value"],
            center_rows,
        )
    else:
        lines.extend(
            [
                "Reaction-center summary statistics "
                "are stored in the dataset build "
                "report and processed Parquet.",
                "",
            ]
        )

    lines.extend(
        [
            "## Leakage-controlled splits",
            "",
            "Two group-aware 70/15/15 protocols "
            "are retained. Split assignment uses "
            "group sizes and row counts only; "
            "reaction scores and condition ranks "
            "are not used.",
            "",
        ]
    )

    split_rows = []

    for protocol in [
        "transformation",
        "reaction_center",
    ]:
        protocol_result = split[
            "protocols"
        ][protocol]

        for split_name in [
            "train",
            "valid",
            "test",
        ]:
            values = protocol_result[
                "splits"
            ][split_name]

            split_rows.append(
                [
                    protocol_name(protocol),
                    split_name,
                    (
                        f"{values['condition_pairs']:,}"
                    ),
                    format_percent(
                        values[
                            "condition_pair_proportion"
                        ]
                    ),
                    values["transformations"],
                    values["reaction_centers"],
                    (
                        values[
                            "rankable_transformations"
                        ]
                    ),
                ]
            )

    add_table(
        lines,
        [
            "Protocol",
            "Split",
            "Condition pairs",
            "Pair share",
            "Transformations",
            "Centers",
            "Rankable transformations",
        ],
        split_rows,
    )

    overlap_rows = []

    for protocol in [
        "transformation",
        "reaction_center",
    ]:
        audits = split[
            "protocols"
        ][protocol][
            "overlap_audit"
        ]

        for comparison, values in (
            audits.items()
        ):
            overlap_rows.append(
                [
                    protocol_name(protocol),
                    comparison.replace(
                        "_",
                        "–",
                    ),
                    values[
                        "transformation_overlap"
                    ],
                    values[
                        "reaction_center_overlap"
                    ],
                ]
            )

    add_table(
        lines,
        [
            "Protocol",
            "Comparison",
            "Transformation overlap",
            "Reaction-center overlap",
        ],
        overlap_rows,
    )

    lines.extend(
        [
            "Condition signatures may overlap across "
            "splits because the prediction input is "
            "the reaction structure, not a condition "
            "identifier. Exact transformations never "
            "overlap. Reaction centers additionally "
            "never overlap in the harder reaction-"
            "center protocol.",
            "",
            "## Feature representations",
            "",
        ]
    )

    feature_rows = [
        [
            "Reaction combined",
            (
                " × ".join(
                    str(value)
                    for value in reaction_features[
                        "shape"
                    ]
                )
            ),
            reaction_features["dtype"],
            format_percent(
                reaction_features["density"],
                digits=3,
            ),
        ],
        [
            "Condition combined",
            (
                " × ".join(
                    str(value)
                    for value in condition_features[
                        "shape"
                    ]
                )
            ),
            condition_features["dtype"],
            format_percent(
                condition_features["density"],
                digits=3,
            ),
        ],
    ]

    add_table(
        lines,
        [
            "Feature matrix",
            "Shape",
            "Dtype",
            "Density",
        ],
        feature_rows,
    )

    lines.extend(
        [
            "The classifier input is the 6,144-"
            "dimensional reaction representation: "
            "2,048-bit reactant Morgan fingerprint, "
            "2,048-bit product fingerprint, and "
            "their signed difference. Condition "
            "features are built for later condition-"
            "pair scoring and are not used as inputs "
            "to the solvent/catalyst classifiers.",
            "",
            "No label, score, condition rank, or "
            "split identity is used to construct "
            "reaction features.",
            "",
            "## Multi-label targets",
            "",
            f"Sample unit: {targets['sample_unit']}",
            "",
            f"Positive-label policy: "
            f"{targets['positive_label_policy']}",
            "",
            f"Missing-condition policy: "
            f"{targets['missing_condition_policy']}",
            "",
        ]
    )

    target_rows = []

    for protocol in [
        "transformation",
        "reaction_center",
    ]:
        for target in [
            "solvent",
            "catalyst",
        ]:
            result = targets[
                "protocols"
            ][protocol][target]

            test_statistics = result[
                "split_statistics"
            ]["test"]

            target_rows.append(
                [
                    protocol_name(protocol),
                    target_name(target),
                    result["classes"],
                    (
                        result[
                            "minimum_train_transformations"
                        ]
                    ),
                    (
                        test_statistics[
                            "transformations_with_any_known_label"
                        ]
                    ),
                    format_percent(
                        test_statistics[
                            "all_known_transformation_coverage"
                        ]
                    ),
                ]
            )

    add_table(
        lines,
        [
            "Protocol",
            "Target",
            "Classes",
            "Minimum train frequency",
            "Evaluable test samples",
            "All-label-known coverage",
        ],
        target_rows,
    )

    lines.extend(
        [
            "Vocabularies are constructed from the "
            "training split only. Unknown validation "
            "and test labels remain unknown and are "
            "not silently mapped to negatives.",
            "",
            "## Model selection",
            "",
            "One-vs-rest logistic regression with "
            "balanced class weights is used as the "
            "first condition classifier. The "
            "regularization parameter is selected "
            "on validation micro-average precision. "
            "Test labels are evaluated once after "
            "selection.",
            "",
        ]
    )

    selected = search_results.loc[
        search_results["selected"]
    ]

    selection_rows = []

    for row in selected.itertuples(
        index=False
    ):
        selection_rows.append(
            [
                protocol_name(row.protocol),
                target_name(row.target),
                format_float(row.c, 2),
                format_float(
                    row.micro_average_precision
                ),
                format_float(
                    row.hit_rate_at_5
                ),
            ]
        )

    add_table(
        lines,
        [
            "Protocol",
            "Target",
            "Selected C",
            "Validation micro AP",
            "Validation HitRate@5",
        ],
        selection_rows,
    )

    lines.extend(
        [
            "## Final test results",
            "",
        ]
    )

    final_rows = []

    for row in final_results.itertuples(
        index=False
    ):
        final_rows.append(
            [
                protocol_name(row.protocol),
                target_name(row.target),
                row.classes,
                row.test_evaluated_samples,
                format_float(
                    row.test_micro_ap
                ),
                format_float(
                    row.test_mrr
                ),
                format_float(
                    row.test_hit_rate_at_1
                ),
                format_float(
                    row.test_hit_rate_at_5
                ),
                format_float(
                    row.test_recall_at_5
                ),
                format_float(
                    row.frequency_hit_rate_at_5
                ),
            ]
        )

    add_table(
        lines,
        [
            "Protocol",
            "Target",
            "Classes",
            "Test n",
            "Micro AP",
            "MRR",
            "Hit@1",
            "Hit@5",
            "Recall@5",
            "Frequency Hit@5",
        ],
        final_rows,
    )

    lines.extend(
        [
            "All four models exceed their frequency "
            "baseline on HitRate@5. Solvent prediction "
            "is substantially more mature than "
            "catalyst prediction. Catalyst evaluation "
            "remains difficult because hundreds of "
            "classes are learned from only about two "
            "hundred final training transformations.",
            "",
            "![Final Top-5 performance]"
            "(figures/condition_classifier_hit_rate.png)",
            "",
            "![Final classification metrics]"
            "(figures/condition_classifier_metrics.png)",
            "",
            "## Label and applicability coverage",
            "",
        ]
    )

    ad_rows = []

    for protocol in [
        "transformation",
        "reaction_center",
    ]:
        for target in [
            "solvent",
            "catalyst",
        ]:
            result = applicability[
                "tasks"
            ][f"{protocol}|{target}"]

            coverage = result[
                "complete_test_applicability"
            ]

            in_metrics = result[
                "in_domain_test_metrics"
            ]

            out_metrics = result[
                "out_of_domain_test_metrics"
            ]

            ad_rows.append(
                [
                    protocol_name(protocol),
                    target_name(target),
                    format_float(
                        result["ad_threshold"]
                    ),
                    format_percent(
                        coverage["in_domain_rate"]
                    ),
                    (
                        f"{in_metrics['samples']} / "
                        f"{out_metrics['samples']}"
                    ),
                ]
            )

    add_table(
        lines,
        [
            "Protocol",
            "Target",
            "AD threshold",
            "Complete-test in-domain rate",
            "Evaluated in/out",
        ],
        ad_rows,
    )

    lines.extend(
        [
            "Applicability-domain thresholds are the "
            "fifth percentile of leave-one-out nearest-"
            "neighbor Tanimoto similarity in the final "
            "train-plus-validation subset. Test labels "
            "are not used to define the thresholds.",
            "",
            "The number of evaluated out-of-domain "
            "samples is small, especially in the "
            "transformation protocol. Domain-stratified "
            "performance is therefore descriptive and "
            "must not be interpreted as evidence that "
            "out-of-domain predictions are equally "
            "reliable.",
            "",
            "![Coverage analysis]"
            "(figures/condition_classifier_coverage.png)",
            "",
            "## Inference interface",
            "",
            "The `ReactionConditionPredictor` accepts "
            "reactant and product SMILES and returns:",
            "",
            "- solvent Top-K labels;",
            "- catalyst Top-K labels;",
            "- uncalibrated ranking scores;",
            "- nearest training transformation;",
            "- Tanimoto similarity and AD threshold;",
            "- an `in_domain` warning flag.",
            "",
            "Example:",
            "",
            "```bash",
            "python scripts/predict_reaction_conditions.py \\",
            '  --reactant "Brc1ccc2ncccc2c1" \\',
            '  --reactant "O=S([O-])C1CC1.[Na+]" \\',
            '  --product "c1cnc2ccc(C3CC3)cc2c1" \\',
            "  --protocol reaction_center \\",
            "  --top-k 5",
            "```",
            "",
            "The solvent and catalyst models are "
            "independent. Combining their separate "
            "Top-1 outputs does not establish the "
            "best joint experimental condition.",
            "",
            "## Reproduction",
            "",
            "Day 4 uses the dedicated Python 3.11 "
            "`chempilot-ord` environment because the "
            "ORD schema stack differs from the "
            "PyTorch environment used for Days 1–3.",
            "",
            "```bash",
            "python scripts/build_ord_reaction_dataset.py",
            "python scripts/create_ord_reaction_splits.py",
            "python scripts/build_reaction_features.py",
            "python scripts/audit_reaction_label_space.py",
            "python scripts/build_reaction_label_targets.py",
            "python scripts/search_reaction_condition_classifiers.py",
            "# Run final test evaluation using selected C values",
            "python scripts/analyze_reaction_applicability.py",
            "python scripts/plot_day4.py",
            "python scripts/summarize_day4.py",
            "```",
            "",
            "## Limitations and next steps",
            "",
            "1. LC area percent is not isolated yield.",
            "2. Solvent and catalyst recommendations "
            "are independent multi-label outputs.",
            "3. Logistic scores are not calibrated "
            "probabilities.",
            "4. Catalyst vocabularies are large "
            "relative to the training sample count.",
            "5. Unknown test labels make closed-"
            "vocabulary metrics optimistic for some "
            "catalyst samples.",
            "6. The AD threshold is a structural "
            "coverage heuristic, not a correctness "
            "guarantee.",
            "7. The next modeling stage should rank "
            "joint condition pairs or predict the "
            "screening response using reaction and "
            "condition features.",
            "",
        ]
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Saved:", OUTPUT_PATH)
    print(
        "Lines:",
        len(lines),
    )


if __name__ == "__main__":
    main()