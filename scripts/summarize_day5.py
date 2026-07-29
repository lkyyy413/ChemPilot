"""Generate the reproducible Day 5 report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPORT_ROOT = Path("reports/day5")

MODEL_COMPARISON_PATH = (
    REPORT_ROOT
    / "classification"
    / "three_model_final_test_comparison.csv"
)

VALIDATION_COMPARISON_PATH = (
    REPORT_ROOT
    / "fine_tuning"
    / "three_model_validation_comparison.csv"
)

DEVELOPMENT_SUMMARY_PATH = (
    REPORT_ROOT
    / "fine_tuning"
    / "development_summary.json"
)

SEQUENCE_AUDIT_PATH = (
    REPORT_ROOT
    / "reaction_sequences"
    / "reaction_sequence_audit.json"
)

TOKENIZATION_AUDIT_PATH = (
    REPORT_ROOT
    / "tokenization"
    / "rxnfp_tokenization_audit.json"
)

PRETRAINED_MANIFEST_PATH = (
    REPORT_ROOT
    / "rxnfp_pretrained_manifest.json"
)

RETRIEVAL_VALIDATION_PATH = (
    REPORT_ROOT
    / "retrieval"
    / "retrieval_validation.json"
)

RETRIEVAL_TEST_PATH = (
    REPORT_ROOT
    / "retrieval"
    / "final"
    / "retrieval_test_results.csv"
)

OUTPUT_PATH = (
    REPORT_ROOT
    / "day5_reaction_transformer_report.md"
)


def read_json(path: Path) -> dict:
    with path.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


def markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> list[str]:
    def clean(value: object) -> str:
        return str(value).replace(
            "|",
            "\\|",
        )

    output = [
        "| "
        + " | ".join(
            clean(header)
            for header in headers
        )
        + " |",
        "| "
        + " | ".join(
            "---"
            for _ in headers
        )
        + " |",
    ]

    for row in rows:
        output.append(
            "| "
            + " | ".join(
                clean(value)
                for value in row
            )
            + " |"
        )

    return output


def format_score(value: object) -> str:
    return f"{float(value):.4f}"


def task_name(
    protocol: str,
    target: str,
) -> str:
    protocol_label = {
        "transformation": "Transformation",
        "reaction_center": "Reaction center",
    }[protocol]

    return (
        f"{protocol_label} / "
        f"{target.capitalize()}"
    )


def main() -> None:
    required_paths = [
        MODEL_COMPARISON_PATH,
        VALIDATION_COMPARISON_PATH,
        DEVELOPMENT_SUMMARY_PATH,
        SEQUENCE_AUDIT_PATH,
        TOKENIZATION_AUDIT_PATH,
        PRETRAINED_MANIFEST_PATH,
        RETRIEVAL_VALIDATION_PATH,
        RETRIEVAL_TEST_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    comparison = pd.read_csv(
        MODEL_COMPARISON_PATH
    )

    validation = pd.read_csv(
        VALIDATION_COMPARISON_PATH
    )

    development = read_json(
        DEVELOPMENT_SUMMARY_PATH
    )

    sequence_audit = read_json(
        SEQUENCE_AUDIT_PATH
    )

    tokenization_audit = read_json(
        TOKENIZATION_AUDIT_PATH
    )

    pretrained = read_json(
        PRETRAINED_MANIFEST_PATH
    )

    retrieval_validation = read_json(
        RETRIEVAL_VALIDATION_PATH
    )

    retrieval_test = pd.read_csv(
        RETRIEVAL_TEST_PATH
    )

    modeling = sequence_audit[
        "modeling_transformations"
    ]

    token_lengths = modeling[
        "token_lengths"
    ]

    threshold_exceedance = token_lengths[
        "threshold_exceedance"
    ]

    model_config = pretrained[
        "model_config"
    ]

    best_models = (
        comparison.sort_values(
            "test_micro_ap",
            ascending=False,
        )
        .groupby(
            [
                "protocol",
                "target",
            ],
            as_index=False,
        )
        .first()
    )

    morgan_wins = int(
        best_models[
            "model"
        ].eq(
            "morgan_logistic"
        ).sum()
    )

    lines = [
        "# Day 5: Reaction Transformer and "
        "Similar-Reaction Retrieval",
        "",
        (
            "Generated at "
            f"{datetime.now(timezone.utc).isoformat()}."
        ),
        "",
        "## Objective",
        "",
        (
            "Day 5 evaluates whether a pretrained "
            "reaction-SMILES Transformer improves "
            "solvent and catalyst recommendation over "
            "the Day 4 Morgan-fingerprint baseline. "
            "It also builds a similar-reaction retrieval "
            "system that returns structurally relevant "
            "historical precedents and their observed "
            "conditions."
        ),
        "",
        (
            "The condition labels remain multi-label "
            "targets. Reported scores are ranking scores. "
            "The ORD response is LC area percent at "
            "280 nm and is not treated or reported as "
            "isolated reaction yield."
        ),
        "",
        "## Reaction-sequence representation",
        "",
        (
            "Each input sequence is constructed as "
            "`sorted canonical reactants>>sorted "
            "canonical products`. Reagents, solvents, "
            "catalysts, atom mappings, and target labels "
            "are excluded from the Transformer input."
        ),
        "",
    ]

    lines.extend(
        markdown_table(
            [
                "Statistic",
                "Value",
            ],
            [
                [
                    "Modeling transformations",
                    str(modeling["rows"]),
                ],
                [
                    "Unique sequences",
                    str(
                        modeling[
                            "unique_sequences"
                        ]
                    ),
                ],
                [
                    "RDKit-invalid sequences",
                    str(
                        modeling[
                            "rdkit_invalid"
                        ]
                    ),
                ],
                [
                    "Lexically unmatched sequences",
                    str(
                        modeling[
                            "lexically_unmatched"
                        ]
                    ),
                ],
                [
                    "Maximum lexical tokens",
                    str(
                        token_lengths["maximum"]
                    ),
                ],
                [
                    "Sequences exceeding 128 tokens",
                    (
                        f"{threshold_exceedance['128']['count']} "
                        f"({threshold_exceedance['128']['rate']:.2%})"
                    ),
                ],
                [
                    "Sequences exceeding 256 tokens",
                    str(
                        threshold_exceedance[
                            "256"
                        ]["count"]
                    ),
                ],
            ],
        )
    )

    lines.extend(
        [
            "",
            (
                "A maximum sequence length of 256 was "
                "therefore selected: it avoids truncation "
                "for all 381 modeling transformations "
                "while remaining well below the model's "
                "512-position limit."
            ),
            "",
            "## RXNFP checkpoint and tokenization",
            "",
            (
                "The `bert_pretrained` checkpoint was "
                "extracted from the RXNFP 0.1.0 wheel. "
                "The masked-language-model checkpoint was "
                "chosen instead of the reaction-classification "
                "fine-tuned checkpoint to avoid importing "
                "source classification-task supervision."
            ),
            "",
        ]
    )

    token_summary = tokenization_audit.get(
        "summary",
        tokenization_audit,
    )

    unknown_tokens = token_summary.get(
        "unknown_tokens",
        0,
    )

    sequences_with_unknown = token_summary.get(
        "sequences_with_unknown",
        0,
    )

    lines.extend(
        markdown_table(
            [
                "Checkpoint property",
                "Value",
            ],
            [
                [
                    "Vocabulary size",
                    str(
                        model_config[
                            "vocab_size"
                        ]
                    ),
                ],
                [
                    "Hidden size",
                    str(
                        model_config[
                            "hidden_size"
                        ]
                    ),
                ],
                [
                    "Transformer layers",
                    str(
                        model_config[
                            "num_hidden_layers"
                        ]
                    ),
                ],
                [
                    "Attention heads",
                    str(
                        model_config[
                            "num_attention_heads"
                        ]
                    ),
                ],
                [
                    "Maximum positions",
                    str(
                        model_config[
                            "max_position_embeddings"
                        ]
                    ),
                ],
                [
                    "Unknown tokens",
                    str(unknown_tokens),
                ],
                [
                    "Sequences with unknown tokens",
                    str(
                        sequences_with_unknown
                    ),
                ],
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Frozen Transformer features",
            "",
            (
                "Two 256-dimensional reaction embeddings "
                "were cached for every modeling reaction:"
            ),
            "",
            (
                "- `CLS`: the final hidden state of the "
                "special classification token."
            ),
            (
                "- `masked_mean`: the attention-mask-aware "
                "mean of non-padding token hidden states."
            ),
            "",
            (
                "Validation search compared both pooling "
                "methods and logistic-regression "
                "regularization values. Masked-mean pooling "
                "was selected for all four condition "
                "classification tasks."
            ),
            "",
            "## Partial fine-tuning",
            "",
            (
                "The embedding layer and Transformer layers "
                "0–9 were frozen. Layers 10–11 and a new "
                "multi-label linear classifier were trained "
                "with BCE-with-logits loss. Class imbalance "
                "was handled using positive-class weights "
                "clipped at 20."
            ),
            "",
        ]
    )

    fine_tuning_rows = []

    for result in development["results"]:
        metrics = result[
            "best_validation_metrics"
        ]

        fine_tuning_rows.append(
            [
                task_name(
                    result["protocol"],
                    result["target"],
                ),
                str(result["best_epoch"]),
                str(
                    result[
                        "epochs_completed"
                    ]
                ),
                format_score(
                    metrics[
                        "micro_average_precision"
                    ]
                ),
                format_score(
                    metrics[
                        "mean_reciprocal_rank"
                    ]
                ),
                format_score(
                    metrics[
                        "top_k"
                    ]["5"]["hit_rate"]
                ),
            ]
        )

    lines.extend(
        markdown_table(
            [
                "Task",
                "Selected epoch",
                "Epochs completed",
                "Valid micro AP",
                "Valid MRR",
                "Valid HitRate@5",
            ],
            fine_tuning_rows,
        )
    )

    lines.extend(
        [
            "",
            (
                "The reaction-center solvent and catalyst "
                "runs selected epoch 80, the search boundary. "
                "Their training loss continued to decrease "
                "while validation loss increased, indicating "
                "ranking improvement accompanied by worsening "
                "probability calibration and overfitting risk."
            ),
            "",
            "## Final condition-classification results",
            "",
            (
                "All hyperparameters and epoch counts were "
                "selected using validation data. Final models "
                "were refitted on train plus validation and "
                "evaluated once on the untouched test split."
            ),
            "",
        ]
    )

    model_labels = {
        "morgan_logistic": (
            "Morgan + logistic"
        ),
        "frozen_rxnfp": (
            "Frozen RXNFP"
        ),
        "fine_tuned_rxnfp": (
            "Fine-tuned RXNFP"
        ),
    }

    final_rows = []

    for _, row in comparison.iterrows():
        final_rows.append(
            [
                task_name(
                    row["protocol"],
                    row["target"],
                ),
                model_labels[row["model"]],
                format_score(
                    row["test_micro_ap"]
                ),
                format_score(
                    row["test_mrr"]
                ),
                format_score(
                    row[
                        "test_hit_rate_at_5"
                    ]
                ),
                format_score(
                    row[
                        "test_recall_at_5"
                    ]
                ),
            ]
        )

    lines.extend(
        markdown_table(
            [
                "Task",
                "Model",
                "Test micro AP",
                "Test MRR",
                "Test HitRate@5",
                "Test recall@5",
            ],
            final_rows,
        )
    )

    lines.extend(
        [
            "",
            (
                f"Morgan + logistic achieved the highest "
                f"test micro AP on {morgan_wins}/4 tasks. "
                "Neither frozen embeddings nor partial "
                "fine-tuning consistently improved condition "
                "classification."
            ),
            "",
            (
                "This negative result is informative: with "
                "roughly 170–270 training reactions per task "
                "and highly sparse catalyst labels, the "
                "lower-capacity Morgan baseline generalizes "
                "more reliably than the pretrained Transformer."
            ),
            "",
            "## Similar-reaction retrieval",
            "",
            (
                "Exact cosine similarity search was evaluated "
                "using frozen RXNFP embeddings. Pooling was "
                "selected on train-to-validation retrieval "
                "only; test labels were not used. CLS pooling "
                f"was selected with validation score "
                f"{retrieval_validation['pooling_scores']['cls']:.4f} "
                "versus "
                f"{retrieval_validation['pooling_scores']['masked_mean']:.4f} "
                "for masked-mean pooling."
            ),
            "",
        ]
    )

    retrieval_rows = []

    for _, row in retrieval_test.iterrows():
        retrieval_rows.append(
            [
                row["protocol"],
                str(int(row["index_reactions"])),
                str(int(row["test_queries"])),
                format_score(
                    row[
                        "reaction_type_hit_at_5"
                    ]
                ),
                format_score(
                    row[
                        "reaction_type_mrr_at_10"
                    ]
                ),
                format_score(
                    row[
                        "solvent_recall_at_5"
                    ]
                ),
                format_score(
                    row[
                        "catalyst_recall_at_5"
                    ]
                ),
                format_score(
                    row[
                        "nearest_similarity_mean"
                    ]
                ),
            ]
        )

    lines.extend(
        markdown_table(
            [
                "Protocol",
                "Index reactions",
                "Test queries",
                "Type Hit@5",
                "Type MRR@10",
                "Solvent recall@5",
                "Catalyst recall@5",
                "Mean nearest similarity",
            ],
            retrieval_rows,
        )
    )

    lines.extend(
        [
            "",
            (
                "Retrieval is the strongest Day 5 Transformer "
                "result. Under the transformation split, the "
                "five nearest precedents recover solvent labels "
                "with 0.8601 recall and catalyst labels with "
                "0.7695 recall."
            ),
            "",
            (
                "The end-to-end retrieval interface accepts "
                "reactant and product SMILES, canonicalizes the "
                "reaction, generates an RXNFP CLS embedding, "
                "retrieves train-plus-validation precedents, "
                "and attaches the best historically observed "
                "condition record for each neighbor."
            ),
            "",
            "## Conclusions",
            "",
            (
                "1. The Day 4 Morgan classifier remains the "
                "preferred direct condition-ranking model."
            ),
            (
                "2. Partial RXNFP fine-tuning overfits the small "
                "multi-label training set and does not improve "
                "untouched-test performance."
            ),
            (
                "3. Frozen RXNFP embeddings provide strong "
                "similar-reaction retrieval and useful "
                "evidence for downstream recommendations."
            ),
            (
                "4. Retrieval similarity is not a probability "
                "of reaction success, and retrieved conditions "
                "are precedents rather than guaranteed optimal "
                "conditions."
            ),
            (
                "5. Solvent and catalyst predictions remain "
                "independent; joint condition ranking is not "
                "implemented."
            ),
            "",
            "## Generated artifacts",
            "",
            (
                "- `reports/day5/figures/"
                "day5_model_comparison.png`"
            ),
            (
                "- `reports/day5/figures/"
                "day5_fine_tuning_curves.png`"
            ),
            (
                "- `reports/day5/figures/"
                "day5_retrieval_metrics.png`"
            ),
            (
                "- `reports/day5/retrieval/"
                "retrieval_example.json`"
            ),
            (
                "- `reports/day5/classification/"
                "three_model_final_test_comparison.csv`"
            ),
            "",
        ]
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )

    print("Saved:", OUTPUT_PATH)
    print(
        "Lines:",
        len(lines),
    )


if __name__ == "__main__":
    main()