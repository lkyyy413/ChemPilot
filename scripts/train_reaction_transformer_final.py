"""Final fixed-epoch RXNFP training and one-time test evaluation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from chempilot.evaluation.multilabel import (
    multilabel_metrics,
)
from chempilot.reactions.tokenization import (
    ReactionSmilesTokenizer,
)
from chempilot.reactions.transformer_classifier import (
    ReactionTransformerClassifierConfig,
    ReactionTransformerMultiLabelClassifier,
)

from train_reaction_transformer_development import (
    build_dataset,
    evaluate,
    load_vocabulary,
    make_loader,
    popularity_scores,
    set_seed,
    sha256_file,
    train_epoch,
)


TARGET_ROOT = Path(
    "data/processed/reactions/targets"
)

METADATA_PATH = Path(
    "data/processed/reactions/"
    "features/day5/"
    "rxnfp_reaction_metadata.parquet"
)

PRETRAINED_DIRECTORY = Path(
    "artifacts/pretrained/day5/"
    "rxnfp_bert_pretrained"
)

PLAN_PATH = Path(
    "reports/day5/fine_tuning/"
    "final_training_plan.json"
)

MODEL_ROOT = Path(
    "artifacts/models/day5/"
    "fine_tuned_final"
)

REPORT_ROOT = Path(
    "reports/day5/fine_tuning/final"
)

SUMMARY_JSON_PATH = (
    REPORT_ROOT
    / "final_test_summary.json"
)

SUMMARY_CSV_PATH = (
    REPORT_ROOT
    / "final_test_results.csv"
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def top_prediction_table(
    samples,
    y_true,
    y_score,
    vocabulary,
    k=5,
):
    order = np.argsort(
        -y_score,
        axis=1,
    )[:, :k]

    records = []

    for row_index in range(
        len(samples)
    ):
        true_indices = np.flatnonzero(
            y_true[row_index]
        )

        predicted_indices = order[
            row_index
        ]

        sample = samples.iloc[
            row_index
        ]

        records.append(
            {
                "transformation_signature": (
                    sample[
                        "transformation_signature"
                    ]
                ),
                "reaction_center_signature": (
                    sample[
                        "reaction_center_signature"
                    ]
                ),
                "reaction_type": (
                    sample["reaction_type"]
                ),
                "split": "test",
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
                "top_scores": (
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
                    sample[
                        "known_target_count"
                    ]
                ),
                "unknown_target_count": int(
                    sample[
                        "unknown_target_count"
                    ]
                ),
                "all_targets_known": bool(
                    sample[
                        "all_targets_known"
                    ]
                ),
            }
        )

    return pd.DataFrame(records)


def train_and_evaluate_task(
    task_plan,
    metadata,
    tokenizer,
    device,
):
    protocol = task_plan[
        "protocol"
    ]

    target = task_plan["target"]

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

    sequence_by_transformation = (
        metadata.set_index(
            "transformation_signature"
        )["canonical_reaction"]
    )

    train_valid_mask = (
        samples["split"].isin(
            [
                "train",
                "valid",
            ]
        )
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    test_mask = (
        samples["split"].eq("test")
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    train_valid_indices = (
        np.flatnonzero(
            train_valid_mask
        )
    )

    test_indices = np.flatnonzero(
        test_mask
    )

    train_valid_dataset = (
        build_dataset(
            samples,
            targets,
            train_valid_indices,
            sequence_by_transformation,
            tokenizer,
            task_plan[
                "max_length"
            ],
        )
    )

    test_dataset = build_dataset(
        samples,
        targets,
        test_indices,
        sequence_by_transformation,
        tokenizer,
        task_plan["max_length"],
    )

    train_valid_loader = (
        make_loader(
            train_valid_dataset,
            task_plan[
                "batch_size"
            ],
            True,
            task_plan["seed"],
        )
    )

    test_loader = make_loader(
        test_dataset,
        task_plan["batch_size"],
        False,
        task_plan["seed"],
    )

    y_train_valid = targets[
        train_valid_indices
    ]

    positive_counts = np.asarray(
        y_train_valid.sum(axis=0)
    ).reshape(-1)

    if np.any(
        positive_counts == 0
    ):
        raise RuntimeError(
            "A vocabulary class has no "
            "train+valid positives."
        )

    negative_counts = (
        len(train_valid_indices)
        - positive_counts
    )

    raw_positive_weights = (
        negative_counts
        / positive_counts
    )

    clipped_weights = np.minimum(
        raw_positive_weights,
        task_plan[
            "maximum_positive_weight"
        ],
    )

    positive_weight_tensor = (
        torch.as_tensor(
            clipped_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    set_seed(task_plan["seed"])

    model = (
        ReactionTransformerMultiLabelClassifier(
            ReactionTransformerClassifierConfig(
                number_of_labels=(
                    len(vocabulary)
                ),
                checkpoint_directory=(
                    PRETRAINED_DIRECTORY
                ),
                pooling=task_plan[
                    "pooling"
                ],
                unfreeze_last_n_layers=(
                    task_plan[
                        "unfreeze_last_n_layers"
                    ]
                ),
                dropout=0.1,
            )
        )
        .to(device)
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss(
            pos_weight=(
                positive_weight_tensor
            )
        )
    )

    encoder_parameters = [
        parameter
        for parameter in (
            model.encoder.parameters()
        )
        if parameter.requires_grad
    ]

    head_parameters = list(
        model.classifier.parameters()
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    encoder_parameters
                ),
                "lr": task_plan[
                    "encoder_learning_rate"
                ],
            },
            {
                "params": (
                    head_parameters
                ),
                "lr": task_plan[
                    "head_learning_rate"
                ],
            },
        ],
        weight_decay=task_plan[
            "weight_decay"
        ],
    )

    model_directory = (
        MODEL_ROOT / protocol
    )

    report_directory = (
        REPORT_ROOT / protocol
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_directory
        / f"{target}_final.pt"
    )

    metrics_path = (
        report_directory
        / f"{target}_test_metrics.json"
    )

    predictions_path = (
        report_directory
        / f"{target}_test_top5.csv"
    )

    selected_epochs = task_plan[
        "selected_epoch"
    ]

    print(
        "\n"
        + f"{protocol} | {target}"
    )
    print(
        "  train+valid samples:",
        len(train_valid_indices),
    )
    print(
        "  test evaluated samples:",
        len(test_indices),
    )
    print(
        "  fixed epochs:",
        selected_epochs,
    )

    started = time.perf_counter()

    final_train_loss = None
    maximum_gradient_seen = 0.0

    for epoch in range(
        1,
        selected_epochs + 1,
    ):
        (
            final_train_loss,
            maximum_gradient,
        ) = train_epoch(
            model,
            train_valid_loader,
            criterion,
            optimizer,
            device,
            task_plan[
                "gradient_clip"
            ],
        )

        maximum_gradient_seen = max(
            maximum_gradient_seen,
            maximum_gradient,
        )

        if (
            epoch == 1
            or epoch
            == selected_epochs
            or epoch % 10 == 0
        ):
            print(
                f"  epoch={epoch:02d}",
                (
                    "train_loss="
                    f"{final_train_loss:.4f}"
                ),
            )

    training_seconds = (
        time.perf_counter()
        - started
    )

    # Save the fixed model before examining
    # any test metric.
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in (
            model.state_dict().items()
        )
    }

    torch.save(
        {
            "model_state_dict": (
                state_dict
            ),
            "protocol": protocol,
            "target": target,
            "vocabulary": vocabulary,
            "stage": "final",
            "training_splits": [
                "train",
                "valid",
            ],
            "fixed_epochs": (
                selected_epochs
            ),
            "pooling": task_plan[
                "pooling"
            ],
            "unfreeze_last_n_layers": (
                task_plan[
                    "unfreeze_last_n_layers"
                ]
            ),
            "number_of_labels": (
                len(vocabulary)
            ),
            "seed": task_plan["seed"],
            "final_training_plan_sha256": (
                sha256_file(
                    PLAN_PATH
                )
            ),
        },
        model_path,
    )

    # One test evaluation after the model
    # has been fixed and saved.
    (
        test_loss,
        y_test,
        test_scores,
        test_metrics,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    frequency_scores = (
        popularity_scores(
            y_train_valid,
            len(test_indices),
        )
    )

    frequency_metrics = (
        multilabel_metrics(
            targets[
                test_indices
            ],
            frequency_scores,
            top_k_values=(
                1,
                3,
                5,
            ),
        )
    )

    test_samples = (
        samples.iloc[
            test_indices
        ].reset_index(drop=True)
    )

    top_prediction_table(
        test_samples,
        y_test,
        test_scores,
        vocabulary,
    ).to_csv(
        predictions_path,
        index=False,
    )

    result = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol": protocol,
        "target": target,
        "stage": "final",
        "model": (
            "partially_fine_tuned_"
            "rxnfp_bert"
        ),
        "training_splits": [
            "train",
            "valid",
        ],
        "evaluation_split": "test",
        "test_evaluations": 1,
        "test_labels_used_for_selection": (
            False
        ),
        "classes": len(vocabulary),
        "train_valid_samples": len(
            train_valid_indices
        ),
        "complete_test_samples": int(
            samples["split"].eq(
                "test"
            ).sum()
        ),
        "test_evaluated_samples": len(
            test_indices
        ),
        "all_targets_known": int(
            test_samples[
                "all_targets_known"
            ].sum()
        ),
        "contains_unknown_targets": int(
            (
                test_samples[
                    "unknown_target_count"
                ]
                > 0
            ).sum()
        ),
        "fixed_epochs": selected_epochs,
        "final_train_loss": (
            final_train_loss
        ),
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "frequency_baseline_metrics": (
            frequency_metrics
        ),
        "maximum_gradient_norm_seen": (
            maximum_gradient_seen
        ),
        "positive_weight_statistics": {
            "raw_maximum": float(
                raw_positive_weights.max()
            ),
            "clipped_maximum": float(
                clipped_weights.max()
            ),
            "classes_clipped": int(
                (
                    raw_positive_weights
                    > task_plan[
                        "maximum_positive_weight"
                    ]
                ).sum()
            ),
        },
        "parameter_summary": (
            model.parameter_summary()
        ),
        "configuration": task_plan,
        "training_seconds": (
            training_seconds
        ),
        "final_training_plan": str(
            PLAN_PATH
        ),
        "final_training_plan_sha256": (
            sha256_file(PLAN_PATH)
        ),
        "model_path": str(model_path),
        "model_sha256": (
            sha256_file(model_path)
        ),
        "prediction_path": str(
            predictions_path
        ),
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    print(
        "  test micro AP:",
        round(
            test_metrics[
                "micro_average_precision"
            ],
            4,
        ),
    )
    print(
        "  test MRR:",
        round(
            test_metrics[
                "mean_reciprocal_rank"
            ],
            4,
        ),
    )
    print(
        "  test HitRate@5:",
        round(
            test_metrics[
                "top_k"
            ]["5"]["hit_rate"],
            4,
        ),
    )

    del model
    torch.cuda.empty_cache()

    return result


def main() -> None:
    arguments = parse_arguments()

    if SUMMARY_JSON_PATH.exists():
        raise FileExistsError(
            "Final test summary already "
            "exists; refusing to repeat "
            "the final test evaluation: "
            f"{SUMMARY_JSON_PATH}"
        )

    with PLAN_PATH.open(
        encoding="utf-8"
    ) as file:
        plan = json.load(file)

    if plan["status"] != (
        "locked_before final "
        "test evaluation"
    ):
        raise RuntimeError(
            "Final plan is not locked."
        )

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    tokenizer = ReactionSmilesTokenizer(
        PRETRAINED_DIRECTORY
        / "vocab.txt"
    )

    device = torch.device(
        arguments.device
    )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Final Reaction Transformer"
    )
    print("--------------------------")
    print(
        "Training splits: train+valid"
    )
    print(
        "Evaluation split: test"
    )
    print(
        "Test evaluations per task: 1"
    )
    print("Device:", device)

    results = []

    for task_plan in plan["tasks"]:
        results.append(
            train_and_evaluate_task(
                task_plan,
                metadata,
                tokenizer,
                device,
            )
        )

    summary = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "final_training_plan": str(
            PLAN_PATH
        ),
        "final_training_plan_sha256": (
            sha256_file(PLAN_PATH)
        ),
        "test_labels_used_for_selection": (
            False
        ),
        "test_evaluations_per_task": 1,
        "results": results,
    }

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    rows = []

    for result in results:
        metrics = result[
            "test_metrics"
        ]

        rows.append(
            {
                "protocol": (
                    result["protocol"]
                ),
                "target": (
                    result["target"]
                ),
                "fixed_epochs": (
                    result[
                        "fixed_epochs"
                    ]
                ),
                "classes": (
                    result["classes"]
                ),
                "train_valid_samples": (
                    result[
                        "train_valid_samples"
                    ]
                ),
                "test_evaluated_samples": (
                    result[
                        "test_evaluated_samples"
                    ]
                ),
                "test_micro_ap": (
                    metrics[
                        "micro_average_precision"
                    ]
                ),
                "test_macro_ap_observed": (
                    metrics[
                        "macro_average_precision_"
                        "observed"
                    ]
                ),
                "test_mrr": (
                    metrics[
                        "mean_reciprocal_rank"
                    ]
                ),
                "test_hit_rate_at_1": (
                    metrics[
                        "top_k"
                    ]["1"]["hit_rate"]
                ),
                "test_hit_rate_at_3": (
                    metrics[
                        "top_k"
                    ]["3"]["hit_rate"]
                ),
                "test_hit_rate_at_5": (
                    metrics[
                        "top_k"
                    ]["5"]["hit_rate"]
                ),
                "test_recall_at_5": (
                    metrics[
                        "top_k"
                    ]["5"]["recall"]
                ),
                (
                    "frequency_hit_rate_at_5"
                ): (
                    result[
                        "frequency_baseline_metrics"
                    ][
                        "top_k"
                    ]["5"]["hit_rate"]
                ),
                "all_targets_known": (
                    result[
                        "all_targets_known"
                    ]
                ),
                (
                    "contains_unknown_targets"
                ): (
                    result[
                        "contains_unknown_targets"
                    ]
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        SUMMARY_CSV_PATH,
        index=False,
    )

    print("\nSaved:", SUMMARY_JSON_PATH)
    print("Saved:", SUMMARY_CSV_PATH)


if __name__ == "__main__":
    main()