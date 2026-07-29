"""Development fine-tuning for reaction condition prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

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


PROTOCOLS = (
    "transformation",
    "reaction_center",
)

TARGETS = (
    "solvent",
    "catalyst",
)

TARGET_ROOT = Path(
    "data/processed/reactions/targets"
)

METADATA_PATH = Path(
    "data/processed/reactions/"
    "features/day5/"
    "rxnfp_reaction_metadata.parquet"
)

CHECKPOINT_DIRECTORY = Path(
    "artifacts/pretrained/day5/"
    "rxnfp_bert_pretrained"
)

MODEL_ROOT = Path(
    "artifacts/models/day5/"
    "fine_tuned"
)

REPORT_ROOT = Path(
    "reports/day5/fine_tuning"
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        choices=(
            "all",
            *PROTOCOLS,
        ),
        default="all",
    )

    parser.add_argument(
        "--target",
        choices=(
            "all",
            *TARGETS,
        ),
        default="all",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--maximum-positive-weight",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_vocabulary(
    path: Path,
) -> list[str]:
    with path.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)[
            "labels"
        ]


def build_dataset(
    samples: pd.DataFrame,
    targets,
    indices: np.ndarray,
    sequence_by_transformation,
    tokenizer,
    max_length: int,
) -> TensorDataset:
    selected = samples.iloc[
        indices
    ]

    reactions = selected[
        "transformation_signature"
    ].map(
        sequence_by_transformation
    )

    if reactions.isna().any():
        raise RuntimeError(
            "Missing canonical reaction "
            "sequence."
        )

    encoded = (
        tokenizer.tokenize_reactions(
            reactions.tolist(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    )

    labels = torch.as_tensor(
        targets[
            indices
        ].toarray(),
        dtype=torch.float32,
    )

    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        encoded["token_type_ids"],
        labels,
    )


def make_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=(
            generator
            if shuffle
            else None
        ),
        num_workers=0,
        pin_memory=True,
    )


def unpack_batch(
    batch,
    device,
):
    (
        input_ids,
        attention_mask,
        token_type_ids,
        labels,
    ) = batch

    return (
        {
            "input_ids": input_ids.to(
                device,
                non_blocking=True,
            ),
            "attention_mask": (
                attention_mask.to(
                    device,
                    non_blocking=True,
                )
            ),
            "token_type_ids": (
                token_type_ids.to(
                    device,
                    non_blocking=True,
                )
            ),
        },
        labels.to(
            device,
            non_blocking=True,
        ),
    )


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    losses = []
    true_batches = []
    score_batches = []

    with torch.inference_mode():
        for batch in loader:
            inputs, labels = (
                unpack_batch(
                    batch,
                    device,
                )
            )

            logits = model(**inputs)
            loss = criterion(
                logits,
                labels,
            )

            losses.append(
                (
                    float(loss)
                    * len(labels)
                )
            )

            true_batches.append(
                labels.cpu().numpy()
            )

            score_batches.append(
                torch.sigmoid(
                    logits
                ).cpu().numpy()
            )

    y_true = np.concatenate(
        true_batches,
        axis=0,
    )

    y_score = np.concatenate(
        score_batches,
        axis=0,
    )

    mean_loss = (
        sum(losses)
        / len(y_true)
    )

    metrics = multilabel_metrics(
        y_true,
        y_score,
        top_k_values=(
            1,
            3,
            5,
        ),
    )

    return (
        mean_loss,
        y_true,
        y_score,
        metrics,
    )


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    gradient_clip: float,
):
    model.train()

    total_loss = 0.0
    number_of_samples = 0
    maximum_gradient_norm = 0.0

    for batch in loader:
        inputs, labels = unpack_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(**inputs)

        loss = criterion(
            logits,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Nonfinite training loss."
            )

        loss.backward()

        gradient_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                [
                    parameter
                    for parameter in (
                        model.parameters()
                    )
                    if (
                        parameter
                        .requires_grad
                    )
                ],
                max_norm=gradient_clip,
            )
        )

        if not torch.isfinite(
            gradient_norm
        ):
            raise RuntimeError(
                "Nonfinite gradient norm."
            )

        optimizer.step()

        batch_size = len(labels)

        total_loss += (
            float(loss.detach())
            * batch_size
        )

        number_of_samples += batch_size

        maximum_gradient_norm = max(
            maximum_gradient_norm,
            float(gradient_norm),
        )

    return (
        total_loss
        / number_of_samples,
        maximum_gradient_norm,
    )


def popularity_scores(
    y_train,
    number_of_samples: int,
):
    prevalence = np.asarray(
        y_train.mean(axis=0)
    ).reshape(-1)

    return np.broadcast_to(
        prevalence,
        (
            number_of_samples,
            len(prevalence),
        ),
    ).copy()


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

        records.append(
            {
                "transformation_signature": (
                    samples.iloc[
                        row_index
                    ][
                        "transformation_signature"
                    ]
                ),
                "reaction_center_signature": (
                    samples.iloc[
                        row_index
                    ][
                        "reaction_center_signature"
                    ]
                ),
                "reaction_type": (
                    samples.iloc[
                        row_index
                    ]["reaction_type"]
                ),
                "split": "valid",
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
            }
        )

    return pd.DataFrame(records)


def train_one(
    protocol: str,
    target: str,
    arguments,
    metadata,
    tokenizer,
):
    set_seed(arguments.seed)

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

    train_mask = (
        samples["split"].eq("train")
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    valid_mask = (
        samples["split"].eq("valid")
        & samples[
            "has_any_known_target"
        ]
    ).to_numpy()

    train_indices = np.flatnonzero(
        train_mask
    )

    valid_indices = np.flatnonzero(
        valid_mask
    )

    train_dataset = build_dataset(
        samples,
        targets,
        train_indices,
        sequence_by_transformation,
        tokenizer,
        arguments.max_length,
    )

    valid_dataset = build_dataset(
        samples,
        targets,
        valid_indices,
        sequence_by_transformation,
        tokenizer,
        arguments.max_length,
    )

    train_loader = make_loader(
        train_dataset,
        arguments.batch_size,
        True,
        arguments.seed,
    )

    valid_loader = make_loader(
        valid_dataset,
        arguments.batch_size,
        False,
        arguments.seed,
    )

    y_train = targets[
        train_indices
    ]

    positive_counts = np.asarray(
        y_train.sum(axis=0)
    ).reshape(-1)

    negative_counts = (
        len(train_indices)
        - positive_counts
    )

    raw_positive_weights = (
        negative_counts
        / positive_counts
    )

    clipped_weights = np.minimum(
        raw_positive_weights,
        arguments
        .maximum_positive_weight,
    )

    device = torch.device(
        arguments.device
    )

    positive_weight_tensor = (
        torch.as_tensor(
            clipped_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    model = (
        ReactionTransformerMultiLabelClassifier(
            ReactionTransformerClassifierConfig(
                number_of_labels=(
                    len(vocabulary)
                ),
                checkpoint_directory=(
                    CHECKPOINT_DIRECTORY
                ),
                pooling="masked_mean",
                unfreeze_last_n_layers=2,
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
                "lr": (
                    arguments
                    .encoder_learning_rate
                ),
            },
            {
                "params": head_parameters,
                "lr": (
                    arguments
                    .head_learning_rate
                ),
            },
        ],
        weight_decay=(
            arguments.weight_decay
        ),
    )

    model_directory = (
        MODEL_ROOT
        / protocol
    )

    report_directory = (
        REPORT_ROOT
        / protocol
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
        / f"{target}_development.pt"
    )

    history_path = (
        report_directory
        / f"{target}_development_history.csv"
    )

    metrics_path = (
        report_directory
        / f"{target}_development_metrics.json"
    )

    predictions_path = (
        report_directory
        / f"{target}_development_valid_top5.csv"
    )

    best_micro_ap = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    started = time.perf_counter()

    for epoch in range(
        1,
        arguments.max_epochs + 1,
    ):
        train_loss, maximum_gradient = (
            train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                arguments.gradient_clip,
            )
        )

        (
            valid_loss,
            _,
            _,
            valid_metrics,
        ) = evaluate(
            model,
            valid_loader,
            criterion,
            device,
        )

        micro_ap = valid_metrics[
            "micro_average_precision"
        ]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_micro_ap": micro_ap,
            "valid_mrr": valid_metrics[
                "mean_reciprocal_rank"
            ],
            "valid_hit_rate_at_5": (
                valid_metrics[
                    "top_k"
                ]["5"]["hit_rate"]
            ),
            "valid_recall_at_5": (
                valid_metrics[
                    "top_k"
                ]["5"]["recall"]
            ),
            "maximum_gradient_norm": (
                maximum_gradient
            ),
        }

        history.append(record)

        print(
            f"  epoch={epoch:02d}",
            f"train_loss={train_loss:.4f}",
            f"valid_loss={valid_loss:.4f}",
            f"micro_AP={micro_ap:.4f}",
            (
                "HitRate@5="
                f"{record['valid_hit_rate_at_5']:.4f}"
            ),
        )

        if micro_ap > (
            best_micro_ap + 1e-6
        ):
            best_micro_ap = micro_ap
            best_epoch = epoch
            epochs_without_improvement = 0

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
                    "epoch": best_epoch,
                    "validation_micro_ap": (
                        best_micro_ap
                    ),
                    "pooling": (
                        "masked_mean"
                    ),
                    "unfreeze_last_n_layers": 2,
                    "number_of_labels": (
                        len(vocabulary)
                    ),
                    "seed": arguments.seed,
                },
                model_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= arguments.patience
        ):
            print(
                "  Early stopping at "
                f"epoch {epoch}."
            )
            break

    elapsed = (
        time.perf_counter()
        - started
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    (
        best_valid_loss,
        y_valid,
        best_scores,
        best_metrics,
    ) = evaluate(
        model,
        valid_loader,
        criterion,
        device,
    )

    frequency_scores = popularity_scores(
        y_train,
        len(valid_indices),
    )

    frequency_metrics = (
        multilabel_metrics(
            targets[
                valid_indices
            ],
            frequency_scores,
            top_k_values=(
                1,
                3,
                5,
            ),
        )
    )

    valid_samples = (
        samples.iloc[
            valid_indices
        ].reset_index(drop=True)
    )

    top_prediction_table(
        valid_samples,
        y_valid,
        best_scores,
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
        "stage": "development",
        "training_splits": ["train"],
        "evaluation_split": "valid",
        "test_labels_used": False,
        "model": (
            "partially_fine_tuned_"
            "rxnfp_bert"
        ),
        "pooling": "masked_mean",
        "classes": len(vocabulary),
        "train_samples": len(
            train_indices
        ),
        "valid_samples": len(
            valid_indices
        ),
        "best_epoch": best_epoch,
        "epochs_completed": len(
            history
        ),
        "best_valid_loss": (
            best_valid_loss
        ),
        "best_validation_metrics": (
            best_metrics
        ),
        "frequency_baseline_metrics": (
            frequency_metrics
        ),
        "configuration": {
            "max_length": (
                arguments.max_length
            ),
            "batch_size": (
                arguments.batch_size
            ),
            "maximum_epochs": (
                arguments.max_epochs
            ),
            "patience": (
                arguments.patience
            ),
            "encoder_learning_rate": (
                arguments
                .encoder_learning_rate
            ),
            "head_learning_rate": (
                arguments
                .head_learning_rate
            ),
            "weight_decay": (
                arguments.weight_decay
            ),
            "maximum_positive_weight": (
                arguments
                .maximum_positive_weight
            ),
            "gradient_clip": (
                arguments.gradient_clip
            ),
            "seed": arguments.seed,
            "unfreeze_last_n_layers": 2,
        },
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
                    > arguments
                    .maximum_positive_weight
                ).sum()
            ),
        },
        "parameter_summary": (
            model.parameter_summary()
        ),
        "training_seconds": elapsed,
        "source_metadata_sha256": (
            sha256_file(
                METADATA_PATH
            )
        ),
        "model_path": str(model_path),
        "model_sha256": (
            sha256_file(model_path)
        ),
        "history_path": str(
            history_path
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
        "  selected epoch:",
        best_epoch,
    )
    print(
        "  selected micro AP:",
        round(
            best_metrics[
                "micro_average_precision"
            ],
            4,
        ),
    )
    print(
        "  saved:",
        model_path,
    )

    del model
    torch.cuda.empty_cache()

    return result


def main() -> None:
    arguments = parse_arguments()

    protocols = (
        PROTOCOLS
        if arguments.protocol == "all"
        else (
            arguments.protocol,
        )
    )

    targets = (
        TARGETS
        if arguments.target == "all"
        else (
            arguments.target,
        )
    )

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    tokenizer = ReactionSmilesTokenizer(
        CHECKPOINT_DIRECTORY
        / "vocab.txt"
    )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Reaction Transformer development"
    )
    print("-------------------------------")
    print(
        "Device:",
        arguments.device,
    )
    print(
        "Test labels used: False"
    )

    results = []

    for protocol in protocols:
        for target in targets:
            print(
                "\n"
                + f"{protocol} | {target}"
            )

            results.append(
                train_one(
                    protocol,
                    target,
                    arguments,
                    metadata,
                    tokenizer,
                )
            )

    summary_path = (
        REPORT_ROOT
        / "development_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "created_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "test_labels_used": False,
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()