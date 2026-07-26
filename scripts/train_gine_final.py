"""Refit GINE on train plus validation and evaluate test."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from chempilot.data.graph_dataset import (
    MoleculeGraphDataset,
)
from chempilot.models.gine import (
    GINEConfig,
    GINERegressor,
)
from chempilot.training.gine_trainer import (
    GINETrainer,
    TrainerConfig,
    set_global_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
T_CRITICAL_95_DF_2 = 4.302652729911275


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Refit GINE using the validation-selected "
            "epoch count and evaluate the fixed test set."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/gine.yaml",
    )
    parser.add_argument(
        "--protocol",
        choices=["random", "scaffold"],
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=(
            num_workers > 0
        ),
    )


def finite_metrics(metrics: dict) -> dict:
    result = {}

    for key, value in metrics.items():
        numeric_value = float(value)

        result[key] = (
            numeric_value
            if np.isfinite(numeric_value)
            else None
        )

    return result


def save_predictions(
    prediction_result: dict,
    output_path: Path,
) -> None:
    y_true = np.asarray(
        prediction_result["y_true"],
        dtype=np.float64,
    )
    y_pred = np.asarray(
        prediction_result["y_pred"],
        dtype=np.float64,
    )

    table = pd.DataFrame({
        "sample_id": (
            prediction_result["sample_ids"]
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": y_pred - y_true,
        "absolute_error": np.abs(
            y_pred - y_true
        ),
    })

    table = table.sort_values(
        "absolute_error",
        ascending=False,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    table.to_csv(
        output_path,
        index=False,
    )


def calculate_confidence_interval(
    values,
) -> dict:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(values.mean())

    if len(values) < 2:
        standard_deviation = 0.0
        half_width = 0.0
    else:
        standard_deviation = float(
            values.std(ddof=1)
        )
        half_width = float(
            T_CRITICAL_95_DF_2
            * standard_deviation
            / math.sqrt(len(values))
        )

    return {
        "mean": mean,
        "standard_deviation": (
            standard_deviation
        ),
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "ci95_half_width": half_width,
        "number_of_seeds": len(values),
    }


def main() -> None:
    arguments = parse_arguments()

    config_path = resolve_path(
        arguments.config
    )
    with config_path.open(
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    protocol = arguments.protocol

    seeds = (
        arguments.seeds
        if arguments.seeds is not None
        else config["experiment"]["final_seeds"]
    )
    seeds = [int(seed) for seed in seeds]

    if len(seeds) != 3:
        raise ValueError(
            "The final experiment requires exactly "
            "three predetermined seeds."
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "Final seeds must be unique."
        )

    if (
        arguments.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    device = torch.device(
        arguments.device
    )

    development_summary_path = (
        PROJECT_ROOT
        / "reports/day3/"
          f"{protocol}_gine_development.json"
    )

    with development_summary_path.open(
        encoding="utf-8"
    ) as file:
        development_summary = json.load(file)

    selected_epochs = int(
        development_summary["best_epoch"]
    )

    protocol_paths = (
        config["data"]["protocols"][protocol]
    )

    dataset = MoleculeGraphDataset(
        cache_path=resolve_path(
            config["data"]["graph_cache"]
        ),
        metadata_path=resolve_path(
            config["data"]["metadata"]
        ),
        manifest_path=(
            PROJECT_ROOT
            / "reports/"
              "graph_feature_manifest.json"
        ),
    )

    train_dataset = dataset.subset_from_split(
        resolve_path(
            protocol_paths["train"]
        )
    )
    valid_dataset = dataset.subset_from_split(
        resolve_path(
            protocol_paths["valid"]
        )
    )
    test_dataset = dataset.subset_from_split(
        resolve_path(
            protocol_paths["test"]
        )
    )

    final_train_dataset = ConcatDataset([
        train_dataset,
        valid_dataset,
    ])

    loader_config = config["loader"]
    batch_size = int(
        loader_config["batch_size"]
    )
    num_workers = int(
        loader_config["num_workers"]
    )
    pin_memory = bool(
        loader_config["pin_memory"]
        and device.type == "cuda"
    )

    checkpoint_directory = resolve_path(
        config["outputs"][
            "checkpoint_directory"
        ]
    )
    report_directory = resolve_path(
        config["outputs"][
            "report_directory"
        ]
    )
    prediction_directory = resolve_path(
        config["outputs"][
            "prediction_directory"
        ]
    )
    history_directory = resolve_path(
        config["outputs"][
            "history_directory"
        ]
    )

    for directory in [
        checkpoint_directory,
        report_directory,
        prediction_directory,
        history_directory,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Final fixed-epoch refit")
    print("-----------------------")
    print("Protocol:", protocol)
    print("Device:", device)
    print("Selected epochs:", selected_epochs)
    print(
        "Final train samples:",
        len(final_train_dataset),
    )
    print("Test samples:", len(test_dataset))
    print("Seeds:", seeds)

    seed_results = []

    for seed in seeds:
        print(f"\nStarting seed {seed}")

        set_global_seed(seed)

        train_loader = make_loader(
            dataset=final_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        test_loader = make_loader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )

        model_config = GINEConfig(
            hidden_dim=int(
                config["model"]["hidden_dim"]
            ),
            num_layers=int(
                config["model"]["num_layers"]
            ),
            dropout=float(
                config["model"]["dropout"]
            ),
            pooling=str(
                config["model"]["pooling"]
            ),
        )

        # Fix the seed before parameter initialization.
        set_global_seed(seed)
        model = GINERegressor(
            model_config
        )

        training_options = dict(
            config["training"]
        )
        training_options["seed"] = seed

        trainer = GINETrainer(
            model=model,
            trainer_config=TrainerConfig(
                **training_options
            ),
            device=device,
        )

        checkpoint_path = (
            checkpoint_directory
            / f"{protocol}_final_seed_{seed}.pt"
        )
        history_path = (
            history_directory
            / f"{protocol}_gine_"
              f"final_seed_{seed}.csv"
        )
        prediction_path = (
            prediction_directory
            / f"{protocol}_gine_"
              f"final_seed_{seed}_test.csv"
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        fit_result = trainer.fit_fixed_epochs(
            train_loader=train_loader,
            number_of_epochs=selected_epochs,
            checkpoint_path=checkpoint_path,
        )

        test_result = trainer.predict(
            test_loader
        )

        test_metrics = finite_metrics(
            test_result["metrics"]
        )

        save_predictions(
            prediction_result=test_result,
            output_path=prediction_path,
        )

        pd.DataFrame(
            fit_result["history"]
        ).to_csv(
            history_path,
            index=False,
        )

        peak_gpu_memory_mb = None
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_gpu_memory_mb = float(
                torch.cuda.max_memory_allocated()
                / 1024**2
            )

        seed_result = {
            "protocol": protocol,
            "seed": seed,
            "selected_epochs": (
                selected_epochs
            ),
            "train_samples": len(
                final_train_dataset
            ),
            "test_samples": len(
                test_dataset
            ),
            "parameter_count": (
                model.parameter_count()
            ),
            "training_seconds": float(
                fit_result[
                    "training_seconds"
                ]
            ),
            "peak_gpu_memory_mb": (
                peak_gpu_memory_mb
            ),
            **{
                f"test_{key}": value
                for key, value
                in test_metrics.items()
            },
            "checkpoint": str(
                checkpoint_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "history_file": str(
                history_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "prediction_file": str(
                prediction_path.relative_to(
                    PROJECT_ROOT
                )
            ),
        }

        seed_results.append(seed_result)

        print(
            f"Seed {seed}:",
            f"MAE={test_metrics['mae']:.4f}",
            f"RMSE={test_metrics['rmse']:.4f}",
            f"R2={test_metrics['r2']:.4f}",
            (
                "Spearman="
                f"{test_metrics['spearman']:.4f}"
            ),
            (
                "seconds="
                f"{fit_result['training_seconds']:.2f}"
            ),
        )

    results_table = pd.DataFrame(
        seed_results
    )

    results_csv_path = (
        report_directory
        / f"{protocol}_gine_final_results.csv"
    )
    results_table.to_csv(
        results_csv_path,
        index=False,
    )

    aggregate = {}

    for metric in [
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_spearman",
        "training_seconds",
    ]:
        aggregate[metric] = (
            calculate_confidence_interval(
                results_table[metric].to_numpy()
            )
        )

    summary = {
        "stage": "final_refit",
        "protocol": protocol,
        "development_summary": str(
            development_summary_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "epoch_selection_rule": (
            "Best validation MAE from the "
            "seed-42 development run."
        ),
        "selected_epochs": selected_epochs,
        "final_train_samples": len(
            final_train_dataset
        ),
        "test_samples": len(test_dataset),
        "seeds": seeds,
        "model_config": (
            model_config.to_dict()
        ),
        "individual_results": seed_results,
        "aggregate": aggregate,
        "test_usage": (
            "The fixed test set was used only "
            "for final evaluation and did not "
            "affect model selection."
        ),
    }

    summary_path = (
        report_directory
        / f"{protocol}_gine_final_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nAggregate result")
    print("----------------")

    for metric in [
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_spearman",
    ]:
        statistics = aggregate[metric]
        print(
            f"{metric}: "
            f"{statistics['mean']:.4f} "
            f"± {statistics['ci95_half_width']:.4f} "
            f"(95% CI)"
        )

    print("Results:", results_csv_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()