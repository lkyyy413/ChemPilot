"""Run validation-based GINE development training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train GINE with validation-based "
            "early stopping."
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

    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)

    if candidate.is_absolute():
        return candidate

    return PROJECT_ROOT / candidate


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


def save_prediction_table(
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    table.to_csv(
        output_path,
        index=False,
    )


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
    seed = int(
        config["experiment"][
            "development_seed"
        ]
    )

    set_global_seed(seed)

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

    loader_config = config["loader"]
    pin_memory = bool(
        loader_config["pin_memory"]
        and device.type == "cuda"
    )

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=int(
            loader_config["batch_size"]
        ),
        shuffle=True,
        num_workers=int(
            loader_config["num_workers"]
        ),
        pin_memory=pin_memory,
        seed=seed,
    )
    valid_loader = make_loader(
        dataset=valid_dataset,
        batch_size=int(
            loader_config["batch_size"]
        ),
        shuffle=False,
        num_workers=int(
            loader_config["num_workers"]
        ),
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

    # The seed must be fixed before model creation
    # because initialization happens in __init__.
    set_global_seed(seed)
    model = GINERegressor(model_config)

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

    checkpoint_path = (
        checkpoint_directory
        / f"{protocol}_development_"
          f"seed_{seed}.pt"
    )
    summary_path = (
        report_directory
        / f"{protocol}_gine_"
          f"development.json"
    )
    history_path = (
        history_directory
        / f"{protocol}_gine_"
          f"development.csv"
    )
    prediction_path = (
        prediction_directory
        / f"{protocol}_gine_"
          f"development_valid.csv"
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("Development training")
    print("--------------------")
    print("Protocol:", protocol)
    print("Device:", device)
    print("Seed:", seed)
    print("Train samples:", len(train_dataset))
    print("Valid samples:", len(valid_dataset))
    print("Parameters:", model.parameter_count())
    print("Checkpoint:", checkpoint_path)

    fit_result = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        checkpoint_path=checkpoint_path,
    )

    validation_result = trainer.predict(
        valid_loader
    )

    save_prediction_table(
        prediction_result=validation_result,
        output_path=prediction_path,
    )

    history = pd.DataFrame(
        fit_result["history"]
    )
    history.to_csv(
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

    validation_metrics = {
        key: (
            None
            if not np.isfinite(value)
            else float(value)
        )
        for key, value in (
            validation_result["metrics"].items()
        )
    }

    summary = {
        "stage": "development",
        "protocol": protocol,
        "seed": seed,
        "device": str(device),
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "parameter_count": (
            model.parameter_count()
        ),
        "best_epoch": int(
            fit_result["best_epoch"]
        ),
        "best_valid_mae": float(
            fit_result["best_valid_mae"]
        ),
        "epochs_completed": int(
            fit_result["epochs_completed"]
        ),
        "training_seconds": float(
            fit_result["training_seconds"]
        ),
        "peak_gpu_memory_mb": (
            peak_gpu_memory_mb
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "model_config": (
            model_config.to_dict()
        ),
        "trainer_config": (
            trainer.trainer_config.to_dict()
        ),
        "target_standardizer": (
            fit_result["target_standardizer"]
        ),
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

    print("\nDevelopment result")
    print("------------------")
    print(
        "Best epoch:",
        summary["best_epoch"],
    )
    print(
        "Validation MAE:",
        round(
            validation_metrics["mae"],
            4,
        ),
    )
    print(
        "Validation RMSE:",
        round(
            validation_metrics["rmse"],
            4,
        ),
    )
    print(
        "Validation R2:",
        round(
            validation_metrics["r2"],
            4,
        ),
    )
    print(
        "Validation Spearman:",
        round(
            validation_metrics["spearman"],
            4,
        ),
    )
    print(
        "Training seconds:",
        round(
            summary["training_seconds"],
            2,
        ),
    )
    print(
        "Peak GPU memory MB:",
        None
        if peak_gpu_memory_mb is None
        else round(
            peak_gpu_memory_mb,
            2,
        ),
    )
    print("Summary:", summary_path)
    print("History:", history_path)


if __name__ == "__main__":
    main()