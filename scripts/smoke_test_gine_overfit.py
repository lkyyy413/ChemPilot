"""Overfit a tiny dataset to validate the GINE training pipeline."""

import json
from pathlib import Path

import numpy as np
import torch
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

GRAPH_CACHE = (
    PROJECT_ROOT
    / "data/processed/solubility_aqsoldb_graphs.pt"
)
PROCESSED_DATA = (
    PROJECT_ROOT
    / "data/processed/solubility_aqsoldb_processed.csv"
)
TRAIN_SPLIT = (
    PROJECT_ROOT
    / "data/splits/random/seed_42/train.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts/models/day3/"
      "gine_smoke_overfit.pt"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports/day3/"
      "gine_smoke_overfit.json"
)


def main() -> None:
    seed = 7
    number_of_samples = 32

    set_global_seed(seed)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = MoleculeGraphDataset(
        cache_path=GRAPH_CACHE,
        metadata_path=PROCESSED_DATA,
    )

    train_split = dataset.subset_from_split(
        TRAIN_SPLIT
    )

    tiny_dataset = [
        train_split[index]
        for index in range(number_of_samples)
    ]

    # The same samples are deliberately used for training
    # and validation. This is only an overfitting diagnostic.
    train_loader = DataLoader(
        tiny_dataset,
        batch_size=number_of_samples,
        shuffle=False,
        num_workers=0,
    )
    evaluation_loader = DataLoader(
        tiny_dataset,
        batch_size=number_of_samples,
        shuffle=False,
        num_workers=0,
    )

    model_config = GINEConfig(
        hidden_dim=128,
        num_layers=4,
        dropout=0.0,
        pooling="add",
    )

    model = GINERegressor(model_config)

    trainer_config = TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=500,
        patience=100,
        min_delta=1e-5,
        gradient_clip_norm=5.0,
        scheduler_factor=0.5,
        scheduler_patience=30,
        minimum_learning_rate=1e-6,
        huber_delta=1.0,
        seed=seed,
    )

    trainer = GINETrainer(
        model=model,
        trainer_config=trainer_config,
        device=device,
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = np.array(
        [
            float(graph.y.item())
            for graph in tiny_dataset
        ],
        dtype=np.float64,
    )

    mean_baseline = np.full_like(
        targets,
        targets.mean(),
    )
    baseline_mae = float(
        np.mean(
            np.abs(targets - mean_baseline)
        )
    )

    print("Samples:", number_of_samples)
    print("Parameters:", model.parameter_count())
    print("Target mean:", round(targets.mean(), 4))
    print("Target standard deviation:", round(
        targets.std(),
        4,
    ))
    print("Mean-baseline MAE:", round(
        baseline_mae,
        4,
    ))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    fit_result = trainer.fit(
        train_loader=train_loader,
        valid_loader=evaluation_loader,
        checkpoint_path=CHECKPOINT_PATH,
    )

    prediction_result = trainer.predict(
        evaluation_loader
    )

    y_true = np.asarray(
        prediction_result["y_true"],
        dtype=np.float64,
    )
    y_pred = np.asarray(
        prediction_result["y_pred"],
        dtype=np.float64,
    )

    absolute_errors = np.abs(y_true - y_pred)

    final_mae = float(absolute_errors.mean())
    final_rmse = float(
        np.sqrt(
            np.mean((y_true - y_pred) ** 2)
        )
    )
    maximum_absolute_error = float(
        absolute_errors.max()
    )

    peak_gpu_memory_mb = None

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_gpu_memory_mb = float(
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    passed = (
        final_mae < 0.35
        and final_mae < baseline_mae * 0.35
    )

    report = {
        "purpose": (
            "Pipeline overfitting diagnostic; "
            "not a benchmark result."
        ),
        "seed": seed,
        "device": str(device),
        "number_of_samples": number_of_samples,
        "parameter_count": (
            model.parameter_count()
        ),
        "mean_baseline_mae": baseline_mae,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "maximum_absolute_error": (
            maximum_absolute_error
        ),
        "peak_gpu_memory_mb": (
            peak_gpu_memory_mb
        ),
        "checkpoint_path": str(
            CHECKPOINT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "fit_result": {
            "best_epoch": fit_result["best_epoch"],
            "best_valid_mae": (
                fit_result["best_valid_mae"]
            ),
            "epochs_completed": (
                fit_result["epochs_completed"]
            ),
            "training_seconds": (
                fit_result["training_seconds"]
            ),
            "target_standardizer": (
                fit_result["target_standardizer"]
            ),
            "checkpoint_valid_metrics": (
                fit_result["checkpoint_valid_metrics"]
            ),
            "history_tail": (
                fit_result["history"][-5:]
            ),
        },
        "passed": passed,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nOverfitting diagnostic")
    print("----------------------")
    print("Final MAE:", round(final_mae, 4))
    print("Final RMSE:", round(final_rmse, 4))
    print(
        "Maximum absolute error:",
        round(maximum_absolute_error, 4),
    )
    print(
        "Peak GPU memory MB:",
        None
        if peak_gpu_memory_mb is None
        else round(peak_gpu_memory_mb, 2),
    )
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Report:", REPORT_PATH)
    print("Diagnostic passed:", passed)

    if not passed:
        raise RuntimeError(
            "The model did not sufficiently overfit "
            "the 32-sample diagnostic dataset."
        )


if __name__ == "__main__":
    main()