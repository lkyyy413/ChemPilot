"""Training utilities for molecular GINE regression."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chempilot.evaluation.regression import (
    regression_metrics,
)
from chempilot.models.gine import (
    GINEConfig,
    GINERegressor,
)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU, and CUDA generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class TargetStandardizer:
    """Training-label mean and standard deviation."""

    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError(
                "Target mean must be finite."
            )

        if (
            not np.isfinite(
                self.standard_deviation
            )
            or self.standard_deviation <= 0
        ):
            raise ValueError(
                "Target standard deviation must "
                "be positive and finite."
            )

    def transform(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return (
            values - self.mean
        ) / self.standard_deviation

    def inverse_transform(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return (
            values * self.standard_deviation
            + self.mean
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainerConfig:
    """Optimization and early-stopping configuration."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_epochs: int = 300
    patience: int = 30
    min_delta: float = 1e-4
    gradient_clip_norm: float = 5.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    minimum_learning_rate: float = 1e-6
    huber_delta: float = 1.0
    seed: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_target_standardizer(
    dataset,
) -> TargetStandardizer:
    """Calculate target statistics from training data only."""

    targets = torch.cat(
        [
            dataset[index].y.reshape(-1)
            for index in range(len(dataset))
        ]
    ).to(dtype=torch.float64)

    mean = float(targets.mean())
    standard_deviation = float(
        targets.std(unbiased=False)
    )

    return TargetStandardizer(
        mean=mean,
        standard_deviation=standard_deviation,
    )


class GINETrainer:
    """Train, validate, checkpoint, and evaluate GINE."""

    def __init__(
        self,
        model: GINERegressor,
        trainer_config: TrainerConfig,
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.trainer_config = trainer_config
        self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=trainer_config.learning_rate,
            weight_decay=(
                trainer_config.weight_decay
            ),
        )

        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=(
                    trainer_config.scheduler_factor
                ),
                patience=(
                    trainer_config.scheduler_patience
                ),
                min_lr=(
                    trainer_config.minimum_learning_rate
                ),
            )
        )

        self.target_standardizer: (
            TargetStandardizer | None
        ) = None

        self.history: list[dict] = []
        self.best_epoch: int | None = None
        self.best_valid_mae = float("inf")

    def _train_epoch(
        self,
        loader,
    ) -> dict[str, float]:
        if self.target_standardizer is None:
            raise RuntimeError(
                "Target standardizer is not initialized."
            )

        self.model.train()

        total_loss = 0.0
        total_absolute_error = 0.0
        total_samples = 0
        maximum_gradient_norm = 0.0

        for batch in loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad(
                set_to_none=True
            )

            standardized_predictions = (
                self.model(batch)
            )

            standardized_targets = (
                self.target_standardizer.transform(
                    batch.y
                )
            )

            loss = F.huber_loss(
                standardized_predictions,
                standardized_targets,
                delta=(
                    self.trainer_config.huber_delta
                ),
                reduction="mean",
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss detected."
                )

            loss.backward()

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=(
                        self.trainer_config
                        .gradient_clip_norm
                    ),
                )
            )

            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    "Non-finite gradient norm detected."
                )

            self.optimizer.step()

            predictions = (
                self.target_standardizer
                .inverse_transform(
                    standardized_predictions.detach()
                )
            )

            batch_size = int(batch.y.numel())

            total_loss += (
                float(loss.detach()) * batch_size
            )
            total_absolute_error += float(
                torch.abs(
                    predictions - batch.y
                ).sum()
            )
            total_samples += batch_size

            maximum_gradient_norm = max(
                maximum_gradient_norm,
                float(gradient_norm),
            )

        return {
            "train_loss": (
                total_loss / total_samples
            ),
            "train_mae": (
                total_absolute_error / total_samples
            ),
            "max_gradient_norm": (
                maximum_gradient_norm
            ),
        }

    @torch.no_grad()
    def predict(
        self,
        loader,
    ) -> dict:
        if self.target_standardizer is None:
            raise RuntimeError(
                "Target standardizer is not initialized."
            )

        self.model.eval()

        predictions = []
        targets = []
        sample_ids = []
        graph_indices = []
        druglike_flags = []
        molecular_weights = []

        for batch in loader:
            batch = batch.to(self.device)

            standardized_predictions = (
                self.model(batch)
            )

            batch_predictions = (
                self.target_standardizer
                .inverse_transform(
                    standardized_predictions
                )
            )

            predictions.append(
                batch_predictions.cpu()
            )
            targets.append(batch.y.cpu())

            sample_ids.extend(
                list(batch.sample_id)
            )

            graph_indices.extend(
                batch.graph_index
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

            druglike_flags.extend(
                batch.in_druglike_scope
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

            if hasattr(batch, "molecular_weight"):
                molecular_weights.extend(
                    batch.molecular_weight
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )

        y_pred = torch.cat(
            predictions
        ).numpy()

        y_true = torch.cat(
            targets
        ).numpy()

        result = {
            "y_true": y_true,
            "y_pred": y_pred,
            "sample_ids": sample_ids,
            "graph_indices": graph_indices,
            "in_druglike_scope": (
                druglike_flags
            ),
            "metrics": regression_metrics(
                y_true,
                y_pred,
            ),
        }

        if molecular_weights:
            result["molecular_weights"] = (
                molecular_weights
            )

        return result

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        valid_metrics: dict,
    ) -> None:
        if self.target_standardizer is None:
            raise RuntimeError(
                "Target standardizer is not initialized."
            )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "format_version": 1,
            "model_class": "GINERegressor",
            "model_config": (
                self.model.config.to_dict()
            ),
            "trainer_config": (
                self.trainer_config.to_dict()
            ),
            "model_state_dict": (
                self.model.state_dict()
            ),
            "target_standardizer": (
                self.target_standardizer.to_dict()
            ),
            "epoch": int(epoch),
            "valid_metrics": valid_metrics,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> dict:
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        standardizer = checkpoint[
            "target_standardizer"
        ]

        self.target_standardizer = (
            TargetStandardizer(
                mean=float(standardizer["mean"]),
                standard_deviation=float(
                    standardizer[
                        "standard_deviation"
                    ]
                ),
            )
        )

        return checkpoint

    def fit_fixed_epochs(
        self,
        train_loader,
        number_of_epochs: int,
        checkpoint_path: str | Path,
    ) -> dict:
        """Fit on all development data for fixed epochs.

        This method is intended for the final refit after
        the number of epochs has already been selected on
        an independent validation set. It must not be used
        for hyperparameter selection.
        """

        if number_of_epochs < 1:
            raise ValueError(
                "number_of_epochs must be at least 1."
            )

        set_global_seed(
            self.trainer_config.seed
        )

        self.target_standardizer = (
            calculate_target_standardizer(
                train_loader.dataset
            )
        )

        self.history = []
        checkpoint_path = Path(
            checkpoint_path
        )

        start_time = time.perf_counter()
        final_train_statistics = None

        for epoch in range(
            1,
            number_of_epochs + 1,
        ):
            epoch_start = time.perf_counter()

            train_statistics = self._train_epoch(
                train_loader
            )

            # No validation set is available during the
            # final refit. The scheduler therefore monitors
            # training MAE, while the epoch count remains
            # fixed by the earlier validation experiment.
            self.scheduler.step(
                train_statistics["train_mae"]
            )

            learning_rate = float(
                self.optimizer.param_groups[0]["lr"]
            )

            epoch_record = {
                "epoch": epoch,
                **train_statistics,
                "learning_rate": learning_rate,
                "epoch_seconds": (
                    time.perf_counter()
                    - epoch_start
                ),
            }

            self.history.append(epoch_record)
            final_train_statistics = (
                train_statistics
            )

        if final_train_statistics is None:
            raise RuntimeError(
                "Fixed-epoch training produced no result."
            )

        training_seconds = (
            time.perf_counter() - start_time
        )

        checkpoint_metrics = {
            "protocol": "fixed_epoch_refit",
            "number_of_training_samples": len(
                train_loader.dataset
            ),
            "train_loss": float(
                final_train_statistics[
                    "train_loss"
                ]
            ),
            "train_mae": float(
                final_train_statistics[
                    "train_mae"
                ]
            ),
        }

        self._save_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch=number_of_epochs,
            valid_metrics=checkpoint_metrics,
        )

        self.best_epoch = number_of_epochs

        return {
            "protocol": "fixed_epoch_refit",
            "epochs_completed": number_of_epochs,
            "training_seconds": training_seconds,
            "number_of_training_samples": len(
                train_loader.dataset
            ),
            "target_standardizer": (
                self.target_standardizer.to_dict()
            ),
            "final_train_statistics": (
                final_train_statistics
            ),
            "history": self.history,
        }

    def fit(
        self,
        train_loader,
        valid_loader,
        checkpoint_path: str | Path,
    ) -> dict:
        set_global_seed(
            self.trainer_config.seed
        )

        self.target_standardizer = (
            calculate_target_standardizer(
                train_loader.dataset
            )
        )

        checkpoint_path = Path(
            checkpoint_path
        )

        start_time = time.perf_counter()
        epochs_without_improvement = 0

        for epoch in range(
            1,
            self.trainer_config.max_epochs + 1,
        ):
            epoch_start = time.perf_counter()

            train_statistics = (
                self._train_epoch(train_loader)
            )

            valid_result = self.predict(
                valid_loader
            )
            valid_metrics = valid_result[
                "metrics"
            ]

            valid_mae = float(
                valid_metrics["mae"]
            )

            self.scheduler.step(valid_mae)

            learning_rate = float(
                self.optimizer.param_groups[0][
                    "lr"
                ]
            )

            epoch_record = {
                "epoch": epoch,
                **train_statistics,
                "valid_mae": valid_mae,
                "valid_rmse": float(
                    valid_metrics["rmse"]
                ),
                "valid_r2": float(
                    valid_metrics["r2"]
                ),
                "valid_spearman": float(
                    valid_metrics["spearman"]
                ),
                "learning_rate": learning_rate,
                "epoch_seconds": (
                    time.perf_counter()
                    - epoch_start
                ),
            }

            self.history.append(epoch_record)

            improved = (
                valid_mae
                < self.best_valid_mae
                - self.trainer_config.min_delta
            )

            if improved:
                self.best_valid_mae = valid_mae
                self.best_epoch = epoch
                epochs_without_improvement = 0

                self._save_checkpoint(
                    checkpoint_path,
                    epoch,
                    valid_metrics,
                )
            else:
                epochs_without_improvement += 1

            logging.info(
                "Epoch %03d | "
                "loss %.4f | "
                "train MAE %.4f | "
                "valid MAE %.4f | "
                "lr %.2e | "
                "grad %.3f%s",
                epoch,
                train_statistics["train_loss"],
                train_statistics["train_mae"],
                valid_mae,
                learning_rate,
                train_statistics[
                    "max_gradient_norm"
                ],
                " | best" if improved else "",
            )

            if (
                epochs_without_improvement
                >= self.trainer_config.patience
            ):
                logging.info(
                    "Early stopping at epoch %d",
                    epoch,
                )
                break

        total_seconds = (
            time.perf_counter() - start_time
        )

        checkpoint = self.load_checkpoint(
            checkpoint_path
        )

        return {
            "best_epoch": self.best_epoch,
            "best_valid_mae": (
                self.best_valid_mae
            ),
            "epochs_completed": len(
                self.history
            ),
            "training_seconds": total_seconds,
            "target_standardizer": (
                self.target_standardizer.to_dict()
            ),
            "checkpoint_valid_metrics": (
                checkpoint["valid_metrics"]
            ),
            "history": self.history,
        }


def load_gine_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[GINERegressor, TargetStandardizer, dict]:
    """Rebuild a GINE model using a safe tensor checkpoint."""

    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=True,
    )

    model = GINERegressor(
        GINEConfig(
            **checkpoint["model_config"]
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(device)
    model.eval()

    standardizer_data = checkpoint[
        "target_standardizer"
    ]

    standardizer = TargetStandardizer(
        mean=float(
            standardizer_data["mean"]
        ),
        standard_deviation=float(
            standardizer_data[
                "standard_deviation"
            ]
        ),
    )

    return model, standardizer, checkpoint