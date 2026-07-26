from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from chempilot.features.graph import GraphFeaturizer
from chempilot.models.gine import (
    GINEConfig,
    GINERegressor,
)
from chempilot.training.gine_trainer import (
    GINETrainer,
    TargetStandardizer,
    TrainerConfig,
    calculate_target_standardizer,
    load_gine_from_checkpoint,
    set_global_seed,
)


def build_tiny_graph_dataset():
    """Build a small self-contained dataset for trainer tests."""

    records = [
        ("sample_0", "CCO", -0.3),
        ("sample_1", "CCCO", -0.8),
        ("sample_2", "CCCCO", -1.4),
        ("sample_3", "c1ccccc1", -2.1),
        ("sample_4", "Cc1ccccc1", -2.7),
        ("sample_5", "CC(=O)O", 0.1),
    ]

    featurizer = GraphFeaturizer()
    graphs = []

    for graph_index, (
        sample_id,
        smiles,
        target,
    ) in enumerate(records):
        graph = featurizer.transform_one(
            smiles=smiles,
            y=target,
            sample_id=sample_id,
            in_druglike_scope=True,
        )

        graph.graph_index = torch.tensor(
            [graph_index],
            dtype=torch.long,
        )
        graph.drug_id = sample_id
        graph.molecular_weight = torch.tensor(
            [100.0 + graph_index],
            dtype=torch.float32,
        )

        graphs.append(graph)

    return graphs


def test_target_standardizer_round_trip():
    standardizer = TargetStandardizer(
        mean=-2.0,
        standard_deviation=1.5,
    )

    values = torch.tensor(
        [-4.0, -2.0, 0.5],
        dtype=torch.float32,
    )

    transformed = standardizer.transform(values)
    reconstructed = standardizer.inverse_transform(
        transformed
    )

    assert torch.isfinite(transformed).all()
    assert torch.allclose(
        reconstructed,
        values,
        atol=1e-6,
    )


def test_calculate_target_standardizer():
    dataset = build_tiny_graph_dataset()

    standardizer = calculate_target_standardizer(
        dataset
    )

    expected = np.array(
        [
            -0.3,
            -0.8,
            -1.4,
            -2.1,
            -2.7,
            0.1,
        ],
        dtype=np.float64,
    )

    assert standardizer.mean == pytest.approx(
        expected.mean()
    )
    assert (
        standardizer.standard_deviation
        == pytest.approx(
            expected.std(ddof=0)
        )
    )


def test_trainer_fit_checkpoint_and_predict(
    tmp_path: Path,
):
    set_global_seed(7)

    dataset = build_tiny_graph_dataset()

    train_loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=0,
    )

    model = GINERegressor(
        GINEConfig(
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
            pooling="add",
        )
    )

    trainer = GINETrainer(
        model=model,
        trainer_config=TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            max_epochs=3,
            patience=3,
            scheduler_patience=2,
            seed=7,
        ),
        device="cpu",
    )

    checkpoint_path = (
        tmp_path / "tiny_gine_checkpoint.pt"
    )

    fit_result = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0
    assert isinstance(fit_result, dict)

    prediction_result = trainer.predict(
        valid_loader
    )

    assert prediction_result["y_true"].shape == (6,)
    assert prediction_result["y_pred"].shape == (6,)
    assert len(
        prediction_result["sample_ids"]
    ) == 6

    assert np.isfinite(
        prediction_result["y_true"]
    ).all()
    assert np.isfinite(
        prediction_result["y_pred"]
    ).all()

    assert prediction_result["sample_ids"] == [
        f"sample_{index}"
        for index in range(6)
    ]

    restored_model, restored_standardizer, metadata = (
        load_gine_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device="cpu",
        )
    )

    assert isinstance(
        restored_model,
        GINERegressor,
    )
    assert isinstance(
        restored_standardizer,
        TargetStandardizer,
    )
    assert isinstance(metadata, dict)

    restored_trainer = GINETrainer(
        model=restored_model,
        trainer_config=TrainerConfig(
            max_epochs=1,
            seed=7,
        ),
        device="cpu",
    )
    restored_trainer.target_standardizer = (
        restored_standardizer
    )

    restored_predictions = restored_trainer.predict(
        valid_loader
    )

    assert np.allclose(
        prediction_result["y_pred"],
        restored_predictions["y_pred"],
        atol=1e-6,
    )

def test_fixed_epoch_refit(
    tmp_path: Path,
):
    set_global_seed(11)

    dataset = build_tiny_graph_dataset()

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=0,
    )

    model = GINERegressor(
        GINEConfig(
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
            pooling="add",
        )
    )

    trainer = GINETrainer(
        model=model,
        trainer_config=TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            max_epochs=10,
            patience=10,
            seed=11,
        ),
        device="cpu",
    )

    checkpoint_path = (
        tmp_path
        / "fixed_epoch_checkpoint.pt"
    )

    result = trainer.fit_fixed_epochs(
        train_loader=loader,
        number_of_epochs=3,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert result["protocol"] == (
        "fixed_epoch_refit"
    )
    assert result["epochs_completed"] == 3
    assert result[
        "number_of_training_samples"
    ] == 6
    assert len(result["history"]) == 3

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["epoch"] == 3
    assert (
        checkpoint["valid_metrics"]["protocol"]
        == "fixed_epoch_refit"
    )
    assert (
        checkpoint["valid_metrics"][
            "number_of_training_samples"
        ]
        == 6
    )

    predictions = trainer.predict(loader)

    assert predictions["y_pred"].shape == (6,)
    assert np.isfinite(
        predictions["y_pred"]
    ).all()


def test_fixed_epoch_refit_rejects_zero_epochs(
    tmp_path: Path,
):
    dataset = build_tiny_graph_dataset()

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=0,
    )

    model = GINERegressor(
        GINEConfig(
            hidden_dim=32,
            num_layers=2,
        )
    )

    trainer = GINETrainer(
        model=model,
        trainer_config=TrainerConfig(),
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        trainer.fit_fixed_epochs(
            train_loader=loader,
            number_of_epochs=0,
            checkpoint_path=(
                tmp_path / "invalid.pt"
            ),
        )