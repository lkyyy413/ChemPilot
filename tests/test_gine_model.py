import pytest
import torch
from torch_geometric.loader import DataLoader

from chempilot.features.graph import (
    GraphFeaturizer,
)
from chempilot.models.gine import (
    CategoricalFeatureEncoder,
    GINEConfig,
    GINERegressor,
)


def make_batch():
    featurizer = GraphFeaturizer()

    graphs = [
        featurizer.transform_one(
            "CCO",
            y=-2.0,
            sample_id="ethanol",
        ),
        featurizer.transform_one(
            "c1ccccc1",
            y=-3.0,
            sample_id="benzene",
        ),
        featurizer.transform_one(
            "[Na+]",
            y=0.5,
            sample_id="sodium",
        ),
    ]

    loader = DataLoader(
        graphs,
        batch_size=3,
        shuffle=False,
    )

    return next(iter(loader))


def test_categorical_encoder_shape():
    encoder = CategoricalFeatureEncoder(
        cardinalities=[5, 3],
        embedding_dim=16,
    )

    features = torch.tensor(
        [
            [0, 1],
            [4, 2],
        ],
        dtype=torch.long,
    )

    encoded = encoder(features)

    assert encoded.shape == (2, 16)
    assert torch.isfinite(encoded).all()


def test_categorical_encoder_rejects_float():
    encoder = CategoricalFeatureEncoder(
        cardinalities=[5, 3],
        embedding_dim=16,
    )

    features = torch.zeros(
        (2, 2),
        dtype=torch.float32,
    )

    with pytest.raises(
        TypeError,
        match="torch.long",
    ):
        encoder(features)


def test_gine_forward_shape_and_finiteness():
    batch = make_batch()

    model = GINERegressor(
        GINEConfig(
            hidden_dim=64,
            num_layers=3,
            dropout=0.1,
        )
    )

    predictions = model(batch)

    assert predictions.shape == (3,)
    assert torch.isfinite(predictions).all()


def test_gine_supports_zero_edge_graph():
    featurizer = GraphFeaturizer()
    graph = featurizer.transform_one(
        "[Na+]",
        y=0.5,
    )

    model = GINERegressor(
        GINEConfig(
            hidden_dim=32,
            num_layers=2,
        )
    )

    prediction = model(graph)

    assert prediction.shape == (1,)
    assert torch.isfinite(prediction).all()


def test_gine_backward_has_finite_gradients():
    batch = make_batch()

    model = GINERegressor(
        GINEConfig(
            hidden_dim=64,
            num_layers=3,
            dropout=0.1,
        )
    )

    predictions = model(batch)
    loss = torch.nn.functional.smooth_l1_loss(
        predictions,
        batch.y,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

    total_gradient = sum(
        float(gradient.abs().sum())
        for gradient in gradients
    )

    assert total_gradient > 0.0


def test_parameter_count_is_controlled():
    model = GINERegressor(
        GINEConfig(
            hidden_dim=128,
            num_layers=4,
        )
    )

    number_of_parameters = (
        model.parameter_count()
    )

    assert number_of_parameters > 0
    assert number_of_parameters < 2_000_000