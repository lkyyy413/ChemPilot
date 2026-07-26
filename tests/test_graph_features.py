import pytest
import torch

from chempilot.features.graph import (
    ATOM_FEATURE_CARDINALITIES,
    BOND_FEATURE_CARDINALITIES,
    GraphFeaturizer,
)


@pytest.fixture
def featurizer():
    return GraphFeaturizer()


def test_ethanol_graph_shape(featurizer):
    graph = featurizer.transform_one(
        smiles="CCO",
        y=-2.5,
        sample_id="ethanol",
        in_druglike_scope=True,
    )

    assert graph.x.shape == (3, 7)
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.shape == (4, 4)
    assert graph.y.shape == (1,)

    assert graph.x.dtype == torch.long
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_attr.dtype == torch.long
    assert graph.y.dtype == torch.float32

    assert graph.sample_id == "ethanol"
    assert graph.smiles == "CCO"
    assert graph.in_druglike_scope.item() is True
    assert graph.y.item() == pytest.approx(-2.5)


def test_edges_are_bidirectional(featurizer):
    graph = featurizer.transform_one("CCO")

    directed_edges = {
        tuple(edge)
        for edge in graph.edge_index.t().tolist()
    }

    assert directed_edges == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }

    assert torch.equal(
        graph.edge_attr[0],
        graph.edge_attr[1],
    )
    assert torch.equal(
        graph.edge_attr[2],
        graph.edge_attr[3],
    )


def test_benzene_aromatic_features(featurizer):
    graph = featurizer.transform_one("c1ccccc1")

    aromatic_column = 5
    bond_type_column = 0
    ring_column = 2

    assert graph.x.shape == (6, 7)
    assert graph.edge_attr.shape == (12, 4)

    assert torch.all(
        graph.x[:, aromatic_column] == 1
    )

    assert torch.all(
        graph.edge_attr[:, bond_type_column] == 3
    )

    assert torch.all(
        graph.edge_attr[:, ring_column] == 1
    )


def test_disconnected_salt_has_no_edges(featurizer):
    graph = featurizer.transform_one("[Na+].[Cl-]")

    assert graph.x.shape == (2, 7)
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 4)
    assert graph.num_nodes == 2


def test_feature_indices_within_cardinalities(
    featurizer,
):
    smiles_list = [
        "CCO",
        "c1ccccc1",
        "[Na+].[Cl-]",
        "N[C@@H](C)C(=O)O",
        "C#N",
        "C=C",
    ]

    for smiles in smiles_list:
        graph = featurizer.transform_one(smiles)

        for column, cardinality in enumerate(
            ATOM_FEATURE_CARDINALITIES
        ):
            values = graph.x[:, column]

            assert int(values.min()) >= 0
            assert int(values.max()) < cardinality

        if graph.edge_attr.numel() > 0:
            for column, cardinality in enumerate(
                BOND_FEATURE_CARDINALITIES
            ):
                values = graph.edge_attr[:, column]

                assert int(values.min()) >= 0
                assert int(values.max()) < cardinality


def test_chiral_enantiomers_have_different_features(
    featurizer,
):
    clockwise = featurizer.transform_one(
        "N[C@H](C)C(=O)O"
    )
    anticlockwise = featurizer.transform_one(
        "N[C@@H](C)C(=O)O"
    )

    chirality_column = 6

    assert not torch.equal(
        clockwise.x[:, chirality_column],
        anticlockwise.x[:, chirality_column],
    )


def test_invalid_smiles_raises(featurizer):
    with pytest.raises(ValueError):
        featurizer.transform_one("not_a_smiles")