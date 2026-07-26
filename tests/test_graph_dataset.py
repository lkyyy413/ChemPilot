import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from chempilot.data.graph_dataset import (
    MoleculeGraphDataset,
    MoleculeGraphSubset,
)


@pytest.fixture
def graph_files(tmp_path):
    cache_path = tmp_path / "graphs.pt"
    metadata_path = tmp_path / "metadata.csv"
    split_path = tmp_path / "split.csv"

    metadata = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "Drug_ID": ["ethanol", "sodium"],
            "smiles_canonical": ["CCO", "[Na+]"],
            "Y": [-2.0, 0.5],
            "in_druglike_scope": [True, False],
            "molecular_weight": [46.07, 22.99],
        }
    )
    metadata.to_csv(metadata_path, index=False)

    cache = {
        "x": torch.tensor(
            [
                [6, 1, 4, 5, 4, 0, 0],
                [6, 2, 4, 5, 4, 0, 0],
                [8, 1, 2, 5, 4, 0, 0],
                [11, 0, 0, 6, 1, 0, 0],
            ],
            dtype=torch.long,
        ),
        "edge_index": torch.tensor(
            [
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ],
            dtype=torch.long,
        ),
        "edge_attr": torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
        "node_ptr": torch.tensor(
            [0, 3, 4],
            dtype=torch.long,
        ),
        "edge_ptr": torch.tensor(
            [0, 4, 4],
            dtype=torch.long,
        ),
        "y": torch.tensor(
            [-2.0, 0.5],
            dtype=torch.float32,
        ),
        "in_druglike_scope": torch.tensor(
            [True, False],
            dtype=torch.bool,
        ),
    }
    torch.save(cache, cache_path)

    split = metadata.iloc[[1, 0]].copy()
    split.to_csv(split_path, index=False)

    return cache_path, metadata_path, split_path


def test_graph_reconstruction(graph_files):
    cache_path, metadata_path, _ = graph_files

    dataset = MoleculeGraphDataset(
        cache_path=cache_path,
        metadata_path=metadata_path,
    )

    graph = dataset[0]

    assert len(dataset) == 2
    assert graph.sample_id == "S1"
    assert graph.x.shape == (3, 7)
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.shape == (4, 4)
    assert graph.y.item() == pytest.approx(-2.0)
    assert graph.molecular_weight.item() == pytest.approx(
        46.07,
        abs=1e-4,
    )

    expected_edges = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        graph.edge_index,
        expected_edges,
    )


def test_zero_edge_graph(graph_files):
    cache_path, metadata_path, _ = graph_files

    dataset = MoleculeGraphDataset(
        cache_path=cache_path,
        metadata_path=metadata_path,
    )

    graph = dataset[1]

    assert graph.sample_id == "S2"
    assert graph.num_nodes == 1
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 4)
    assert graph.in_druglike_scope.item() is False


def test_split_order_is_preserved(graph_files):
    cache_path, metadata_path, split_path = graph_files

    dataset = MoleculeGraphDataset(
        cache_path=cache_path,
        metadata_path=metadata_path,
    )

    subset = dataset.subset_from_split(split_path)

    assert len(subset) == 2
    assert subset.sample_ids == ["S2", "S1"]
    assert subset[0].sample_id == "S2"
    assert subset[1].sample_id == "S1"


def test_mixed_batch_with_zero_edge_graph(
    graph_files,
):
    cache_path, metadata_path, _ = graph_files

    dataset = MoleculeGraphDataset(
        cache_path=cache_path,
        metadata_path=metadata_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch.num_graphs == 2
    assert batch.x.shape == (4, 7)
    assert batch.edge_index.shape == (2, 4)
    assert batch.edge_attr.shape == (4, 4)
    assert batch.y.shape == (2,)
    assert batch.batch.tolist() == [0, 0, 0, 1]


def test_subset_accepts_explicit_indices(
    graph_files,
):
    cache_path, metadata_path, _ = graph_files

    dataset = MoleculeGraphDataset(
        cache_path=cache_path,
        metadata_path=metadata_path,
    )

    subset = MoleculeGraphSubset(
        dataset,
        indices=[1],
    )

    assert len(subset) == 1
    assert subset[0].sample_id == "S2"


def test_split_label_mismatch_is_detected(
    graph_files,
):
    cache_path, metadata_path, split_path = graph_files

    split = pd.read_csv(split_path)
    split.loc[0, "Y"] = -99.0
    split.to_csv(split_path, index=False)

    dataset = MoleculeGraphDataset(
        cache_path=cache_path,
        metadata_path=metadata_path,
    )

    with pytest.raises(
        ValueError,
        match="Label mismatch",
    ):
        dataset.subset_from_split(split_path)