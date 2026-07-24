import numpy as np
import pandas as pd
import pytest

from chempilot.data.feature_store import (
    MolecularFeatureStore,
)


def create_cache(path):
    np.savez_compressed(
        path,
        sample_ids=np.asarray(
            ["S1", "S2"],
            dtype=np.str_,
        ),
        smiles=np.asarray(
            ["CCO", "CC"],
            dtype=np.str_,
        ),
        y=np.asarray(
            [-1.0, -2.0],
            dtype=np.float32,
        ),
        in_druglike_scope=np.asarray(
            [True, False],
            dtype=bool,
        ),
        descriptors=np.asarray(
            [
                [1.0] * 10,
                [2.0] * 10,
            ],
            dtype=np.float32,
        ),
        ecfp=np.asarray(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.uint8,
        ),
        descriptor_names=np.asarray(
            [f"d{i}" for i in range(10)],
            dtype=np.str_,
        ),
        ecfp_names=np.asarray(
            [f"ecfp_{i}" for i in range(4)],
            dtype=np.str_,
        ),
    )


def test_feature_store_preserves_split_order(tmp_path):
    cache_path = tmp_path / "features.npz"
    split_path = tmp_path / "split.csv"

    create_cache(cache_path)

    pd.DataFrame(
        {
            "sample_id": ["S2", "S1"],
            "smiles_canonical": ["CC", "CCO"],
            "Y": [-2.0, -1.0],
            "in_druglike_scope": [False, True],
        }
    ).to_csv(split_path, index=False)

    store = MolecularFeatureStore(cache_path)
    batch = store.load_split(
        split_path,
        representation="descriptors",
    )

    assert batch.sample_ids.tolist() == ["S2", "S1"]
    assert batch.x.shape == (2, 10)
    assert batch.y.tolist() == [-2.0, -1.0]
    assert batch.x[0, 0] == pytest.approx(2.0)


def test_feature_store_detects_label_mismatch(tmp_path):
    cache_path = tmp_path / "features.npz"
    split_path = tmp_path / "split.csv"

    create_cache(cache_path)

    pd.DataFrame(
        {
            "sample_id": ["S1"],
            "smiles_canonical": ["CCO"],
            "Y": [-9.0],
            "in_druglike_scope": [True],
        }
    ).to_csv(split_path, index=False)

    store = MolecularFeatureStore(cache_path)

    with pytest.raises(
        ValueError,
        match="Label mismatch",
    ):
        store.load_split(
            split_path,
            representation="descriptors",
        )