"""Dataset utilities for cached molecular graphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


REQUIRED_CACHE_KEYS = {
    "x",
    "edge_index",
    "edge_attr",
    "node_ptr",
    "edge_ptr",
    "y",
    "in_druglike_scope",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


class MoleculeGraphDataset(Dataset):
    """Random-access molecular graphs aligned to processed data."""

    def __init__(
        self,
        cache_path: str | Path,
        metadata_path: str | Path,
        manifest_path: str | Path | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.metadata_path = Path(metadata_path)

        if not self.cache_path.exists():
            raise FileNotFoundError(self.cache_path)

        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)

        self.cache = torch.load(
            self.cache_path,
            map_location="cpu",
            weights_only=True,
        )

        missing_keys = (
            REQUIRED_CACHE_KEYS.difference(
                self.cache.keys()
            )
        )

        if missing_keys:
            raise ValueError(
                f"Missing graph-cache keys: "
                f"{sorted(missing_keys)}"
            )

        self.metadata = pd.read_csv(
            self.metadata_path
        )

        required_columns = {
            "sample_id",
            "Drug_ID",
            "smiles_canonical",
            "Y",
            "in_druglike_scope",
        }

        missing_columns = required_columns.difference(
            self.metadata.columns
        )

        if missing_columns:
            raise ValueError(
                f"Missing metadata columns: "
                f"{sorted(missing_columns)}"
            )

        if self.metadata["sample_id"].duplicated().any():
            raise ValueError(
                "Duplicate sample IDs in metadata."
            )

        self._validate_shapes()

        cached_y = self.cache["y"].numpy()
        metadata_y = self.metadata[
            "Y"
        ].to_numpy(dtype=np.float32)

        if not np.allclose(
            cached_y,
            metadata_y,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                "Graph-cache labels do not match metadata."
            )

        cached_scope = self.cache[
            "in_druglike_scope"
        ].numpy()
        metadata_scope = self.metadata[
            "in_druglike_scope"
        ].to_numpy(dtype=bool)

        if not np.array_equal(
            cached_scope,
            metadata_scope,
        ):
            raise ValueError(
                "Drug-like flags do not match metadata."
            )

        self.sample_to_index = {
            sample_id: index
            for index, sample_id in enumerate(
                self.metadata[
                    "sample_id"
                ].astype(str)
            )
        }

        if manifest_path is not None:
            self._validate_manifest(
                Path(manifest_path),
                verify_hashes=verify_hashes,
            )

    def _validate_shapes(self) -> None:
        number_of_graphs = len(self.metadata)

        if self.cache["node_ptr"].shape != (
            number_of_graphs + 1,
        ):
            raise ValueError(
                "Invalid node pointer shape."
            )

        if self.cache["edge_ptr"].shape != (
            number_of_graphs + 1,
        ):
            raise ValueError(
                "Invalid edge pointer shape."
            )

        if self.cache["y"].shape != (
            number_of_graphs,
        ):
            raise ValueError("Invalid label shape.")

        if self.cache[
            "in_druglike_scope"
        ].shape != (number_of_graphs,):
            raise ValueError(
                "Invalid drug-like flag shape."
            )

        if int(self.cache["node_ptr"][0]) != 0:
            raise ValueError(
                "Node pointers must start at zero."
            )

        if int(self.cache["edge_ptr"][0]) != 0:
            raise ValueError(
                "Edge pointers must start at zero."
            )

        if int(self.cache["node_ptr"][-1]) != len(
            self.cache["x"]
        ):
            raise ValueError(
                "Final node pointer is inconsistent."
            )

        if int(self.cache["edge_ptr"][-1]) != (
            self.cache["edge_index"].shape[1]
        ):
            raise ValueError(
                "Final edge pointer is inconsistent."
            )

        if self.cache["edge_attr"].shape[0] != (
            self.cache["edge_index"].shape[1]
        ):
            raise ValueError(
                "Edge attributes and edge indices "
                "have different lengths."
            )

        if not torch.all(
            self.cache["node_ptr"][1:]
            >= self.cache["node_ptr"][:-1]
        ):
            raise ValueError(
                "Node pointers are not monotonic."
            )

        if not torch.all(
            self.cache["edge_ptr"][1:]
            >= self.cache["edge_ptr"][:-1]
        ):
            raise ValueError(
                "Edge pointers are not monotonic."
            )

    def _validate_manifest(
        self,
        manifest_path: Path,
        verify_hashes: bool,
    ) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        with manifest_path.open(
            encoding="utf-8"
        ) as file:
            manifest = json.load(file)

        if manifest["number_of_graphs"] != len(self):
            raise ValueError(
                "Manifest graph count mismatch."
            )

        if verify_hashes:
            input_hash = sha256_file(
                self.metadata_path
            )
            output_hash = sha256_file(
                self.cache_path
            )

            if input_hash != manifest["input_sha256"]:
                raise ValueError(
                    "Metadata SHA-256 mismatch."
                )

            if output_hash != manifest["output_sha256"]:
                raise ValueError(
                    "Graph-cache SHA-256 mismatch."
                )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Data:
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(index)

        node_start = int(
            self.cache["node_ptr"][index]
        )
        node_end = int(
            self.cache["node_ptr"][index + 1]
        )

        edge_start = int(
            self.cache["edge_ptr"][index]
        )
        edge_end = int(
            self.cache["edge_ptr"][index + 1]
        )

        x = self.cache["x"][
            node_start:node_end
        ]

        global_edge_index = self.cache[
            "edge_index"
        ][:, edge_start:edge_end]

        edge_index = (
            global_edge_index - node_start
        )

        edge_attr = self.cache["edge_attr"][
            edge_start:edge_end
        ]

        row = self.metadata.iloc[index]

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=self.cache["y"][index].reshape(1),
            in_druglike_scope=self.cache[
                "in_druglike_scope"
            ][index].reshape(1),
            num_nodes=node_end - node_start,
        )

        data.sample_id = str(row["sample_id"])
        data.drug_id = str(row["Drug_ID"])
        data.smiles = str(
            row["smiles_canonical"]
        )
        data.graph_index = torch.tensor(
            [index],
            dtype=torch.long,
        )

        if "molecular_weight" in self.metadata.columns:
            data.molecular_weight = torch.tensor(
                [float(row["molecular_weight"])],
                dtype=torch.float32,
            )

        return data

    def indices_from_split(
        self,
        split_path: str | Path,
    ) -> list[int]:
        split_path = Path(split_path)

        if not split_path.exists():
            raise FileNotFoundError(split_path)

        split = pd.read_csv(split_path)

        required_columns = {
            "sample_id",
            "smiles_canonical",
            "Y",
            "in_druglike_scope",
        }

        missing_columns = required_columns.difference(
            split.columns
        )

        if missing_columns:
            raise ValueError(
                f"Split is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if split["sample_id"].duplicated().any():
            raise ValueError(
                "Duplicate sample IDs in split."
            )

        indices = []

        for row in split.itertuples(index=False):
            sample_id = str(row.sample_id)

            if sample_id not in self.sample_to_index:
                raise ValueError(
                    f"Unknown sample ID: {sample_id}"
                )

            index = self.sample_to_index[sample_id]
            metadata_row = self.metadata.iloc[index]

            if str(
                metadata_row["smiles_canonical"]
            ) != str(row.smiles_canonical):
                raise ValueError(
                    f"SMILES mismatch for {sample_id}"
                )

            if not np.isclose(
                float(metadata_row["Y"]),
                float(row.Y),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError(
                    f"Label mismatch for {sample_id}"
                )

            if bool(
                metadata_row["in_druglike_scope"]
            ) != bool(row.in_druglike_scope):
                raise ValueError(
                    f"Scope mismatch for {sample_id}"
                )

            indices.append(index)

        return indices

    def subset_from_split(
        self,
        split_path: str | Path,
    ) -> "MoleculeGraphSubset":
        return MoleculeGraphSubset(
            dataset=self,
            indices=self.indices_from_split(
                split_path
            ),
        )


class MoleculeGraphSubset(Dataset):
    """Ordered graph subset backed by one graph store."""

    def __init__(
        self,
        dataset: MoleculeGraphDataset,
        indices: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.indices = [
            int(index)
            for index in indices
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Data:
        return self.dataset[
            self.indices[index]
        ]

    @property
    def sample_ids(self) -> list[str]:
        return [
            str(
                self.dataset.metadata.iloc[
                    graph_index
                ]["sample_id"]
            )
            for graph_index in self.indices
        ]