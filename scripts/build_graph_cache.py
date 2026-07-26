#!/usr/bin/env python
"""Build a tensor-only molecular graph cache for AqSolDB."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from rdkit import rdBase
from tqdm.auto import tqdm

from chempilot.features.graph import (
    ATOM_FEATURE_CARDINALITIES,
    ATOM_FEATURE_NAMES,
    BOND_FEATURE_CARDINALITIES,
    BOND_FEATURE_NAMES,
    GraphFeaturizer,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "solubility_aqsoldb_processed.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "solubility_aqsoldb_graphs.pt"
)

MANIFEST_PATH = (
    ROOT
    / "reports"
    / "graph_feature_manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def percentile_summary(values: list[int]) -> dict:
    array = np.asarray(values, dtype=np.float64)

    return {
        "minimum": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": int(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "sample_id",
        "smiles_canonical",
        "Y",
        "in_druglike_scope",
    }

    missing_columns = required_columns.difference(
        df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs detected.")

    featurizer = GraphFeaturizer()

    node_parts = []
    edge_index_parts = []
    edge_attribute_parts = []

    node_ptr = [0]
    edge_ptr = [0]

    node_counts = []
    edge_counts = []
    zero_edge_sample_ids = []

    total_nodes = 0
    total_edges = 0

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Building graph cache",
    ):
        graph = featurizer.transform_one(
            smiles=row.smiles_canonical,
            y=row.Y,
            sample_id=row.sample_id,
            in_druglike_scope=row.in_druglike_scope,
        )

        number_of_nodes = int(graph.num_nodes)
        number_of_edges = int(
            graph.edge_index.shape[1]
        )

        node_parts.append(graph.x)

        if number_of_edges > 0:
            global_edge_index = (
                graph.edge_index + total_nodes
            )
            edge_index_parts.append(global_edge_index)
            edge_attribute_parts.append(
                graph.edge_attr
            )

        total_nodes += number_of_nodes
        total_edges += number_of_edges

        node_ptr.append(total_nodes)
        edge_ptr.append(total_edges)

        node_counts.append(number_of_nodes)
        edge_counts.append(number_of_edges)

        if number_of_edges == 0:
            zero_edge_sample_ids.append(
                row.sample_id
            )

    x = torch.cat(node_parts, dim=0)

    if edge_index_parts:
        edge_index = torch.cat(
            edge_index_parts,
            dim=1,
        )
        edge_attr = torch.cat(
            edge_attribute_parts,
            dim=0,
        )
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )
        edge_attr = torch.empty(
            (0, len(BOND_FEATURE_NAMES)),
            dtype=torch.long,
        )

    cache = {
        "x": x.contiguous(),
        "edge_index": edge_index.contiguous(),
        "edge_attr": edge_attr.contiguous(),
        "node_ptr": torch.tensor(
            node_ptr,
            dtype=torch.long,
        ),
        "edge_ptr": torch.tensor(
            edge_ptr,
            dtype=torch.long,
        ),
        "y": torch.tensor(
            df["Y"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ),
        "in_druglike_scope": torch.tensor(
            df["in_druglike_scope"].to_numpy(
                dtype=bool
            ),
            dtype=torch.bool,
        ),
    }

    number_of_graphs = len(df)

    if cache["node_ptr"].shape != (
        number_of_graphs + 1,
    ):
        raise ValueError("Invalid node pointer shape.")

    if cache["edge_ptr"].shape != (
        number_of_graphs + 1,
    ):
        raise ValueError("Invalid edge pointer shape.")

    if cache["x"].shape != (
        total_nodes,
        len(ATOM_FEATURE_NAMES),
    ):
        raise ValueError("Invalid node feature shape.")

    if cache["edge_index"].shape != (
        2,
        total_edges,
    ):
        raise ValueError("Invalid edge index shape.")

    if cache["edge_attr"].shape != (
        total_edges,
        len(BOND_FEATURE_NAMES),
    ):
        raise ValueError("Invalid edge attribute shape.")

    if cache["y"].shape != (number_of_graphs,):
        raise ValueError("Invalid label shape.")

    if int(cache["node_ptr"][-1]) != total_nodes:
        raise ValueError("Node pointer total mismatch.")

    if int(cache["edge_ptr"][-1]) != total_edges:
        raise ValueError("Edge pointer total mismatch.")

    if total_edges > 0:
        if int(edge_index.min()) < 0:
            raise ValueError("Negative global edge index.")

        if int(edge_index.max()) >= total_nodes:
            raise ValueError("Global edge index overflow.")

    atom_unknown_counts = {
        "atomic_number_unknown": int(
            (x[:, 0] == 0).sum()
        ),
        "degree_overflow": int(
            (
                x[:, 1]
                == ATOM_FEATURE_CARDINALITIES[1] - 1
            ).sum()
        ),
        "total_valence_overflow": int(
            (
                x[:, 2]
                == ATOM_FEATURE_CARDINALITIES[2] - 1
            ).sum()
        ),
        "formal_charge_overflow": int(
            (
                x[:, 3]
                == ATOM_FEATURE_CARDINALITIES[3] - 1
            ).sum()
        ),
        "hybridization_unknown": int(
            (
                x[:, 4]
                == ATOM_FEATURE_CARDINALITIES[4] - 1
            ).sum()
        ),
        "chirality_unknown": int(
            (
                x[:, 6]
                == ATOM_FEATURE_CARDINALITIES[6] - 1
            ).sum()
        ),
    }

    bond_unknown_counts = {
        "bond_type_unknown": int(
            (
                edge_attr[:, 0]
                == BOND_FEATURE_CARDINALITIES[0] - 1
            ).sum()
        ),
        "stereo_unknown": int(
            (
                edge_attr[:, 3]
                == BOND_FEATURE_CARDINALITIES[3] - 1
            ).sum()
        ),
    }

    multi_fragment_count = int(
        df["smiles_canonical"]
        .astype(str)
        .str.contains(
            ".",
            regex=False,
        )
        .sum()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(cache, OUTPUT_PATH)

    loaded = torch.load(
        OUTPUT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    for key in cache:
        if not torch.equal(cache[key], loaded[key]):
            raise ValueError(
                f"Cache round-trip mismatch for {key}"
            )

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": str(
            INPUT_PATH.relative_to(ROOT)
        ),
        "output_file": str(
            OUTPUT_PATH.relative_to(ROOT)
        ),
        "input_sha256": sha256_file(INPUT_PATH),
        "output_sha256": sha256_file(OUTPUT_PATH),
        "number_of_graphs": number_of_graphs,
        "total_nodes": total_nodes,
        "total_directed_edges": total_edges,
        "node_count_statistics": (
            percentile_summary(node_counts)
        ),
        "directed_edge_count_statistics": (
            percentile_summary(edge_counts)
        ),
        "zero_edge_graphs": len(
            zero_edge_sample_ids
        ),
        "zero_edge_sample_id_examples": (
            zero_edge_sample_ids[:20]
        ),
        "multi_fragment_graphs": (
            multi_fragment_count
        ),
        "atom_feature_names": (
            ATOM_FEATURE_NAMES
        ),
        "atom_feature_cardinalities": (
            ATOM_FEATURE_CARDINALITIES
        ),
        "bond_feature_names": (
            BOND_FEATURE_NAMES
        ),
        "bond_feature_cardinalities": (
            BOND_FEATURE_CARDINALITIES
        ),
        "atom_unknown_or_overflow_counts": (
            atom_unknown_counts
        ),
        "bond_unknown_counts": (
            bond_unknown_counts
        ),
        "boolean_feature_counts": {
            "aromatic_atoms": int(
                (x[:, 5] == 1).sum()
            ),
            "conjugated_directed_edges": int(
                (edge_attr[:, 1] == 1).sum()
            ),
            "ring_directed_edges": int(
                (edge_attr[:, 2] == 1).sum()
            ),
        },
        "cache_security": {
            "contains_pyg_data_objects": False,
            "contains_python_object_arrays": False,
            "load_with_weights_only": True,
        },
        "row_alignment": (
            "Cache graph index matches the row order "
            "of the processed input CSV."
        ),
        "software_versions": {
            "torch": torch.__version__,
            "torch_geometric": (
                torch_geometric.__version__
            ),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logging.info(
        "Saved tensor graph cache to %s",
        OUTPUT_PATH,
    )
    logging.info(
        "Saved graph manifest to %s",
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()