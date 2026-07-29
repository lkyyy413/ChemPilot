"""Build final RXNFP retrieval indices and evaluate test queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chempilot.reactions.retrieval import (
    ReactionEmbeddingRetriever,
)

from evaluate_reaction_retrieval_validation import (
    TOP_K_VALUES,
    condition_retrieval_metrics,
    reciprocal_rank_at_k,
)


PROTOCOLS = (
    "transformation",
    "reaction_center",
)

FEATURE_PATH = Path(
    "data/processed/reactions/"
    "features/day5/"
    "rxnfp_reaction_embeddings.npz"
)

METADATA_PATH = Path(
    "data/processed/reactions/"
    "features/day5/"
    "rxnfp_reaction_metadata.parquet"
)

SELECTION_PATH = Path(
    "reports/day5/retrieval/"
    "retrieval_validation.json"
)

INDEX_ROOT = Path(
    "data/processed/reactions/"
    "retrieval/day5"
)

REPORT_ROOT = Path(
    "reports/day5/retrieval/final"
)

SUMMARY_PATH = (
    REPORT_ROOT
    / "retrieval_test_summary.json"
)

RESULT_PATH = (
    REPORT_ROOT
    / "retrieval_test_results.csv"
)

NEIGHBOR_PATH = (
    REPORT_ROOT
    / "retrieval_test_neighbors.csv"
)


def evaluate_protocol(
    protocol,
    pooling,
    embeddings,
    metadata,
):
    split_column = (
        f"{protocol}_split"
    )

    index_indices = np.flatnonzero(
        metadata[
            split_column
        ].isin(
            [
                "train",
                "valid",
            ]
        ).to_numpy()
    )

    test_indices = np.flatnonzero(
        metadata[
            split_column
        ].eq("test").to_numpy()
    )

    index_embeddings = embeddings[
        index_indices
    ]

    index_metadata = (
        metadata.iloc[
            index_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    index_metadata[
        "retrieval_index_row"
    ] = np.arange(
        len(index_metadata),
        dtype=np.int64,
    )

    index_directory = (
        INDEX_ROOT / protocol
    )

    index_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = (
        index_directory
        / f"{pooling}_index.npz"
    )

    index_metadata_path = (
        index_directory
        / f"{pooling}_metadata.parquet"
    )

    # The final retrieval index is fixed
    # before test queries are searched.
    np.savez_compressed(
        index_path,
        embeddings=index_embeddings,
        feature_indices=(
            index_indices
        ),
    )

    index_metadata.to_parquet(
        index_metadata_path,
        index=False,
    )

    retriever = (
        ReactionEmbeddingRetriever(
            index_embeddings
        )
    )

    local_neighbors, similarities = (
        retriever.search(
            embeddings[
                test_indices
            ],
            top_k=max(TOP_K_VALUES),
        )
    )

    neighbor_indices = (
        index_indices[
            local_neighbors
        ]
    )

    query_types = metadata.iloc[
        test_indices
    ][
        "reaction_type"
    ].to_numpy()

    neighbor_types = (
        metadata.iloc[
            neighbor_indices.reshape(-1)
        ][
            "reaction_type"
        ]
        .to_numpy()
        .reshape(
            neighbor_indices.shape
        )
    )

    query_centers = metadata.iloc[
        test_indices
    ][
        "reaction_center_signature"
    ].to_numpy()

    neighbor_centers = (
        metadata.iloc[
            neighbor_indices.reshape(-1)
        ][
            "reaction_center_signature"
        ]
        .to_numpy()
        .reshape(
            neighbor_indices.shape
        )
    )

    index_centers = set(
        metadata.iloc[
            index_indices
        ][
            "reaction_center_signature"
        ]
    )

    reaction_type_results = {}

    for k in TOP_K_VALUES:
        hits = [
            query_types[row_index]
            in neighbor_types[
                row_index,
                :k,
            ]
            for row_index in range(
                len(test_indices)
            )
        ]

        reaction_type_results[
            str(k)
        ] = {
            "k": k,
            "hit_rate": float(
                np.mean(hits)
            ),
        }

    reaction_type_mrr = float(
        np.mean(
            [
                reciprocal_rank_at_k(
                    neighbor_types[
                        row_index
                    ]
                    == query_types[
                        row_index
                    ]
                )
                for row_index in range(
                    len(test_indices)
                )
            ]
        )
    )

    center_eligible = np.asarray(
        [
            center in index_centers
            for center in (
                query_centers
            )
        ],
        dtype=bool,
    )

    center_results = {
        "eligible_queries": int(
            center_eligible.sum()
        ),
        "total_queries": int(
            len(test_indices)
        ),
    }

    for k in TOP_K_VALUES:
        hits = []

        for row_index in np.flatnonzero(
            center_eligible
        ):
            hits.append(
                query_centers[row_index]
                in neighbor_centers[
                    row_index,
                    :k,
                ]
            )

        center_results[str(k)] = {
            "k": k,
            "hit_rate": (
                float(np.mean(hits))
                if hits
                else None
            ),
        }

    condition_results = {}

    for target in [
        "solvent",
        "catalyst",
    ]:
        condition_results[target] = (
            condition_retrieval_metrics(
                protocol=protocol,
                target=target,
                query_feature_indices=(
                    test_indices
                ),
                neighbor_feature_indices=(
                    neighbor_indices
                ),
                metadata=metadata,
            )
        )

    neighbor_records = []

    for query_row, query_index in (
        enumerate(test_indices)
    ):
        query = metadata.iloc[
            query_index
        ]

        for rank in range(
            neighbor_indices.shape[1]
        ):
            neighbor_index = (
                neighbor_indices[
                    query_row,
                    rank,
                ]
            )

            neighbor = metadata.iloc[
                neighbor_index
            ]

            neighbor_records.append(
                {
                    "protocol": protocol,
                    "pooling": pooling,
                    "query_feature_index": int(
                        query_index
                    ),
                    "query_transformation": (
                        query[
                            "transformation_signature"
                        ]
                    ),
                    "query_canonical_reaction": (
                        query[
                            "canonical_reaction"
                        ]
                    ),
                    "query_reaction_type": (
                        query["reaction_type"]
                    ),
                    "query_reaction_center": (
                        query[
                            "reaction_center_signature"
                        ]
                    ),
                    "rank": rank + 1,
                    "similarity": float(
                        similarities[
                            query_row,
                            rank,
                        ]
                    ),
                    "neighbor_feature_index": int(
                        neighbor_index
                    ),
                    "neighbor_transformation": (
                        neighbor[
                            "transformation_signature"
                        ]
                    ),
                    "neighbor_canonical_reaction": (
                        neighbor[
                            "canonical_reaction"
                        ]
                    ),
                    "neighbor_reaction_type": (
                        neighbor[
                            "reaction_type"
                        ]
                    ),
                    "neighbor_reaction_center": (
                        neighbor[
                            "reaction_center_signature"
                        ]
                    ),
                }
            )

    result = {
        "protocol": protocol,
        "pooling": pooling,
        "index_splits": [
            "train",
            "valid",
        ],
        "query_split": "test",
        "test_evaluations": 1,
        "test_labels_used_for_selection": (
            False
        ),
        "index_reactions": len(
            index_indices
        ),
        "test_queries": len(
            test_indices
        ),
        "nearest_similarity": {
            "minimum": float(
                similarities[:, 0].min()
            ),
            "mean": float(
                similarities[:, 0].mean()
            ),
            "median": float(
                np.median(
                    similarities[:, 0]
                )
            ),
            "maximum": float(
                similarities[:, 0].max()
            ),
        },
        "reaction_type": {
            "mrr_at_10": (
                reaction_type_mrr
            ),
            "top_k": (
                reaction_type_results
            ),
        },
        "reaction_center": (
            center_results
        ),
        "condition_retrieval": (
            condition_results
        ),
        "index_path": str(
            index_path
        ),
        "index_metadata_path": str(
            index_metadata_path
        ),
    }

    return result, neighbor_records


def main():
    if SUMMARY_PATH.exists():
        raise FileExistsError(
            "Final retrieval test summary "
            "already exists; refusing to "
            "repeat test evaluation."
        )

    with SELECTION_PATH.open(
        encoding="utf-8"
    ) as file:
        selection = json.load(file)

    assert (
        selection["test_used"]
        is False
    )

    pooling = selection[
        "selected_pooling"
    ]

    archive = np.load(
        FEATURE_PATH
    )

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    embeddings = archive[
        pooling
    ]

    INDEX_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Final reaction retrieval"
    )
    print("------------------------")
    print("Selected pooling:", pooling)
    print(
        "Index splits: train+valid"
    )
    print("Query split: test")
    print(
        "Test evaluations per protocol: 1"
    )

    results = []
    neighbor_records = []

    for protocol in PROTOCOLS:
        result, neighbors = (
            evaluate_protocol(
                protocol,
                pooling,
                embeddings,
                metadata,
            )
        )

        results.append(result)
        neighbor_records.extend(
            neighbors
        )

        print(
            "\n"
            + protocol
        )

        print(
            "  index reactions:",
            result[
                "index_reactions"
            ],
        )

        print(
            "  test queries:",
            result[
                "test_queries"
            ],
        )

        print(
            "  reaction-type Hit@5:",
            round(
                result[
                    "reaction_type"
                ][
                    "top_k"
                ]["5"]["hit_rate"],
                4,
            ),
        )

        print(
            "  reaction-type MRR@10:",
            round(
                result[
                    "reaction_type"
                ]["mrr_at_10"],
                4,
            ),
        )

        print(
            "  solvent recall@5:",
            round(
                result[
                    "condition_retrieval"
                ][
                    "solvent"
                ]["5"]["mean_recall"],
                4,
            ),
        )

        print(
            "  catalyst recall@5:",
            round(
                result[
                    "condition_retrieval"
                ][
                    "catalyst"
                ]["5"]["mean_recall"],
                4,
            ),
        )

        print(
            "  nearest similarity mean:",
            round(
                result[
                    "nearest_similarity"
                ]["mean"],
                4,
            ),
        )

    report = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "selection_source": str(
            SELECTION_PATH
        ),
        "selected_pooling": pooling,
        "test_labels_used_for_selection": (
            False
        ),
        "test_evaluations_per_protocol": 1,
        "results": results,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    rows = []

    for result in results:
        rows.append(
            {
                "protocol": (
                    result["protocol"]
                ),
                "pooling": (
                    result["pooling"]
                ),
                "index_reactions": (
                    result[
                        "index_reactions"
                    ]
                ),
                "test_queries": (
                    result[
                        "test_queries"
                    ]
                ),
                "reaction_type_hit_at_1": (
                    result[
                        "reaction_type"
                    ][
                        "top_k"
                    ]["1"]["hit_rate"]
                ),
                "reaction_type_hit_at_5": (
                    result[
                        "reaction_type"
                    ][
                        "top_k"
                    ]["5"]["hit_rate"]
                ),
                "reaction_type_hit_at_10": (
                    result[
                        "reaction_type"
                    ][
                        "top_k"
                    ]["10"]["hit_rate"]
                ),
                "reaction_type_mrr_at_10": (
                    result[
                        "reaction_type"
                    ]["mrr_at_10"]
                ),
                "solvent_recall_at_5": (
                    result[
                        "condition_retrieval"
                    ][
                        "solvent"
                    ]["5"]["mean_recall"]
                ),
                "catalyst_recall_at_5": (
                    result[
                        "condition_retrieval"
                    ][
                        "catalyst"
                    ]["5"]["mean_recall"]
                ),
                "nearest_similarity_mean": (
                    result[
                        "nearest_similarity"
                    ]["mean"]
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        RESULT_PATH,
        index=False,
    )

    pd.DataFrame(
        neighbor_records
    ).to_csv(
        NEIGHBOR_PATH,
        index=False,
    )

    print("\nSaved:", SUMMARY_PATH)
    print("Saved:", RESULT_PATH)
    print("Saved:", NEIGHBOR_PATH)


if __name__ == "__main__":
    main()