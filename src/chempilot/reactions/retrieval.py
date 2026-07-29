"""Exact cosine-similarity retrieval for reaction embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ReactionEmbeddingRetriever:
    """Retrieve nearest reactions using cosine similarity."""

    def __init__(
        self,
        index_embeddings,
    ) -> None:
        embeddings = np.asarray(
            index_embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "index_embeddings must be "
                "two-dimensional."
            )

        if len(embeddings) == 0:
            raise ValueError(
                "The retrieval index cannot "
                "be empty."
            )

        if not np.isfinite(
            embeddings
        ).all():
            raise ValueError(
                "Index embeddings contain "
                "nonfinite values."
            )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "Index embeddings contain "
                "zero vectors."
            )

        self.index_embeddings = (
            embeddings
        )

        self.normalized_index = (
            embeddings / norms
        ).astype(
            np.float32,
            copy=False,
        )

        self.number_of_items = len(
            embeddings
        )

        self.dimension = embeddings.shape[
            1
        ]

    def search(
        self,
        query_embeddings,
        *,
        top_k: int = 5,
        excluded_index_by_query: (
            Sequence[int]
            | np.ndarray
            | None
        ) = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        queries = np.asarray(
            query_embeddings,
            dtype=np.float32,
        )

        if queries.ndim == 1:
            queries = queries.reshape(
                1,
                -1,
            )

        if queries.ndim != 2:
            raise ValueError(
                "query_embeddings must be "
                "one- or two-dimensional."
            )

        if queries.shape[1] != (
            self.dimension
        ):
            raise ValueError(
                "Query and index dimensions "
                "do not match."
            )

        if not np.isfinite(
            queries
        ).all():
            raise ValueError(
                "Query embeddings contain "
                "nonfinite values."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        query_norms = np.linalg.norm(
            queries,
            axis=1,
            keepdims=True,
        )

        if np.any(query_norms == 0):
            raise ValueError(
                "Query embeddings contain "
                "zero vectors."
            )

        normalized_queries = (
            queries / query_norms
        ).astype(
            np.float32,
            copy=False,
        )

        similarities = (
            normalized_queries
            @ self.normalized_index.T
        )

        if (
            excluded_index_by_query
            is not None
        ):
            exclusions = np.asarray(
                excluded_index_by_query,
                dtype=np.int64,
            )

            if exclusions.shape != (
                len(queries),
            ):
                raise ValueError(
                    "One excluded index is "
                    "required per query."
                )

            for query_index, excluded in (
                enumerate(exclusions)
            ):
                if excluded < 0:
                    continue

                if excluded >= (
                    self.number_of_items
                ):
                    raise IndexError(
                        "Excluded index is "
                        "outside the retrieval "
                        "index."
                    )

                similarities[
                    query_index,
                    excluded,
                ] = -np.inf

        maximum_available = (
            self.number_of_items
            - (
                1
                if excluded_index_by_query
                is not None
                else 0
            )
        )

        selected_k = min(
            top_k,
            maximum_available,
        )

        order = np.argsort(
            -similarities,
            axis=1,
            kind="stable",
        )[:, :selected_k]

        selected_similarities = (
            np.take_along_axis(
                similarities,
                order,
                axis=1,
            )
        )

        return (
            order.astype(
                np.int64,
                copy=False,
            ),
            selected_similarities.astype(
                np.float32,
                copy=False,
            ),
        )