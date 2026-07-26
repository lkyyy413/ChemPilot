"""Edge-aware Graph Isomorphism Network for molecular regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    global_add_pool,
    global_mean_pool,
)

from chempilot.features.graph import (
    ATOM_FEATURE_CARDINALITIES,
    BOND_FEATURE_CARDINALITIES,
)


@dataclass(frozen=True)
class GINEConfig:
    """Serializable GINE architecture configuration."""

    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.15
    mlp_expansion: int = 2
    train_eps: bool = True
    pooling: str = "add"

    def to_dict(self) -> dict:
        return asdict(self)


class CategoricalFeatureEncoder(nn.Module):
    """Embed multiple categorical columns and sum them."""

    def __init__(
        self,
        cardinalities: Sequence[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()

        if not cardinalities:
            raise ValueError(
                "At least one categorical feature is required."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        self.cardinalities = [
            int(value)
            for value in cardinalities
        ]
        self.embedding_dim = int(embedding_dim)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=cardinality,
                    embedding_dim=embedding_dim,
                )
                for cardinality in self.cardinalities
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(
                embedding.weight
            )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "Categorical features must be a "
                "two-dimensional tensor."
            )

        if features.shape[1] != len(
            self.embeddings
        ):
            raise ValueError(
                f"Expected {len(self.embeddings)} "
                f"feature columns, received "
                f"{features.shape[1]}."
            )

        if features.dtype != torch.long:
            raise TypeError(
                "Categorical feature tensor must "
                "have dtype torch.long."
            )

        encoded = self.embeddings[
            0
        ].weight.new_zeros(
            (
                features.shape[0],
                self.embedding_dim,
            )
        )

        for column, embedding in enumerate(
            self.embeddings
        ):
            encoded = encoded + embedding(
                features[:, column]
            )

        return encoded


class GINERegressor(nn.Module):
    """GINE graph-level regression model."""

    def __init__(
        self,
        config: GINEConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or GINEConfig()

        if self.config.num_layers < 1:
            raise ValueError(
                "num_layers must be at least one."
            )

        if not 0.0 <= self.config.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if self.config.pooling not in {
            "add",
            "mean",
        }:
            raise ValueError(
                "pooling must be 'add' or 'mean'."
            )

        hidden_dim = self.config.hidden_dim
        expanded_dim = (
            hidden_dim
            * self.config.mlp_expansion
        )

        self.node_encoder = (
            CategoricalFeatureEncoder(
                cardinalities=(
                    ATOM_FEATURE_CARDINALITIES
                ),
                embedding_dim=hidden_dim,
            )
        )

        self.edge_encoder = (
            CategoricalFeatureEncoder(
                cardinalities=(
                    BOND_FEATURE_CARDINALITIES
                ),
                embedding_dim=hidden_dim,
            )
        )

        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        for _ in range(self.config.num_layers):
            update_mlp = nn.Sequential(
                nn.Linear(
                    hidden_dim,
                    expanded_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    expanded_dim,
                    hidden_dim,
                ),
            )

            convolution = GINEConv(
                nn=update_mlp,
                train_eps=self.config.train_eps,
            )

            self.convolutions.append(convolution)
            self.normalizations.append(
                nn.LayerNorm(hidden_dim)
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(
        self,
        data,
    ) -> torch.Tensor:
        node_embeddings = self.node_encoder(
            data.x
        )
        edge_embeddings = self.edge_encoder(
            data.edge_attr
        )

        hidden = node_embeddings

        for convolution, normalization in zip(
            self.convolutions,
            self.normalizations,
        ):
            update = convolution(
                hidden,
                data.edge_index,
                edge_embeddings,
            )

            update = normalization(update)
            update = F.relu(update)
            update = F.dropout(
                update,
                p=self.config.dropout,
                training=self.training,
            )

            hidden = hidden + update

        return hidden

    def forward(self, data) -> torch.Tensor:
        hidden = self.encode_nodes(data)

        batch = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(
                hidden.shape[0],
                dtype=torch.long,
                device=hidden.device,
            )

        if self.config.pooling == "add":
            graph_embeddings = global_add_pool(
                hidden,
                batch,
            )
        else:
            graph_embeddings = global_mean_pool(
                hidden,
                batch,
            )

        predictions = self.readout(
            graph_embeddings
        )

        return predictions.reshape(-1)

    def parameter_count(
        self,
        trainable_only: bool = True,
    ) -> int:
        parameters = self.parameters()

        if trainable_only:
            parameters = (
                parameter
                for parameter in parameters
                if parameter.requires_grad
            )

        return sum(
            parameter.numel()
            for parameter in parameters
        )