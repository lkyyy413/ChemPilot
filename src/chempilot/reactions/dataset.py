"""Standardized streaming interface for ORD reaction data."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from ord_schema.proto import reaction_pb2

from .standardizer import (
    ReactionStandardizer,
    enum_name,
)


@dataclass(frozen=True)
class ScorePolicy:
    """Select one product measurement as the experiment score."""

    measurement_type: str
    details: str | None = None
    score_name: str = "score"


@dataclass(frozen=True)
class ReactionRecord:
    """One standardized reaction-condition experiment."""

    reaction_id: str
    source_dataset: str
    reaction_type: str | None
    reaction_smiles_mapped: str | None
    transformation_signature: str

    reactant_labels: tuple[str, ...]
    product_labels: tuple[str, ...]
    reagent_labels: tuple[str, ...]
    solvent_labels: tuple[str, ...]
    catalyst_labels: tuple[str, ...]

    reactant_smiles: tuple[str, ...]
    product_smiles: tuple[str, ...]
    reagent_smiles: tuple[str, ...]
    solvent_smiles: tuple[str, ...]
    catalyst_smiles: tuple[str, ...]

    condition_signature: str
    temperature_celsius: float | None
    reaction_time_hours: float | None

    score: float | None
    score_name: str

    def to_dict(self) -> dict:
        """Convert the immutable record to a serializable dictionary."""

        return asdict(self)


class ReactionDataset:
    """Stream standardized reactions from an ORD Parquet file."""

    def __init__(
        self,
        parquet_path: str | Path,
        source_dataset: str,
        score_policy: ScorePolicy,
        standardizer: ReactionStandardizer | None = None,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.source_dataset = source_dataset
        self.score_policy = score_policy

        self.standardizer = (
            standardizer
            if standardizer is not None
            else ReactionStandardizer()
        )

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                self.parquet_path
            )

        self.parquet = pq.ParquetFile(
            self.parquet_path
        )

    def __len__(self) -> int:
        return self.parquet.metadata.num_rows

    def _identifier_values(
        self,
        message,
    ) -> dict[str, list[str]]:
        values: dict[str, list[str]] = {}

        for identifier in message.identifiers:
            identifier_type = enum_name(
                identifier,
                "type",
            )

            value = identifier.value.strip()

            if value:
                values.setdefault(
                    identifier_type,
                    [],
                ).append(value)

        return values

    def _reaction_metadata(
        self,
        reaction,
    ) -> tuple[str | None, str | None]:
        identifiers = self._identifier_values(
            reaction
        )

        reaction_types = identifiers.get(
            "REACTION_TYPE",
            [],
        )

        reaction_smiles = identifiers.get(
            "REACTION_SMILES",
            [],
        )

        return (
            reaction_types[0]
            if reaction_types
            else None,
            reaction_smiles[0]
            if reaction_smiles
            else None,
        )

    def _temperature_celsius(
        self,
        reaction,
    ) -> float | None:
        if not reaction.HasField("conditions"):
            return None

        temperature = (
            reaction.conditions.temperature
        )

        if not reaction.conditions.HasField(
            "temperature"
        ):
            return None

        if not temperature.HasField("setpoint"):
            return None

        setpoint = temperature.setpoint
        value = float(setpoint.value)
        if not math.isfinite(value):
            return None
        units = enum_name(setpoint, "units")

        if units == "CELSIUS":
            return value

        if units == "KELVIN":
            return value - 273.15

        if units == "FAHRENHEIT":
            return (value - 32.0) * 5.0 / 9.0

        return None

    def _reaction_time_hours(
        self,
        reaction,
    ) -> float | None:
        """Return the latest recorded outcome time in hours."""

        values = []

        for outcome in reaction.outcomes:
            if not outcome.HasField(
                "reaction_time"
            ):
                continue

            reaction_time = (
                outcome.reaction_time
            )

            value = float(
                reaction_time.value
            )

            if not math.isfinite(value):
                continue

            units = enum_name(
                reaction_time,
                "units",
            )

            if units in {
                "HOUR",
                "HOURS",
            }:
                hours = value

            elif units in {
                "MINUTE",
                "MINUTES",
            }:
                hours = value / 60.0

            elif units in {
                "SECOND",
                "SECONDS",
            }:
                hours = value / 3600.0

            elif units in {
                "DAY",
                "DAYS",
            }:
                hours = value * 24.0

            else:
                continue

            values.append(hours)

        if not values:
            return None

        return max(values)

    def _extract_score(
        self,
        reaction,
    ) -> float | None:
        scores = []

        for outcome in reaction.outcomes:
            for product in outcome.products:
                if not product.is_desired_product:
                    continue

                for measurement in (
                    product.measurements
                ):
                    measurement_type = enum_name(
                        measurement,
                        "type",
                    )

                    if (
                        measurement_type
                        != self.score_policy.measurement_type
                    ):
                        continue

                    if (
                        self.score_policy.details
                        is not None
                        and measurement.details
                        != self.score_policy.details
                    ):
                        continue

                    if measurement.HasField(
                        "percentage"
                    ):
                        value = float(
                            measurement
                            .percentage.value
                        )

                        if math.isfinite(value):
                            scores.append(value)

        if not scores:
            return None

        return max(scores)

    def _labels_for_role(
        self,
        reaction,
        role_name: str,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
    ]:
        identities = (
            self.standardizer.role_identities(
                reaction,
                role_name,
            )
        )

        labels = tuple(
            identity.label
            for identity in identities
        )

        smiles = tuple(
            identity.canonical_smiles
            for identity in identities
            if identity.canonical_smiles
            is not None
        )

        return labels, smiles

    def _desired_products(
        self,
        reaction,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
    ]:
        identities = (
            self.standardizer
            .desired_product_identities(
                reaction
            )
        )

        labels = tuple(
            identity.label
            for identity in identities
        )

        smiles = tuple(
            identity.canonical_smiles
            for identity in identities
            if identity.canonical_smiles
            is not None
        )

        return labels, smiles

    def standardize_reaction(
        self,
        reaction,
    ) -> ReactionRecord | None:
        """Convert one ORD reaction to a standardized record."""

        transformation_signature = (
            self.standardizer
            .transformation_signature(
                reaction
            )
        )

        if transformation_signature is None:
            return None

        (
            reactant_labels,
            reactant_smiles,
        ) = self._labels_for_role(
            reaction,
            "REACTANT",
        )

        (
            reagent_labels,
            reagent_smiles,
        ) = self._labels_for_role(
            reaction,
            "REAGENT",
        )

        (
            solvent_labels,
            solvent_smiles,
        ) = self._labels_for_role(
            reaction,
            "SOLVENT",
        )

        (
            catalyst_labels,
            catalyst_smiles,
        ) = self._labels_for_role(
            reaction,
            "CATALYST",
        )

        (
            product_labels,
            product_smiles,
        ) = self._desired_products(
            reaction
        )

        (
            reaction_type,
            reaction_smiles_mapped,
        ) = self._reaction_metadata(
            reaction
        )

        temperature_celsius = (
            self._temperature_celsius(
                reaction
            )
        )

        reaction_time_hours = (
            self._reaction_time_hours(
                reaction
            )
        )

        temperature_label = (
            "MISSING"
            if temperature_celsius is None
            else f"{temperature_celsius:.6g}"
        )

        time_label = (
            "MISSING"
            if reaction_time_hours is None
            else f"{reaction_time_hours:.6g}"
        )

        condition_signature = (
            "SOLVENT="
            + "|".join(solvent_labels)
            + ";CATALYST="
            + "|".join(catalyst_labels)
            + ";REAGENT="
            + "|".join(reagent_labels)
            + ";TEMP_C="
            + temperature_label
            + ";TIME_H="
            + time_label
        )


        return ReactionRecord(
            reaction_id=reaction.reaction_id,
            source_dataset=(
                self.source_dataset
            ),
            reaction_type=reaction_type,
            reaction_smiles_mapped=(
                reaction_smiles_mapped
            ),
            transformation_signature=(
                transformation_signature
            ),
            reactant_labels=(
                reactant_labels
            ),
            product_labels=product_labels,
            reagent_labels=reagent_labels,
            solvent_labels=solvent_labels,
            catalyst_labels=catalyst_labels,
            reactant_smiles=reactant_smiles,
            product_smiles=product_smiles,
            reagent_smiles=reagent_smiles,
            solvent_smiles=solvent_smiles,
            catalyst_smiles=catalyst_smiles,
            condition_signature=(
                condition_signature
            ),
            temperature_celsius=(
                temperature_celsius
            ),
            reaction_time_hours=(
                reaction_time_hours
            ),
            score=self._extract_score(
                reaction
            ),
            score_name=(
                self.score_policy.score_name
            ),
        )

    def iter_records(
        self,
        limit: int | None = None,
    ) -> Iterator[ReactionRecord]:
        """Yield standardized records in source row order."""

        yielded = 0

        for row_group_index in range(
            self.parquet.num_row_groups
        ):
            table = self.parquet.read_row_group(
                row_group_index,
                columns=[
                    "reaction_id",
                    "reaction",
                ],
            )

            reaction_ids = table[
                "reaction_id"
            ].to_pylist()

            serialized_reactions = table[
                "reaction"
            ].to_pylist()

            for reaction_id, serialized in zip(
                reaction_ids,
                serialized_reactions,
                strict=True,
            ):
                reaction = (
                    reaction_pb2.Reaction()
                )

                consumed = (
                    reaction.ParseFromString(
                        serialized
                    )
                )

                if consumed != len(serialized):
                    raise ValueError(
                        "Partial protobuf "
                        "deserialization for "
                        f"{reaction_id}."
                    )

                if (
                    reaction.reaction_id
                    != reaction_id
                ):
                    raise ValueError(
                        "Reaction ID mismatch: "
                        f"{reaction_id}."
                    )

                record = (
                    self.standardize_reaction(
                        reaction
                    )
                )

                if record is None:
                    continue

                yield record
                yielded += 1

                if (
                    limit is not None
                    and yielded >= limit
                ):
                    return