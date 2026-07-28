"""Audit reaction-condition coverage in ORD Parquet files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from ord_schema.proto import reaction_pb2


ROLE_NAMES = (
    "REACTANT",
    "REAGENT",
    "SOLVENT",
    "CATALYST",
)


def enum_name(message, field_name: str) -> str:
    """Return the symbolic name of a protobuf enum field."""

    field = message.DESCRIPTOR.fields_by_name[
        field_name
    ]

    value = int(getattr(message, field_name))

    descriptor = (
        field.enum_type.values_by_number.get(
            value
        )
    )

    if descriptor is None:
        return f"UNKNOWN_{value}"

    return descriptor.name


def extract_identifiers(message) -> dict[str, list[str]]:
    """Collect non-empty chemical identifiers by type."""

    identifiers: dict[str, list[str]] = {}

    for identifier in message.identifiers:
        identifier_type = enum_name(
            identifier,
            "type",
        )

        value = identifier.value.strip()

        if not value:
            continue

        identifiers.setdefault(
            identifier_type,
            [],
        ).append(value)

    return identifiers


def chemical_label(
    identifiers: dict[str, list[str]],
) -> str | None:
    """Prefer a structural label, then fall back to a name."""

    for identifier_type in (
        "SMILES",
        "INCHI",
        "NAME",
    ):
        values = identifiers.get(
            identifier_type,
            [],
        )

        if values:
            return (
                f"{identifier_type}:"
                f"{values[0].strip()}"
            )

    return None


def audit_file(path: Path) -> dict:
    """Audit one ORD Parquet file."""

    parquet = pq.ParquetFile(path)

    counts = Counter()
    role_label_counts = {
        role: Counter()
        for role in ROLE_NAMES
    }

    role_cardinality_counts = {
        role: Counter()
        for role in ROLE_NAMES
    }

    yield_method_counts = Counter()
    temperature_unit_counts = Counter()

    for row_group_index in range(
        parquet.num_row_groups
    ):
        table = parquet.read_row_group(
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
            counts["reactions"] += 1

            reaction = reaction_pb2.Reaction()
            consumed = reaction.ParseFromString(
                serialized
            )

            if consumed != len(serialized):
                counts[
                    "partial_deserialization"
                ] += 1

            if reaction.reaction_id != reaction_id:
                counts["id_mismatch"] += 1

            labels_by_role = {
                role: set()
                for role in ROLE_NAMES
            }

            smiles_by_role = {
                role: set()
                for role in ROLE_NAMES
            }

            for reaction_input in (
                reaction.inputs.values()
            ):
                for component in (
                    reaction_input.components
                ):
                    role = enum_name(
                        component,
                        "reaction_role",
                    )

                    if role not in ROLE_NAMES:
                        continue

                    identifiers = (
                        extract_identifiers(
                            component
                        )
                    )

                    label = chemical_label(
                        identifiers
                    )

                    if label is not None:
                        labels_by_role[
                            role
                        ].add(label)

                    for smiles in identifiers.get(
                        "SMILES",
                        [],
                    ):
                        smiles_by_role[
                            role
                        ].add(smiles)

            for role in ROLE_NAMES:
                role_count = len(
                    labels_by_role[role]
                )

                role_cardinality_counts[
                    role
                ][role_count] += 1

                if role_count > 0:
                    counts[
                        f"has_{role.lower()}"
                    ] += 1

                if smiles_by_role[role]:
                    counts[
                        f"has_{role.lower()}_smiles"
                    ] += 1

                role_label_counts[
                    role
                ].update(
                    labels_by_role[role]
                )

            if (
                reaction.HasField("conditions")
                and reaction.conditions.HasField(
                    "temperature"
                )
                and reaction.conditions.temperature.HasField(
                    "setpoint"
                )
            ):
                counts["has_temperature"] += 1

                setpoint = (
                    reaction.conditions
                    .temperature.setpoint
                )

                temperature_unit_counts[
                    enum_name(setpoint, "units")
                ] += 1

            has_reaction_time = False
            has_desired_product = False
            has_desired_product_smiles = False
            has_desired_yield = False

            desired_product_labels = set()

            for outcome in reaction.outcomes:
                if outcome.HasField(
                    "reaction_time"
                ):
                    has_reaction_time = True

                for product in outcome.products:
                    if not product.is_desired_product:
                        continue

                    has_desired_product = True

                    identifiers = (
                        extract_identifiers(product)
                    )

                    label = chemical_label(
                        identifiers
                    )

                    if label is not None:
                        desired_product_labels.add(
                            label
                        )

                    if identifiers.get("SMILES"):
                        has_desired_product_smiles = (
                            True
                        )

                    for measurement in (
                        product.measurements
                    ):
                        measurement_type = (
                            enum_name(
                                measurement,
                                "type",
                            )
                        )

                        if measurement_type != "YIELD":
                            continue

                        if not measurement.HasField(
                            "percentage"
                        ):
                            continue

                        has_desired_yield = True

                        method = (
                            measurement.analysis_key
                            or measurement.details
                            or "UNSPECIFIED"
                        )

                        yield_method_counts[
                            method
                        ] += 1

            if has_reaction_time:
                counts[
                    "has_reaction_time"
                ] += 1

            if has_desired_product:
                counts[
                    "has_desired_product"
                ] += 1

            if has_desired_product_smiles:
                counts[
                    "has_desired_product_smiles"
                ] += 1

            if has_desired_yield:
                counts[
                    "has_desired_yield"
                ] += 1

            if desired_product_labels:
                counts[
                    "has_desired_product_label"
                ] += 1

    total = counts["reactions"]

    def rate(key: str) -> float:
        if total == 0:
            return 0.0

        return counts[key] / total

    result = {
        "path": str(path),
        "number_of_reactions": total,
        "integrity": {
            "partial_deserialization": counts[
                "partial_deserialization"
            ],
            "reaction_id_mismatch": counts[
                "id_mismatch"
            ],
        },
        "coverage": {
            key: {
                "count": counts[key],
                "rate": rate(key),
            }
            for key in (
                "has_reactant",
                "has_reactant_smiles",
                "has_reagent",
                "has_reagent_smiles",
                "has_solvent",
                "has_solvent_smiles",
                "has_catalyst",
                "has_catalyst_smiles",
                "has_temperature",
                "has_reaction_time",
                "has_desired_product",
                "has_desired_product_smiles",
                "has_desired_yield",
            )
        },
        "unique_labels": {
            role.lower(): len(
                role_label_counts[role]
            )
            for role in ROLE_NAMES
        },
        "top_labels": {
            role.lower(): (
                role_label_counts[role]
                .most_common(10)
            )
            for role in ROLE_NAMES
        },
        "role_cardinality": {
            role.lower(): dict(
                sorted(
                    role_cardinality_counts[
                        role
                    ].items()
                )
            )
            for role in ROLE_NAMES
        },
        "temperature_units": dict(
            temperature_unit_counts
        ),
        "top_yield_methods": (
            yield_method_counts.most_common(10)
        ),
    }

    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(
            "/tmp/ord-data-source"
        ),
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/day4/"
            "ord_candidate_audit.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    results = []

    for relative_path in arguments.inputs:
        path = (
            arguments.repository
            / relative_path
        )

        print(f"Auditing {relative_path}...")

        result = audit_file(path)

        results.append(result)

        coverage = result["coverage"]

        print(
            "  reactions=",
            result["number_of_reactions"],
            "solvent=",
            round(
                coverage[
                    "has_solvent"
                ]["rate"],
                4,
            ),
            "catalyst=",
            round(
                coverage[
                    "has_catalyst"
                ]["rate"],
                4,
            ),
            "product_smiles=",
            round(
                coverage[
                    "has_desired_product_smiles"
                ]["rate"],
                4,
            ),
        )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "number_of_files": len(results),
        "files": results,
    }

    arguments.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved audit report to",
        arguments.output,
    )


if __name__ == "__main__":
    main()