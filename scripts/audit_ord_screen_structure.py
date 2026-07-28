"""Audit repeated transformations and condition screens."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem

from ord_schema.proto import reaction_pb2


DATASETS = {
    "805ad863": {
        "path": Path(
            "/tmp/ord-data-source/data/80/"
            "ord_dataset-805ad863feef48579d95d86a728035f4.parquet"
        ),
        "score_type": "YIELD",
        "score_details": None,
    },
    "d9297630": {
        "path": Path(
            "/tmp/ord-data-source/data/d9/"
            "ord_dataset-d92976309c3a48a3a64a4cf5e7048086.parquet"
        ),
        "score_type": "CUSTOM",
        "score_details": "LC area percent",
    },
}

OUTPUT_PATH = Path(
    "reports/day4/"
    "ord_screen_structure_audit.json"
)


def enum_name(message, field_name: str) -> str:
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


def canonicalize_smiles(
    smiles: str,
) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def identifiers_by_type(message) -> dict:
    result = defaultdict(list)

    for identifier in message.identifiers:
        identifier_type = enum_name(
            identifier,
            "type",
        )

        value = identifier.value.strip()

        if value:
            result[identifier_type].append(
                value
            )

    return dict(result)


def component_label(message) -> str | None:
    identifiers = identifiers_by_type(
        message
    )

    for smiles in identifiers.get(
        "SMILES",
        [],
    ):
        canonical = canonicalize_smiles(
            smiles
        )

        if canonical is not None:
            return f"SMILES:{canonical}"

    for inchi in identifiers.get(
        "INCHI",
        [],
    ):
        molecule = Chem.MolFromInchi(inchi)

        if molecule is not None:
            canonical = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )

            return f"SMILES:{canonical}"

    for name in identifiers.get(
        "NAME",
        [],
    ):
        normalized_name = " ".join(
            name.lower().split()
        )

        if normalized_name not in {
            "",
            "none",
            "unknown",
            "not specified",
        }:
            return f"NAME:{normalized_name}"

    return None


def extract_score(
    reaction,
    score_type: str,
    score_details: str | None,
) -> float | None:
    values = []

    for outcome in reaction.outcomes:
        for product in outcome.products:
            if not product.is_desired_product:
                continue

            for measurement in (
                product.measurements
            ):
                if (
                    enum_name(
                        measurement,
                        "type",
                    )
                    != score_type
                ):
                    continue

                if (
                    score_details is not None
                    and measurement.details
                    != score_details
                ):
                    continue

                if measurement.HasField(
                    "percentage"
                ):
                    values.append(
                        float(
                            measurement
                            .percentage.value
                        )
                    )

    if not values:
        return None

    return max(values)


def extract_reaction_record(
    reaction,
    score_type: str,
    score_details: str | None,
) -> dict | None:
    labels_by_role = {
        "REACTANT": [],
        "SOLVENT": [],
        "CATALYST": [],
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

            if role not in labels_by_role:
                continue

            label = component_label(
                component
            )

            if label is not None:
                labels_by_role[role].append(
                    label
                )

    product_labels = []

    for outcome in reaction.outcomes:
        for product in outcome.products:
            if not product.is_desired_product:
                continue

            label = component_label(product)

            if label is not None:
                product_labels.append(label)

    reactants = tuple(
        sorted(set(
            labels_by_role["REACTANT"]
        ))
    )

    products = tuple(
        sorted(set(product_labels))
    )

    solvents = tuple(
        sorted(set(
            labels_by_role["SOLVENT"]
        ))
    )

    catalysts = tuple(
        sorted(set(
            labels_by_role["CATALYST"]
        ))
    )

    if not reactants or not products:
        return None

    transformation_signature = (
        ".".join(reactants)
        + ">>"
        + ".".join(products)
    )

    condition_signature = (
        "SOLVENT="
        + "|".join(solvents)
        + ";CATALYST="
        + "|".join(catalysts)
    )

    score = extract_score(
        reaction,
        score_type,
        score_details,
    )

    return {
        "transformation_signature": (
            transformation_signature
        ),
        "condition_signature": (
            condition_signature
        ),
        "score": score,
    }


def multiplicity_bin(value: int) -> str:
    if value == 1:
        return "1"
    if value <= 5:
        return "2-5"
    if value <= 20:
        return "6-20"
    if value <= 100:
        return "21-100"
    return ">100"


def audit_dataset(
    path: Path,
    score_type: str,
    score_details: str | None,
) -> dict:
    parquet = pq.ParquetFile(path)

    experiments_by_transformation = (
        Counter()
    )

    conditions_by_transformation = (
        defaultdict(set)
    )

    scores_by_transformation = (
        defaultdict(list)
    )

    exact_pair_counts = Counter()

    total_reactions = 0
    usable_reactions = 0
    scores = []

    for row_group_index in range(
        parquet.num_row_groups
    ):
        table = parquet.read_row_group(
            row_group_index,
            columns=["reaction"],
        )

        for serialized in (
            table["reaction"].to_pylist()
        ):
            total_reactions += 1

            reaction = reaction_pb2.Reaction()
            reaction.ParseFromString(
                serialized
            )

            record = extract_reaction_record(
                reaction,
                score_type,
                score_details,
            )

            if record is None:
                continue

            usable_reactions += 1

            transformation = record[
                "transformation_signature"
            ]

            condition = record[
                "condition_signature"
            ]

            score = record["score"]

            experiments_by_transformation[
                transformation
            ] += 1

            conditions_by_transformation[
                transformation
            ].add(condition)

            exact_pair_counts[
                (transformation, condition)
            ] += 1

            if score is not None:
                scores.append(score)

                scores_by_transformation[
                    transformation
                ].append(score)

    multiplicity_distribution = Counter(
        multiplicity_bin(count)
        for count in (
            experiments_by_transformation
            .values()
        )
    )

    condition_count_distribution = Counter(
        multiplicity_bin(len(conditions))
        for conditions in (
            conditions_by_transformation
            .values()
        )
    )

    score_array = np.asarray(
        scores,
        dtype=float,
    )

    repeated_exact_pairs = sum(
        count > 1
        for count in exact_pair_counts.values()
    )

    reactions_in_repeated_pairs = sum(
        count
        for count in exact_pair_counts.values()
        if count > 1
    )

    return {
        "total_reactions": total_reactions,
        "usable_reactions": usable_reactions,
        "unique_transformations": len(
            experiments_by_transformation
        ),
        "transformations_with_multiple_experiments": (
            sum(
                count > 1
                for count in (
                    experiments_by_transformation
                    .values()
                )
            )
        ),
        "transformations_with_multiple_conditions": (
            sum(
                len(conditions) > 1
                for conditions in (
                    conditions_by_transformation
                    .values()
                )
            )
        ),
        "experiment_multiplicity_bins": dict(
            sorted(
                multiplicity_distribution.items()
            )
        ),
        "condition_count_bins": dict(
            sorted(
                condition_count_distribution.items()
            )
        ),
        "unique_transformation_condition_pairs": (
            len(exact_pair_counts)
        ),
        "repeated_exact_pairs": (
            repeated_exact_pairs
        ),
        "reactions_in_repeated_exact_pairs": (
            reactions_in_repeated_pairs
        ),
        "score_count": int(
            score_array.size
        ),
        "score_statistics": (
            {
                "minimum": float(
                    np.min(score_array)
                ),
                "p25": float(
                    np.percentile(
                        score_array,
                        25,
                    )
                ),
                "median": float(
                    np.median(score_array)
                ),
                "p75": float(
                    np.percentile(
                        score_array,
                        75,
                    )
                ),
                "p90": float(
                    np.percentile(
                        score_array,
                        90,
                    )
                ),
                "maximum": float(
                    np.max(score_array)
                ),
                "zero_count": int(
                    np.sum(score_array == 0)
                ),
            }
            if score_array.size
            else {}
        ),
    }


def main() -> None:
    report = {}

    for dataset_name, configuration in (
        DATASETS.items()
    ):
        print(f"Auditing {dataset_name}...")

        result = audit_dataset(
            configuration["path"],
            configuration["score_type"],
            configuration["score_details"],
        )

        report[dataset_name] = result

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()