"""Predict reaction solvent and catalyst labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chempilot.reactions.inference import (
    ReactionConditionPredictor,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict Top-K solvent and "
            "catalyst labels for a reaction."
        )
    )

    parser.add_argument(
        "--reactant",
        action="append",
        required=True,
        help=(
            "Reactant SMILES. Repeat this "
            "argument for multiple reactants."
        ),
    )

    parser.add_argument(
        "--product",
        action="append",
        required=True,
        help=(
            "Product SMILES. Repeat this "
            "argument for multiple products."
        ),
    )

    parser.add_argument(
        "--protocol",
        choices=[
            "transformation",
            "reaction_center",
        ],
        default="reaction_center",
        help=(
            "Final model protocol. The "
            "reaction-center protocol is the "
            "stricter default."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of labels to return.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path. "
            "The result is always printed."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    predictor = ReactionConditionPredictor(
        protocol=arguments.protocol
    )

    result = predictor.predict(
        reactant_smiles=(
            arguments.reactant
        ),
        product_smiles=(
            arguments.product
        ),
        top_k=arguments.top_k,
    )

    serialized = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    print(serialized)

    if arguments.output is not None:
        arguments.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        arguments.output.write_text(
            serialized + "\n",
            encoding="utf-8",
        )

        print(
            "\nSaved:",
            arguments.output,
        )


if __name__ == "__main__":
    main()