"""Command-line interface for similar-reaction retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chempilot.reactions.similarity import (
    SimilarReactionSearch,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve similar historical "
            "reactions using RXNFP."
        )
    )

    parser.add_argument(
        "--reactant",
        action="append",
        required=True,
        help=(
            "Reactant SMILES. Repeat this "
            "argument for multiple "
            "reactants."
        ),
    )

    parser.add_argument(
        "--product",
        action="append",
        required=True,
        help=(
            "Product SMILES. Repeat this "
            "argument for multiple "
            "products."
        ),
    )

    parser.add_argument(
        "--protocol",
        choices=[
            "transformation",
            "reaction_center",
        ],
        default="transformation",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device, for example "
            "cuda, cuda:0, or cpu. "
            "Defaults to automatic "
            "selection."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.top_k <= 0:
        raise ValueError(
            "top-k must be positive."
        )

    searcher = SimilarReactionSearch(
        protocol=arguments.protocol,
        device=arguments.device,
    )

    result = searcher.search(
        reactant_smiles=(
            arguments.reactant
        ),
        product_smiles=(
            arguments.product
        ),
        top_k=arguments.top_k,
    )

    rendered = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    print(rendered)

    if arguments.output is not None:
        arguments.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with arguments.output.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(rendered)
            file.write("\n")

        print(
            "\nSaved:",
            arguments.output,
        )


if __name__ == "__main__":
    main()