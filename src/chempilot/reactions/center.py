"""Reaction-center proxy derived from RDKit difference fingerprints."""

from __future__ import annotations

import hashlib
import json

from rdkit.Chem import rdChemReactions


def reaction_center_elements(
    reaction_smiles_mapped: str,
) -> tuple[tuple[int, int], ...]:
    """Return non-zero RDKit reaction-difference features."""

    if not isinstance(
        reaction_smiles_mapped,
        str,
    ):
        raise TypeError(
            "Reaction SMILES must be a string."
        )

    reaction_smiles_mapped = (
        reaction_smiles_mapped.strip()
    )

    if not reaction_smiles_mapped:
        raise ValueError(
            "Reaction SMILES is empty."
        )

    reaction = (
        rdChemReactions.ReactionFromSmarts(
            reaction_smiles_mapped,
            useSmiles=True,
        )
    )

    if reaction is None:
        raise ValueError(
            "RDKit could not parse reaction SMILES."
        )

    fingerprint = (
        rdChemReactions
        .CreateDifferenceFingerprintForReaction(
            reaction
        )
    )

    return tuple(
        sorted(
            (
                int(index),
                int(value),
            )
            for index, value in (
                fingerprint
                .GetNonzeroElements()
                .items()
            )
        )
    )


def reaction_center_signature(
    reaction_smiles_mapped: str,
    length: int = 16,
) -> str:
    """Hash a deterministic reaction-difference fingerprint."""

    if not 8 <= length <= 64:
        raise ValueError(
            "Signature length must be "
            "between 8 and 64."
        )

    elements = reaction_center_elements(
        reaction_smiles_mapped
    )

    serialized = json.dumps(
        elements,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:length]