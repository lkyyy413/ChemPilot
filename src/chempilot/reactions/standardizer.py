"""Chemical identity and reaction standardization utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rdkit import Chem


MISSING_NAMES = {
    "",
    "none",
    "unknown",
    "unspecified",
    "not specified",
    "n/a",
    "na",
}

NAME_ALIASES = {
    "acetonitrile": "acetonitrile",
    "mecn": "acetonitrile",
    "dimethyl sulfoxide": "dimethyl sulfoxide",
    "dmso": "dimethyl sulfoxide",
    "n,n-dimethylformamide": (
        "n,n-dimethylformamide"
    ),
    "dimethylformamide": (
        "n,n-dimethylformamide"
    ),
    "dmf": "n,n-dimethylformamide",
    "tetrahydrofuran": "tetrahydrofuran",
    "thf": "tetrahydrofuran",
    "1,4-dioxane": "1,4-dioxane",
    "dioxane": "1,4-dioxane",
    "toluene": "toluene",
    "methanol": "methanol",
    "meoh": "methanol",
    "ethanol": "ethanol",
    "etoh": "ethanol",
}


@dataclass(frozen=True)
class ChemicalIdentity:
    """A normalized chemical identity."""

    label: str
    canonical_smiles: str | None
    normalized_name: str | None
    source_type: str


def enum_name(
    message,
    field_name: str,
) -> str:
    """Return the symbolic name of a protobuf enum."""

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


class ReactionStandardizer:
    """Standardize ORD chemicals and reaction signatures."""

    def canonicalize_smiles(
        self,
        smiles: str,
    ) -> str | None:
        """Canonicalize SMILES while preserving chemistry."""

        smiles = smiles.strip()

        if not smiles:
            return None

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

    def canonicalize_inchi(
        self,
        inchi: str,
    ) -> str | None:
        """Convert an InChI identifier to canonical SMILES."""

        inchi = inchi.strip()

        if not inchi:
            return None

        molecule = Chem.MolFromInchi(inchi)

        if molecule is None:
            return None

        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)

        return Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )

    def normalize_name(
        self,
        name: str,
    ) -> str | None:
        """Normalize spacing, case, and common aliases."""

        normalized = name.casefold().strip()

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        if normalized in MISSING_NAMES:
            return None

        return NAME_ALIASES.get(
            normalized,
            normalized,
        )

    def identity_from_message(
        self,
        message,
    ) -> ChemicalIdentity | None:
        """Resolve one ORD compound or product identity."""

        identifiers: dict[str, list[str]] = {}

        for identifier in message.identifiers:
            identifier_type = enum_name(
                identifier,
                "type",
            )

            value = identifier.value.strip()

            if value:
                identifiers.setdefault(
                    identifier_type,
                    [],
                ).append(value)

        for smiles in identifiers.get(
            "SMILES",
            [],
        ):
            canonical = (
                self.canonicalize_smiles(
                    smiles
                )
            )

            if canonical is not None:
                return ChemicalIdentity(
                    label=f"SMILES:{canonical}",
                    canonical_smiles=canonical,
                    normalized_name=None,
                    source_type="SMILES",
                )

        for inchi in identifiers.get(
            "INCHI",
            [],
        ):
            canonical = (
                self.canonicalize_inchi(
                    inchi
                )
            )

            if canonical is not None:
                return ChemicalIdentity(
                    label=f"SMILES:{canonical}",
                    canonical_smiles=canonical,
                    normalized_name=None,
                    source_type="INCHI",
                )

        for name in identifiers.get(
            "NAME",
            [],
        ):
            normalized_name = (
                self.normalize_name(name)
            )

            if normalized_name is not None:
                return ChemicalIdentity(
                    label=(
                        f"NAME:{normalized_name}"
                    ),
                    canonical_smiles=None,
                    normalized_name=(
                        normalized_name
                    ),
                    source_type="NAME",
                )

        return None

    def role_identities(
        self,
        reaction,
        role_name: str,
    ) -> tuple[ChemicalIdentity, ...]:
        """Return unique identities assigned to an input role."""

        identities = {}

        for reaction_input in (
            reaction.inputs.values()
        ):
            for component in (
                reaction_input.components
            ):
                component_role = enum_name(
                    component,
                    "reaction_role",
                )

                if component_role != role_name:
                    continue

                identity = (
                    self.identity_from_message(
                        component
                    )
                )

                if identity is not None:
                    identities[
                        identity.label
                    ] = identity

        return tuple(
            identities[label]
            for label in sorted(identities)
        )

    def desired_product_identities(
        self,
        reaction,
    ) -> tuple[ChemicalIdentity, ...]:
        """Return unique explicitly marked desired products."""

        identities = {}

        for outcome in reaction.outcomes:
            for product in outcome.products:
                if not product.is_desired_product:
                    continue

                identity = (
                    self.identity_from_message(
                        product
                    )
                )

                if identity is not None:
                    identities[
                        identity.label
                    ] = identity

        return tuple(
            identities[label]
            for label in sorted(identities)
        )

    def transformation_signature(
        self,
        reaction,
    ) -> str | None:
        """Build a condition-independent reaction signature."""

        reactants = self.role_identities(
            reaction,
            "REACTANT",
        )

        products = (
            self.desired_product_identities(
                reaction
            )
        )

        if not reactants or not products:
            return None

        reactant_side = ".".join(
            identity.label
            for identity in reactants
        )

        product_side = ".".join(
            identity.label
            for identity in products
        )

        return (
            reactant_side
            + ">>"
            + product_side
        )