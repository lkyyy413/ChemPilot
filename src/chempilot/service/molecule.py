"""Molecular standardization and lightweight risk assessment."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Contrib.SA_Score import (
    sascorer,
)

from .errors import InvalidSmilesError
from .schemas import (
    DrugLikenessAssessment,
    MolecularDescriptors,
    SynthesizabilityRisk,
)


@dataclass(frozen=True)
class SynthesizabilityConfig:
    """Thresholds for lightweight synthesis-risk rules."""

    rare_fragment_score_threshold: float = -2.5

    moderate_sa_score: float = 4.0
    high_sa_score: float = 6.0

    moderate_complexity: float = 500.0
    high_complexity: float = 1000.0

    moderate_rare_fragments: int = 1
    high_rare_fragments: int = 3

    moderate_history_similarity: float = 0.50
    low_history_similarity: float = 0.30

    moderate_condition_uncertainty: float = 0.35
    high_condition_uncertainty: float = 0.60


class MoleculeStandardizer:
    """Validate and canonicalize molecular SMILES."""

    def standardize(
        self,
        smiles: str,
        *,
        field: str = "molecule_smiles",
    ) -> tuple[str, Chem.Mol]:
        if not isinstance(smiles, str):
            raise InvalidSmilesError(
                (
                    f"{field} must be a "
                    "SMILES string."
                ),
                field=field,
                context={
                    "received_type": (
                        type(smiles).__name__
                    ),
                },
            )

        normalized = smiles.strip()

        if not normalized:
            raise InvalidSmilesError(
                (
                    f"{field} must not "
                    "be empty."
                ),
                field=field,
            )

        molecule = Chem.MolFromSmiles(
            normalized
        )

        if molecule is None:
            raise InvalidSmilesError(
                (
                    f"{field} could not "
                    "be parsed as SMILES."
                ),
                field=field,
                context={
                    "input_smiles": normalized,
                },
            )

        canonical = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )

        canonical_molecule = (
            Chem.MolFromSmiles(
                canonical
            )
        )

        if canonical_molecule is None:
            raise InvalidSmilesError(
                (
                    f"{field} failed "
                    "canonicalization."
                ),
                field=field,
            )

        return (
            canonical,
            canonical_molecule,
        )


class MoleculeRiskAnalyzer:
    """Compute descriptors and lightweight synthesis risk."""

    def __init__(
        self,
        config: (
            SynthesizabilityConfig
            | None
        ) = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else SynthesizabilityConfig()
        )

        self._morgan_generator = (
            rdFingerprintGenerator
            .GetMorganGenerator(
                radius=2,
            )
        )

        sascorer.readFragmentScores()

    def descriptors(
        self,
        molecule: Chem.Mol,
    ) -> MolecularDescriptors:
        return MolecularDescriptors(
            molecular_weight=float(
                Descriptors.MolWt(
                    molecule
                )
            ),
            log_p=float(
                Crippen.MolLogP(
                    molecule
                )
            ),
            tpsa=float(
                rdMolDescriptors
                .CalcTPSA(
                    molecule
                )
            ),
            hydrogen_bond_donors=int(
                Lipinski.NumHDonors(
                    molecule
                )
            ),
            hydrogen_bond_acceptors=int(
                Lipinski.NumHAcceptors(
                    molecule
                )
            ),
            rotatable_bonds=int(
                Lipinski
                .NumRotatableBonds(
                    molecule
                )
            ),
            ring_count=int(
                Lipinski.RingCount(
                    molecule
                )
            ),
            heavy_atom_count=int(
                Lipinski
                .HeavyAtomCount(
                    molecule
                )
            ),
            fraction_csp3=float(
                rdMolDescriptors
                .CalcFractionCSP3(
                    molecule
                )
            ),
        )

    def drug_likeness(
        self,
        descriptors: (
            MolecularDescriptors
        ),
    ) -> DrugLikenessAssessment:
        rule_results = {
            "molecular_weight_le_500": (
                descriptors
                .molecular_weight
                <= 500.0
            ),
            "log_p_le_5": (
                descriptors.log_p
                <= 5.0
            ),
            "hbd_le_5": (
                descriptors
                .hydrogen_bond_donors
                <= 5
            ),
            "hba_le_10": (
                descriptors
                .hydrogen_bond_acceptors
                <= 10
            ),
        }

        violations = sum(
            not passed
            for passed
            in rule_results.values()
        )

        passed = violations == 0

        if passed:
            interpretation = (
                "Passes the four Lipinski "
                "Rule-of-Five criteria."
            )
        else:
            interpretation = (
                f"Violates {violations} of "
                "the four Lipinski "
                "Rule-of-Five criteria."
            )

        return DrugLikenessAssessment(
            lipinski_pass=passed,
            lipinski_violations=(
                violations
            ),
            rule_results=rule_results,
            interpretation=(
                interpretation
            ),
        )

    def sa_score(
        self,
        molecule: Chem.Mol,
    ) -> float:
        score = float(
            sascorer.calculateScore(
                molecule
            )
        )

        return min(
            10.0,
            max(
                1.0,
                score,
            ),
        )

    def rare_fragment_count(
        self,
        molecule: Chem.Mol,
    ) -> int:
        fingerprint = (
            self._morgan_generator
            .GetSparseCountFingerprint(
                molecule
            )
        )

        fragments = (
            fingerprint
            .GetNonzeroElements()
        )

        fragment_scores = (
            sascorer._fscores
        )

        threshold = (
            self.config
            .rare_fragment_score_threshold
        )

        count = 0

        for fragment_id, occurrence in (
            fragments.items()
        ):
            fragment_score = (
                fragment_scores.get(
                    fragment_id,
                    -4.0,
                )
            )

            if (
                fragment_score
                <= threshold
            ):
                count += int(
                    occurrence
                )

        return count

    def molecular_complexity(
        self,
        molecule: Chem.Mol,
    ) -> float:
        return float(
            Descriptors.BertzCT(
                molecule
            )
        )

    def synthesizability(
        self,
        molecule: Chem.Mol,
        *,
        historical_similarity: (
            float
            | None
        ) = None,
        condition_uncertainty: (
            float
            | None
        ) = None,
    ) -> SynthesizabilityRisk:
        if (
            historical_similarity
            is not None
            and not (
                0.0
                <= historical_similarity
                <= 1.0
            )
        ):
            raise ValueError(
                "historical_similarity "
                "must be in [0, 1]."
            )

        if (
            condition_uncertainty
            is not None
            and not (
                0.0
                <= condition_uncertainty
                <= 1.0
            )
        ):
            raise ValueError(
                "condition_uncertainty "
                "must be in [0, 1]."
            )

        sa_score = self.sa_score(
            molecule
        )

        rare_count = (
            self.rare_fragment_count(
                molecule
            )
        )

        complexity = (
            self.molecular_complexity(
                molecule
            )
        )

        moderate_factors = []
        high_factors = []

        if (
            sa_score
            >= self.config.high_sa_score
        ):
            high_factors.append(
                (
                    "SA score exceeds the "
                    "high-risk threshold."
                )
            )
        elif (
            sa_score
            >= self.config
            .moderate_sa_score
        ):
            moderate_factors.append(
                (
                    "SA score exceeds the "
                    "moderate-risk threshold."
                )
            )

        if (
            rare_count
            >= self.config
            .high_rare_fragments
        ):
            high_factors.append(
                (
                    "Multiple low-frequency "
                    "SA fragments were found."
                )
            )
        elif (
            rare_count
            >= self.config
            .moderate_rare_fragments
        ):
            moderate_factors.append(
                (
                    "At least one "
                    "low-frequency SA "
                    "fragment was found."
                )
            )

        if (
            complexity
            >= self.config
            .high_complexity
        ):
            high_factors.append(
                (
                    "Bertz complexity exceeds "
                    "the high-risk threshold."
                )
            )
        elif (
            complexity
            >= self.config
            .moderate_complexity
        ):
            moderate_factors.append(
                (
                    "Bertz complexity exceeds "
                    "the moderate-risk "
                    "threshold."
                )
            )

        if historical_similarity is not None:
            if (
                historical_similarity
                < self.config
                .low_history_similarity
            ):
                high_factors.append(
                    (
                        "No close historical "
                        "reaction precedent was "
                        "retrieved."
                    )
                )
            elif (
                historical_similarity
                < self.config
                .moderate_history_similarity
            ):
                moderate_factors.append(
                    (
                        "Historical reaction "
                        "similarity is limited."
                    )
                )

        if condition_uncertainty is not None:
            if (
                condition_uncertainty
                >= self.config
                .high_condition_uncertainty
            ):
                high_factors.append(
                    (
                        "Condition-model "
                        "uncertainty is high."
                    )
                )
            elif (
                condition_uncertainty
                >= self.config
                .moderate_condition_uncertainty
            ):
                moderate_factors.append(
                    (
                        "Condition-model "
                        "uncertainty is "
                        "moderate."
                    )
                )

        if high_factors:
            risk_level = "high"
            risk_factors = (
                high_factors
                + moderate_factors
            )
            interpretation = (
                "One or more configured "
                "high-risk indicators were "
                "triggered."
            )

        elif moderate_factors:
            risk_level = "moderate"
            risk_factors = (
                moderate_factors
            )
            interpretation = (
                "No high-risk indicator was "
                "triggered, but at least one "
                "moderate-risk indicator was "
                "found."
            )

        else:
            risk_level = "low"
            risk_factors = []
            interpretation = (
                "No configured structural or "
                "contextual risk threshold was "
                "triggered."
            )

        return SynthesizabilityRisk(
            sa_score=sa_score,
            rare_fragment_count=(
                rare_count
            ),
            molecular_complexity=(
                complexity
            ),
            historical_similarity=(
                historical_similarity
            ),
            condition_uncertainty=(
                condition_uncertainty
            ),
            risk_level=risk_level,
            risk_factors=risk_factors,
            interpretation=(
                interpretation
            ),
        )