import pytest

from chempilot.service.errors import (
    InvalidSmilesError,
)
from chempilot.service.molecule import (
    MoleculeRiskAnalyzer,
    MoleculeStandardizer,
    SynthesizabilityConfig,
)


def test_canonical_smiles():
    standardizer = (
        MoleculeStandardizer()
    )

    canonical, molecule = (
        standardizer.standardize(
            "OCC"
        )
    )

    assert canonical == "CCO"
    assert molecule.GetNumAtoms() == 3


def test_stereochemistry_is_preserved():
    standardizer = (
        MoleculeStandardizer()
    )

    canonical, _ = (
        standardizer.standardize(
            "N[C@@H](C)C(=O)O"
        )
    )

    assert "@" in canonical


def test_invalid_smiles_raises():
    standardizer = (
        MoleculeStandardizer()
    )

    with pytest.raises(
        InvalidSmilesError,
        match="could not be parsed",
    ):
        standardizer.standardize(
            "not-a-smiles"
        )


def test_ethanol_descriptors():
    standardizer = (
        MoleculeStandardizer()
    )

    analyzer = (
        MoleculeRiskAnalyzer()
    )

    _, molecule = (
        standardizer.standardize(
            "CCO"
        )
    )

    result = analyzer.descriptors(
        molecule
    )

    assert result.molecular_weight == (
        pytest.approx(
            46.069,
            abs=0.01,
        )
    )

    assert (
        result.hydrogen_bond_donors
        == 1
    )

    assert (
        result.hydrogen_bond_acceptors
        == 1
    )

    assert result.heavy_atom_count == 3


def test_ethanol_passes_lipinski():
    standardizer = (
        MoleculeStandardizer()
    )

    analyzer = (
        MoleculeRiskAnalyzer()
    )

    _, molecule = (
        standardizer.standardize(
            "CCO"
        )
    )

    descriptors = (
        analyzer.descriptors(
            molecule
        )
    )

    result = analyzer.drug_likeness(
        descriptors
    )

    assert result.lipinski_pass
    assert result.lipinski_violations == 0
    assert all(
        result.rule_results.values()
    )


def test_sa_score_is_bounded():
    standardizer = (
        MoleculeStandardizer()
    )

    analyzer = (
        MoleculeRiskAnalyzer()
    )

    _, molecule = (
        standardizer.standardize(
            "CCO"
        )
    )

    score = analyzer.sa_score(
        molecule
    )

    assert 1.0 <= score <= 10.0


def test_risk_is_deterministic():
    standardizer = (
        MoleculeStandardizer()
    )

    analyzer = (
        MoleculeRiskAnalyzer()
    )

    _, molecule = (
        standardizer.standardize(
            "CCOc1ccc2nc(S(N)(=O)=O)"
            "sc2c1"
        )
    )

    first = analyzer.synthesizability(
        molecule
    )

    second = analyzer.synthesizability(
        molecule
    )

    assert first == second
    assert (
        first.rare_fragment_count
        >= 0
    )
    assert (
        first.molecular_complexity
        >= 0.0
    )


def test_custom_thresholds_trigger_high_risk():
    standardizer = (
        MoleculeStandardizer()
    )

    analyzer = MoleculeRiskAnalyzer(
        SynthesizabilityConfig(
            high_complexity=1.0,
        )
    )

    _, molecule = (
        standardizer.standardize(
            "CCO"
        )
    )

    result = analyzer.synthesizability(
        molecule
    )

    assert result.risk_level == "high"
    assert any(
        "Bertz complexity"
        in factor
        for factor
        in result.risk_factors
    )


def test_context_can_raise_risk():
    standardizer = (
        MoleculeStandardizer()
    )

    analyzer = (
        MoleculeRiskAnalyzer()
    )

    _, molecule = (
        standardizer.standardize(
            "CCO"
        )
    )

    result = analyzer.synthesizability(
        molecule,
        historical_similarity=0.1,
        condition_uncertainty=0.8,
    )

    assert result.risk_level == "high"

    assert result.historical_similarity == (
        0.1
    )

    assert result.condition_uncertainty == (
        0.8
    )


@pytest.mark.parametrize(
    "keyword,value",
    [
        (
            "historical_similarity",
            -0.1,
        ),
        (
            "condition_uncertainty",
            1.1,
        ),
    ],
)
def test_context_range_validation(
    keyword,
    value,
):
    standardizer = (
        MoleculeStandardizer()
    )

    analyzer = (
        MoleculeRiskAnalyzer()
    )

    _, molecule = (
        standardizer.standardize(
            "CCO"
        )
    )

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        analyzer.synthesizability(
            molecule,
            **{
                keyword: value,
            },
        )