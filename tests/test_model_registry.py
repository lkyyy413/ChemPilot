from concurrent.futures import (
    ThreadPoolExecutor,
)
from pathlib import Path

import pytest

import chempilot.service.registry as registry_module

from chempilot.service.errors import (
    ConfigurationError,
)
from chempilot.service.registry import (
    ModelRegistry,
)


CONFIGURATION = """
service:
  name: ChemPilot
  version: 0.1.0
  default_reaction_protocol: reaction_center
  default_top_k: 5
  maximum_top_k: 20

molecule:
  druglike_scope:
    require_single_fragment: true
    require_carbon: true
    require_common_elements_only: true
    minimum_molecular_weight: 50.0
    maximum_molecular_weight: 1000.0
  synthesizability:
    rare_fragment_score_threshold: -2.5
    moderate_sa_score: 4.0
    high_sa_score: 6.0
    moderate_complexity: 500.0
    high_complexity: 1000.0
    moderate_rare_fragments: 1
    high_rare_fragments: 3
    moderate_history_similarity: 0.5
    low_history_similarity: 0.3
    moderate_condition_uncertainty: 0.35
    high_condition_uncertainty: 0.6

solubility:
  model_path: artifacts/solubility.joblib
  model_name: test_model
  model_protocol: test_protocol
  feature_dimension: 2058

reaction_conditions:
  model_root: artifacts/conditions
  target_root: data/targets
  feature_path: data/features.npz
  applicability_report_path: reports/applicability.json
  supported_protocols:
    - transformation
    - reaction_center

retrieval:
  checkpoint_directory: artifacts/rxnfp
  index_root: data/retrieval
  condition_path: data/conditions.parquet
  device: cpu
  supported_protocols:
    - transformation
    - reaction_center
"""


class FakeSolubilityPredictor:
    creation_count = 0

    def __init__(
        self,
        model_path,
        *,
        scope_evaluator,
    ):
        type(self).creation_count += 1
        self.model_path = Path(
            model_path
        )
        self.scope_evaluator = (
            scope_evaluator
        )


def write_configuration(
    tmp_path: Path,
    content: str = CONFIGURATION,
) -> Path:
    path = (
        tmp_path
        / "inference.yaml"
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    return path


def test_registry_starts_empty(
    tmp_path,
):
    path = write_configuration(
        tmp_path
    )

    registry = ModelRegistry(
        config_path=path,
        project_root=tmp_path,
    )

    assert (
        registry.loaded_components
        == ()
    )


def test_standardizer_is_cached(
    tmp_path,
):
    path = write_configuration(
        tmp_path
    )

    registry = ModelRegistry(
        config_path=path,
        project_root=tmp_path,
    )

    first = (
        registry.get_standardizer()
    )
    second = (
        registry.get_standardizer()
    )

    assert first is second
    assert (
        registry.loaded_components
        == (
            "molecule_standardizer",
        )
    )


def test_solubility_predictor_is_lazy_and_cached(
    tmp_path,
    monkeypatch,
):
    path = write_configuration(
        tmp_path
    )

    FakeSolubilityPredictor.creation_count = 0

    monkeypatch.setattr(
        registry_module,
        "SolubilityPredictor",
        FakeSolubilityPredictor,
    )

    registry = ModelRegistry(
        config_path=path,
        project_root=tmp_path,
    )

    assert (
        "solubility_predictor"
        not in registry.loaded_components
    )

    first = (
        registry.get_solubility_predictor()
    )
    second = (
        registry.get_solubility_predictor()
    )

    assert first is second
    assert (
        FakeSolubilityPredictor
        .creation_count
        == 1
    )

    assert first.model_path == (
        tmp_path
        / "artifacts"
        / "solubility.joblib"
    ).resolve()


def test_registry_cache_is_thread_safe(
    tmp_path,
    monkeypatch,
):
    path = write_configuration(
        tmp_path
    )

    FakeSolubilityPredictor.creation_count = 0

    monkeypatch.setattr(
        registry_module,
        "SolubilityPredictor",
        FakeSolubilityPredictor,
    )

    registry = ModelRegistry(
        config_path=path,
        project_root=tmp_path,
    )

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:
        components = list(
            executor.map(
                lambda _: (
                    registry
                    .get_solubility_predictor()
                ),
                range(16),
            )
        )

    assert all(
        component is components[0]
        for component in components
    )

    assert (
        FakeSolubilityPredictor
        .creation_count
        == 1
    )


def test_configured_thresholds_are_loaded(
    tmp_path,
):
    path = write_configuration(
        tmp_path
    )

    registry = ModelRegistry(
        config_path=path,
        project_root=tmp_path,
    )

    scope = (
        registry.get_scope_evaluator()
    )
    risk = (
        registry
        .get_molecule_risk_analyzer()
    )

    assert (
        scope.minimum_molecular_weight
        == 50.0
    )
    assert (
        scope.maximum_molecular_weight
        == 1000.0
    )
    assert (
        risk.config.high_sa_score
        == 6.0
    )
    assert (
        risk.config
        .high_condition_uncertainty
        == 0.6
    )


def test_missing_configuration_is_reported(
    tmp_path,
):
    with pytest.raises(
        ConfigurationError,
        match="missing",
    ):
        ModelRegistry(
            config_path=(
                tmp_path
                / "missing.yaml"
            ),
            project_root=tmp_path,
        )


def test_invalid_yaml_is_reported(
    tmp_path,
):
    path = write_configuration(
        tmp_path,
        content=(
            "service: [unterminated"
        ),
    )

    with pytest.raises(
        ConfigurationError,
        match="invalid YAML",
    ):
        ModelRegistry(
            config_path=path,
            project_root=tmp_path,
        )


def test_non_mapping_configuration_is_rejected(
    tmp_path,
):
    path = write_configuration(
        tmp_path,
        content=(
            "- first\n"
            "- second\n"
        ),
    )

    with pytest.raises(
        ConfigurationError,
        match="YAML mapping",
    ):
        ModelRegistry(
            config_path=path,
            project_root=tmp_path,
        )


def test_unsupported_condition_protocol_is_rejected(
    tmp_path,
):
    path = write_configuration(
        tmp_path
    )

    registry = ModelRegistry(
        config_path=path,
        project_root=tmp_path,
    )

    with pytest.raises(
        ConfigurationError,
        match="Unsupported protocol",
    ):
        registry.get_condition_predictor(
            "unknown_protocol"
        )


def test_unsupported_retrieval_protocol_is_rejected(
    tmp_path,
):
    path = write_configuration(
        tmp_path
    )

    registry = ModelRegistry(
        config_path=path,
        project_root=tmp_path,
    )

    with pytest.raises(
        ConfigurationError,
        match="Unsupported protocol",
    ):
        registry.get_retrieval_search(
            "unknown_protocol"
        )


def test_status_does_not_load_models(
    tmp_path,
):
    path = write_configuration(
        tmp_path
    )

    registry = ModelRegistry(
        config_path=path,
        project_root=tmp_path,
    )

    status = registry.status()

    assert (
        registry.loaded_components
        == ()
    )
    assert status[
        "loaded_components"
    ] == []

    assert (
        "solubility_model"
        in status["artifacts"]
    )
    assert not status[
        "artifacts"
    ][
        "solubility_model"
    ][
        "exists"
    ]