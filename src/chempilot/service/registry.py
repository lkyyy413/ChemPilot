"""Configuration-driven lazy model registry."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from chempilot.reactions.inference import (
    ReactionConditionPredictor,
)
from chempilot.reactions.similarity import (
    SimilarReactionSearch,
)
from chempilot.service.errors import (
    ConfigurationError,
    ModelArtifactError,
)
from chempilot.service.molecule import (
    MoleculeRiskAnalyzer,
    MoleculeStandardizer,
    SynthesizabilityConfig,
)
from chempilot.service.solubility import (
    DruglikeScopeEvaluator,
    SolubilityPredictor,
)


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

DEFAULT_INFERENCE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "inference.yaml"
)


class ModelRegistry:
    """Lazily construct and cache inference components."""

    def __init__(
        self,
        config_path: str | Path = (
            DEFAULT_INFERENCE_CONFIG_PATH
        ),
        *,
        project_root: str | Path = (
            PROJECT_ROOT
        ),
    ) -> None:
        self.config_path = Path(
            config_path
        )
        self.project_root = Path(
            project_root
        ).resolve()

        self._config = (
            self._load_configuration()
        )
        self._components: dict[
            str,
            Any,
        ] = {}
        self._lock = RLock()

    @property
    def config(self) -> dict[str, Any]:
        """Return the parsed configuration."""
        return self._config

    @property
    def loaded_components(
        self,
    ) -> tuple[str, ...]:
        """Return names of currently loaded components."""
        with self._lock:
            return tuple(
                sorted(
                    self._components
                )
            )

    def _load_configuration(
        self,
    ) -> dict[str, Any]:
        if not self.config_path.is_file():
            raise ConfigurationError(
                "Inference configuration file is missing.",
                field="config_path",
                context={
                    "path": str(
                        self.config_path
                    ),
                },
            )

        try:
            with self.config_path.open(
                encoding="utf-8",
            ) as file:
                configuration = (
                    yaml.safe_load(file)
                )
        except yaml.YAMLError as error:
            raise ConfigurationError(
                "Inference configuration contains invalid YAML.",
                field="config_path",
                context={
                    "path": str(
                        self.config_path
                    ),
                    "error_type": (
                        type(error).__name__
                    ),
                },
            ) from error
        except OSError as error:
            raise ConfigurationError(
                "Inference configuration could not be read.",
                field="config_path",
                context={
                    "path": str(
                        self.config_path
                    ),
                    "error_type": (
                        type(error).__name__
                    ),
                },
            ) from error

        if not isinstance(
            configuration,
            dict,
        ):
            raise ConfigurationError(
                "Inference configuration must be a YAML mapping.",
                field="config_path",
                context={
                    "path": str(
                        self.config_path
                    ),
                },
            )

        return configuration

    def _section(
        self,
        name: str,
    ) -> dict[str, Any]:
        section = self._config.get(
            name
        )

        if not isinstance(
            section,
            dict,
        ):
            raise ConfigurationError(
                f"Missing or invalid configuration section: {name}.",
                field=name,
                context={
                    "config_path": str(
                        self.config_path
                    ),
                },
            )

        return section

    def _resolve_path(
        self,
        value: Any,
        *,
        field: str,
    ) -> Path:
        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ConfigurationError(
                "Configured artifact path must be a non-empty string.",
                field=field,
                context={
                    "value": value,
                },
            )

        path = Path(value)

        if not path.is_absolute():
            path = (
                self.project_root
                / path
            )

        return path.resolve()

    def _cached(
        self,
        name: str,
        factory,
    ):
        with self._lock:
            if name not in self._components:
                LOGGER.info(
                    "Loading inference component: %s",
                    name,
                )

                self._components[name] = (
                    factory()
                )

            return self._components[
                name
            ]

    def get_standardizer(
        self,
    ) -> MoleculeStandardizer:
        return self._cached(
            "molecule_standardizer",
            MoleculeStandardizer,
        )

    def get_scope_evaluator(
        self,
    ) -> DruglikeScopeEvaluator:
        def factory():
            molecule_config = (
                self._section(
                    "molecule"
                )
            )
            scope = molecule_config.get(
                "druglike_scope"
            )

            if not isinstance(
                scope,
                dict,
            ):
                raise ConfigurationError(
                    "Missing molecule.druglike_scope configuration.",
                    field=(
                        "molecule.druglike_scope"
                    ),
                )

            try:
                return DruglikeScopeEvaluator(
                    minimum_molecular_weight=float(
                        scope[
                            "minimum_molecular_weight"
                        ]
                    ),
                    maximum_molecular_weight=float(
                        scope[
                            "maximum_molecular_weight"
                        ]
                    ),
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise ConfigurationError(
                    "Invalid drug-like scope configuration.",
                    field=(
                        "molecule.druglike_scope"
                    ),
                    context={
                        "error_type": (
                            type(error).__name__
                        ),
                    },
                ) from error

        return self._cached(
            "druglike_scope_evaluator",
            factory,
        )

    def get_molecule_risk_analyzer(
        self,
    ) -> MoleculeRiskAnalyzer:
        def factory():
            molecule_config = (
                self._section(
                    "molecule"
                )
            )
            risk_config = (
                molecule_config.get(
                    "synthesizability"
                )
            )

            if not isinstance(
                risk_config,
                dict,
            ):
                raise ConfigurationError(
                    "Missing synthesizability configuration.",
                    field=(
                        "molecule.synthesizability"
                    ),
                )

            try:
                configuration = (
                    SynthesizabilityConfig(
                        **risk_config
                    )
                )
            except (
                TypeError,
                ValueError,
            ) as error:
                raise ConfigurationError(
                    "Invalid synthesizability configuration.",
                    field=(
                        "molecule.synthesizability"
                    ),
                    context={
                        "error_type": (
                            type(error).__name__
                        ),
                    },
                ) from error

            return MoleculeRiskAnalyzer(
                configuration
            )

        return self._cached(
            "molecule_risk_analyzer",
            factory,
        )

    def get_solubility_predictor(
        self,
    ) -> SolubilityPredictor:
        def factory():
            configuration = (
                self._section(
                    "solubility"
                )
            )

            model_path = self._resolve_path(
                configuration.get(
                    "model_path"
                ),
                field=(
                    "solubility.model_path"
                ),
            )

            return SolubilityPredictor(
                model_path=model_path,
                scope_evaluator=(
                    self.get_scope_evaluator()
                ),
            )

        return self._cached(
            "solubility_predictor",
            factory,
        )

    def _validate_protocol(
        self,
        protocol: str,
        *,
        section_name: str,
    ) -> None:
        configuration = self._section(
            section_name
        )
        supported = configuration.get(
            "supported_protocols"
        )

        if (
            not isinstance(
                supported,
                list,
            )
            or protocol not in supported
        ):
            raise ConfigurationError(
                f"Unsupported protocol: {protocol}.",
                field="protocol",
                context={
                    "section": section_name,
                    "supported_protocols": (
                        supported
                    ),
                },
            )

    def get_condition_predictor(
        self,
        protocol: str,
    ) -> ReactionConditionPredictor:
        self._validate_protocol(
            protocol,
            section_name=(
                "reaction_conditions"
            ),
        )

        component_name = (
            f"condition_predictor:{protocol}"
        )

        def factory():
            configuration = (
                self._section(
                    "reaction_conditions"
                )
            )

            try:
                return (
                    ReactionConditionPredictor(
                        protocol=protocol,
                        model_root=(
                            self._resolve_path(
                                configuration.get(
                                    "model_root"
                                ),
                                field=(
                                    "reaction_conditions."
                                    "model_root"
                                ),
                            )
                        ),
                        target_root=(
                            self._resolve_path(
                                configuration.get(
                                    "target_root"
                                ),
                                field=(
                                    "reaction_conditions."
                                    "target_root"
                                ),
                            )
                        ),
                        feature_path=(
                            self._resolve_path(
                                configuration.get(
                                    "feature_path"
                                ),
                                field=(
                                    "reaction_conditions."
                                    "feature_path"
                                ),
                            )
                        ),
                        ad_report_path=(
                            self._resolve_path(
                                configuration.get(
                                    "applicability_report_path"
                                ),
                                field=(
                                    "reaction_conditions."
                                    "applicability_report_path"
                                ),
                            )
                        ),
                    )
                )
            except (
                ConfigurationError,
                ModelArtifactError,
            ):
                raise
            except Exception as error:
                raise ModelArtifactError(
                    "Reaction-condition artifacts could not be loaded.",
                    field=(
                        "reaction_condition_model"
                    ),
                    context={
                        "protocol": protocol,
                        "error_type": (
                            type(error).__name__
                        ),
                    },
                ) from error

        return self._cached(
            component_name,
            factory,
        )

    def get_retrieval_search(
        self,
        protocol: str,
    ) -> SimilarReactionSearch:
        self._validate_protocol(
            protocol,
            section_name="retrieval",
        )

        component_name = (
            f"reaction_retrieval:{protocol}"
        )

        def factory():
            configuration = (
                self._section(
                    "retrieval"
                )
            )

            device = configuration.get(
                "device"
            )

            if (
                device is not None
                and not isinstance(
                    device,
                    str,
                )
            ):
                raise ConfigurationError(
                    "retrieval.device must be null or a string.",
                    field=(
                        "retrieval.device"
                    ),
                )

            try:
                return SimilarReactionSearch(
                    protocol=protocol,
                    device=device,
                    checkpoint_directory=(
                        self._resolve_path(
                            configuration.get(
                                "checkpoint_directory"
                            ),
                            field=(
                                "retrieval."
                                "checkpoint_directory"
                            ),
                        )
                    ),
                    index_root=(
                        self._resolve_path(
                            configuration.get(
                                "index_root"
                            ),
                            field=(
                                "retrieval.index_root"
                            ),
                        )
                    ),
                    condition_path=(
                        self._resolve_path(
                            configuration.get(
                                "condition_path"
                            ),
                            field=(
                                "retrieval."
                                "condition_path"
                            ),
                        )
                    ),
                )
            except (
                ConfigurationError,
                ModelArtifactError,
            ):
                raise
            except Exception as error:
                raise ModelArtifactError(
                    "Reaction-retrieval artifacts could not be loaded.",
                    field=(
                        "reaction_retrieval"
                    ),
                    context={
                        "protocol": protocol,
                        "error_type": (
                            type(error).__name__
                        ),
                    },
                ) from error

        return self._cached(
            component_name,
            factory,
        )

    def status(
        self,
    ) -> dict[str, Any]:
        """Return configuration and lazy-loading status."""
        solubility = self._section(
            "solubility"
        )
        conditions = self._section(
            "reaction_conditions"
        )
        retrieval = self._section(
            "retrieval"
        )

        artifact_paths = {
            "solubility_model": (
                self._resolve_path(
                    solubility.get(
                        "model_path"
                    ),
                    field=(
                        "solubility.model_path"
                    ),
                )
            ),
            "condition_model_root": (
                self._resolve_path(
                    conditions.get(
                        "model_root"
                    ),
                    field=(
                        "reaction_conditions."
                        "model_root"
                    ),
                )
            ),
            "reaction_feature_cache": (
                self._resolve_path(
                    conditions.get(
                        "feature_path"
                    ),
                    field=(
                        "reaction_conditions."
                        "feature_path"
                    ),
                )
            ),
            "retrieval_checkpoint": (
                self._resolve_path(
                    retrieval.get(
                        "checkpoint_directory"
                    ),
                    field=(
                        "retrieval."
                        "checkpoint_directory"
                    ),
                )
            ),
            "retrieval_index": (
                self._resolve_path(
                    retrieval.get(
                        "index_root"
                    ),
                    field=(
                        "retrieval.index_root"
                    ),
                )
            ),
        }

        return {
            "config_path": str(
                self.config_path
            ),
            "loaded_components": list(
                self.loaded_components
            ),
            "artifacts": {
                name: {
                    "path": str(path),
                    "exists": path.exists(),
                }
                for name, path
                in artifact_paths.items()
            },
        }