"""Unified ChemPilot inference service."""

from .schemas import (
    ErrorDetail,
    ErrorResponse,
    PredictionRequest,
    UnifiedPredictionResponse,
)

from .base import (
    BaseFeaturizer,
    BasePredictor,
)

from .solubility import (
    COMMON_DRUGLIKE_ATOMIC_NUMBERS,
    DruglikeScopeEvaluator,
    DruglikeScopeResult,
    SolubilityFeaturizer,
    SolubilityPredictor,
)

from .registry import (
    DEFAULT_INFERENCE_CONFIG_PATH,
    ModelRegistry,
)

from .prediction import (
    PredictionService,
)

__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "PredictionRequest",
    "UnifiedPredictionResponse",
    "BaseFeaturizer",
    "BasePredictor",
    "COMMON_DRUGLIKE_ATOMIC_NUMBERS",
    "DruglikeScopeEvaluator",
    "DruglikeScopeResult",
    "SolubilityFeaturizer",
    "SolubilityPredictor",
    "DEFAULT_INFERENCE_CONFIG_PATH",
    "ModelRegistry",
    "PredictionService",
]