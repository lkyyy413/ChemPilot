"""Domain-specific exceptions for inference services."""

from __future__ import annotations

from typing import Any


class ChemPilotServiceError(Exception):
    """Base exception carrying API-safe error metadata."""

    code = "SERVICE_ERROR"
    status_code = 500

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        context: dict[
            str,
            Any,
        ] | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.field = field
        self.context = (
            {}
            if context is None
            else dict(context)
        )


class InvalidSmilesError(
    ChemPilotServiceError
):
    code = "INVALID_SMILES"
    status_code = 422


class InvalidReactionError(
    ChemPilotServiceError
):
    code = "INVALID_REACTION"
    status_code = 422


class ModelArtifactError(
    ChemPilotServiceError
):
    code = "MODEL_ARTIFACT_ERROR"
    status_code = 503


class ConfigurationError(
    ChemPilotServiceError
):
    code = "CONFIGURATION_ERROR"
    status_code = 500