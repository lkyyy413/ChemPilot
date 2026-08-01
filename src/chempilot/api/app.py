"""FastAPI application for unified ChemPilot inference."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import (
    FastAPI,
    Request,
)
from fastapi.encoders import (
    jsonable_encoder,
)
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.responses import (
    JSONResponse,
)

from chempilot.service.errors import (
    ChemPilotServiceError,
)
from chempilot.service.prediction import (
    PredictionService,
)
from chempilot.service.registry import (
    ModelRegistry,
)
from chempilot.service.schemas import (
    ErrorDetail,
    ErrorResponse,
    PredictionRequest,
    UnifiedPredictionResponse,
)


LOGGER = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def _request_id(
    request: Request,
) -> str | None:
    return getattr(
        request.state,
        "request_id",
        None,
    )


def _error_response(
    *,
    request_id: str | None,
    code: str,
    message: str,
    field: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(
            code=code,
            message=message,
            field=field,
            context=context or {},
        ),
    )

    return jsonable_encoder(
        response
    )

def create_app(
    registry: ModelRegistry | None = None,
    *,
    prediction_service: (
        PredictionService | None
    ) = None,
) -> FastAPI:
    """Create the API with optionally injected components."""

    # 确定本次应用使用的注册器。
    if registry is None:
        if prediction_service is not None:
            model_registry = (
                prediction_service.registry
            )
        else:
            model_registry = (
                ModelRegistry()
            )
    else:
        model_registry = registry

    # 确定本次应用使用的统一预测服务。
    if prediction_service is None:
        active_prediction_service = (
            PredictionService(
                model_registry
            )
        )
    else:
        active_prediction_service = (
            prediction_service
        )

        # 防止注册器和预测服务使用两套不同模型配置。
        if (
            active_prediction_service.registry
            is not model_registry
        ):
            raise ValueError(
                "Injected prediction service and "
                "registry must reference the same "
                "registry object."
            )

    application = FastAPI(
        title="ChemPilot API",
        version="0.1.0",
        description=(
            "Unified molecular-property, "
            "reaction-condition, similarity, "
            "and lightweight synthesizability "
            "risk inference."
        ),
    )

    application.state.registry = (
        model_registry
    )
    application.state.prediction_service = (
        active_prediction_service
    )

    @application.middleware(
        "http"
    )
    async def add_request_id(
        request: Request,
        call_next,
    ):
        identifier = (
            request.headers.get(
                REQUEST_ID_HEADER
            )
            or str(uuid4())
        )

        request.state.request_id = (
            identifier
        )

        response = await call_next(
            request
        )

        response.headers[
            REQUEST_ID_HEADER
        ] = identifier

        return response

    @application.exception_handler(
        ChemPilotServiceError
    )
    async def service_error_handler(
        request: Request,
        error: ChemPilotServiceError,
    ):
        LOGGER.warning(
            "Request %s failed with %s: %s",
            _request_id(request),
            error.code,
            error.message,
        )

        return JSONResponse(
            status_code=(
                error.status_code
            ),
            content=_error_response(
                request_id=(
                    _request_id(request)
                ),
                code=error.code,
                message=error.message,
                field=error.field,
                context=error.context,
            ),
        )

    @application.exception_handler(
        RequestValidationError
    )
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ):
        details = error.errors()

        locations = [
            ".".join(
                str(item)
                for item in detail.get(
                    "loc",
                    (),
                )
            )
            for detail in details
        ]

        messages = [
            str(
                detail.get(
                    "msg",
                    "Invalid request value.",
                )
            )
            for detail in details
        ]

        error_types = [
            str(
                detail.get(
                    "type",
                    "validation_error",
                )
            )
            for detail in details
        ]

        LOGGER.info(
            "Request %s failed schema validation",
            _request_id(request),
        )

        return JSONResponse(
            status_code=422,
            content=_error_response(
                request_id=(
                    _request_id(request)
                ),
                code=(
                    "request_validation_error"
                ),
                message=(
                    "The request body failed "
                    "schema validation."
                ),
                context={
                    "locations": locations,
                    "messages": messages,
                    "types": error_types,
                },
            ),
        )

    @application.exception_handler(
        Exception
    )
    async def unexpected_error_handler(
        request: Request,
        error: Exception,
    ):
        LOGGER.exception(
            "Unexpected error for request %s",
            _request_id(request),
        )

        return JSONResponse(
            status_code=500,
            content=_error_response(
                request_id=(
                    _request_id(request)
                ),
                code="internal_error",
                message=(
                    "An unexpected internal "
                    "error occurred."
                ),
            ),
        )

    @application.get(
        "/health",
        tags=["system"],
    )
    def health() -> dict[str, Any]:
        status = (
            model_registry.status()
        )

        artifacts = status[
            "artifacts"
        ]

        missing = [
            name
            for name, artifact
            in artifacts.items()
            if not artifact["exists"]
        ]

        health_status = (
            "healthy"
            if not missing
            else "degraded"
        )

        return {
            "status": health_status,
            "service": "ChemPilot",
            "version": "0.1.0",
            "loaded_components": (
                status[
                    "loaded_components"
                ]
            ),
            "artifacts": artifacts,
            "missing_artifacts": (
                missing
            ),
            "note": (
                "Health checks inspect artifact "
                "availability without loading models."
            ),
        }

    @application.post(
        "/v1/predict",
        response_model=(
            UnifiedPredictionResponse
        ),
        responses={
            422: {
                "model": ErrorResponse,
            },
            500: {
                "model": ErrorResponse,
            },
            503: {
                "model": ErrorResponse,
            },
        },
        tags=["prediction"],
    )
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> UnifiedPredictionResponse:
        return (
            active_prediction_service.predict(
                payload,
                request_id=(
                    _request_id(request)
                ),
            )
        )

    return application


app = create_app()