from uuid import UUID

from fastapi.testclient import (
    TestClient,
)

from chempilot.api.app import (
    create_app,
)
from chempilot.service.errors import (
    ConfigurationError,
    InvalidSmilesError,
    ModelArtifactError,
)
from chempilot.service.schemas import (
    UnifiedPredictionResponse,
)


class FakeRegistry:
    def __init__(
        self,
        *,
        missing=None,
    ):
        self.missing = (
            missing or []
        )

    def status(self):
        names = [
            "solubility_model",
            "condition_model_root",
            "reaction_feature_cache",
            "retrieval_checkpoint",
            "retrieval_index",
        ]

        artifacts = {
            name: {
                "path": (
                    f"/fake/{name}"
                ),
                "exists": (
                    name
                    not in self.missing
                ),
            }
            for name in names
        }

        return {
            "config_path": (
                "/fake/inference.yaml"
            ),
            "loaded_components": [],
            "artifacts": artifacts,
        }


class FakePredictionService:
    def __init__(
        self,
        registry,
        *,
        error=None,
    ):
        self.registry = registry
        self.error = error
        self.calls = []

    def predict(
        self,
        request,
        *,
        request_id=None,
    ):
        self.calls.append(
            {
                "request": request,
                "request_id": request_id,
            }
        )

        if self.error is not None:
            raise self.error

        return UnifiedPredictionResponse(
            request_id=request_id,
            status="success",
            molecule=None,
            reaction=None,
            warnings=[],
            model_versions={
                "test": "fake",
            },
        )


def make_client(
    *,
    missing=None,
    error=None,
    raise_server_exceptions=True,
):
    registry = FakeRegistry(
        missing=missing
    )
    service = (
        FakePredictionService(
            registry,
            error=error,
        )
    )

    application = create_app(
        registry,
        prediction_service=service,
    )

    client = TestClient(
        application,
        raise_server_exceptions=(
            raise_server_exceptions
        ),
    )

    return (
        client,
        registry,
        service,
    )


def test_health_is_healthy():
    client, _, service = (
        make_client()
    )

    response = client.get(
        "/health"
    )

    assert response.status_code == 200
    assert (
        response.json()["status"]
        == "healthy"
    )
    assert (
        response.json()[
            "missing_artifacts"
        ]
        == []
    )
    assert service.calls == []


def test_health_is_degraded_when_artifact_is_missing():
    client, _, _ = make_client(
        missing=[
            "solubility_model",
        ]
    )

    response = client.get(
        "/health"
    )

    assert response.status_code == 200
    assert (
        response.json()["status"]
        == "degraded"
    )
    assert (
        response.json()[
            "missing_artifacts"
        ]
        == [
            "solubility_model",
        ]
    )


def test_request_id_is_propagated():
    client, _, service = (
        make_client()
    )

    response = client.post(
        "/v1/predict",
        headers={
            "X-Request-ID": (
                "api-request-123"
            ),
        },
        json={
            "molecule_smiles": "CCO",
        },
    )

    assert response.status_code == 200
    assert (
        response.headers[
            "X-Request-ID"
        ]
        == "api-request-123"
    )
    assert (
        response.json()[
            "request_id"
        ]
        == "api-request-123"
    )
    assert (
        service.calls[0][
            "request_id"
        ]
        == "api-request-123"
    )


def test_request_id_is_generated():
    client, _, _ = make_client()

    response = client.post(
        "/v1/predict",
        json={
            "molecule_smiles": "CCO",
        },
    )

    identifier = (
        response.headers[
            "X-Request-ID"
        ]
    )

    UUID(identifier)

    assert (
        response.json()[
            "request_id"
        ]
        == identifier
    )


def test_reaction_request_is_parsed():
    client, _, service = (
        make_client()
    )

    response = client.post(
        "/v1/predict",
        json={
            "reactant_smiles": [
                "CCBr",
            ],
            "product_smiles": [
                "CCO",
            ],
            "reaction_protocol": (
                "transformation"
            ),
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    request = service.calls[0][
        "request"
    ]

    assert (
        request.reactant_smiles
        == ["CCBr"]
    )
    assert (
        request.product_smiles
        == ["CCO"]
    )
    assert (
        request.reaction_protocol
        == "transformation"
    )
    assert request.top_k == 3


def test_schema_validation_error_is_structured():
    client, _, service = (
        make_client()
    )

    response = client.post(
        "/v1/predict",
        json={
            "molecule_smiles": "",
            "unknown": True,
        },
    )

    body = response.json()

    assert response.status_code == 422
    assert body["status"] == "error"
    assert (
        body["error"]["code"]
        == "request_validation_error"
    )
    assert (
        "body.molecule_smiles"
        in body["error"][
            "context"
        ][
            "locations"
        ]
    )
    assert service.calls == []


def test_invalid_smiles_error_is_structured():
    error = InvalidSmilesError(
        "RDKit failed to parse SMILES.",
        field="molecule_smiles",
        context={
            "smiles": "invalid",
        },
    )

    client, _, _ = make_client(
        error=error
    )

    response = client.post(
        "/v1/predict",
        json={
            "molecule_smiles": (
                "invalid"
            ),
        },
    )

    body = response.json()

    assert response.status_code == 422
    assert (
        body["error"]["code"]
        == error.code
    )
    assert (
        body["error"]["field"]
        == "molecule_smiles"
    )
    assert (
        body["error"]["context"]
        == {
            "smiles": "invalid",
        }
    )


def test_missing_model_returns_503():
    error = ModelArtifactError(
        "Solubility model is missing.",
        field="solubility_model",
    )

    client, _, _ = make_client(
        error=error
    )

    response = client.post(
        "/v1/predict",
        json={
            "molecule_smiles": "CCO",
        },
    )

    body = response.json()

    assert response.status_code == 503
    assert (
        body["error"]["code"]
        == error.code
    )
    assert (
        body["error"]["message"]
        == "Solubility model is missing."
    )


def test_configuration_error_returns_500():
    error = ConfigurationError(
        "Invalid model configuration.",
        field="config",
    )

    client, _, _ = make_client(
        error=error
    )

    response = client.post(
        "/v1/predict",
        json={
            "molecule_smiles": "CCO",
        },
    )

    assert response.status_code == 500
    assert (
        response.json()[
            "error"
        ][
            "code"
        ]
        == error.code
    )


def test_unexpected_error_is_sanitized():
    client, _, _ = make_client(
        error=RuntimeError(
            "secret internal detail"
        ),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/v1/predict",
        json={
            "molecule_smiles": "CCO",
        },
    )

    body = response.json()

    assert response.status_code == 500
    assert (
        body["error"]["code"]
        == "internal_error"
    )
    assert (
        "secret internal detail"
        not in body["error"][
            "message"
        ]
    )
    assert (
        "traceback"
        not in str(body).lower()
    )


def test_inconsistent_injection_is_rejected():
    first_registry = FakeRegistry()
    second_registry = FakeRegistry()

    service = (
        FakePredictionService(
            second_registry
        )
    )

    try:
        create_app(
            first_registry,
            prediction_service=service,
        )
    except ValueError as error:
        assert (
            "same registry"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected inconsistent "
            "registry injection to fail."
        )