from chempilot.api.app import (
    create_app,
)
from chempilot.service.registry import (
    ModelRegistry,
)


def test_openapi_contract():
    registry = ModelRegistry()
    application = create_app(
        registry
    )

    schema = application.openapi()

    assert (
        schema["info"]["title"]
        == "ChemPilot API"
    )
    assert (
        schema["info"]["version"]
        == "0.1.0"
    )

    paths = schema["paths"]

    assert "/health" in paths
    assert "/v1/predict" in paths
    assert (
        "get"
        in paths["/health"]
    )
    assert (
        "post"
        in paths["/v1/predict"]
    )

    prediction = paths[
        "/v1/predict"
    ][
        "post"
    ]

    request_schema = (
        prediction[
            "requestBody"
        ][
            "content"
        ][
            "application/json"
        ][
            "schema"
        ]
    )

    assert request_schema == {
        "$ref": (
            "#/components/schemas/"
            "PredictionRequest"
        )
    }

    responses = prediction[
        "responses"
    ]

    assert "200" in responses
    assert "422" in responses
    assert "500" in responses
    assert "503" in responses

    components = schema[
        "components"
    ][
        "schemas"
    ]

    required_models = {
        "PredictionRequest",
        "UnifiedPredictionResponse",
        "MoleculeAnalysis",
        "ReactionAnalysis",
        "SolubilityPrediction",
        "ConditionRecommendation",
        "RetrievalResult",
        "SynthesizabilityRisk",
        "ErrorResponse",
    }

    assert required_models.issubset(
        components
    )


def test_openapi_generation_does_not_load_models():
    registry = ModelRegistry()
    application = create_app(
        registry
    )

    application.openapi()

    assert (
        registry.loaded_components
        == ()
    )