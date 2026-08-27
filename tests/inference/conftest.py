"""
Fixtures for the inference service.

The service loads a model from the MLflow registry on startup. Tests replace that
startup with a stub model placed directly in the application state, which is exactly
what the routes read through their dependencies.
"""

import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.conftest import use_service

os.environ.update(
    {
        "RAKUTEN__INFERENCE__PORT": "8000",
        "RAKUTEN__INFERENCE__MLFLOW_HOST": "mlflow-service",
        "RAKUTEN__INFERENCE__MLFLOW_PORT": "5000",
        "RAKUTEN__INFERENCE__MLFLOW_MODEL_NAME": "rakuten-classifier/3",
    }
)

@pytest.fixture(autouse=True)
def service() -> Path:
    """Put `services/inference` on top of `sys.path` for the duration of one test."""
    return use_service("inference")


MODEL_INFO: dict[str, Any] = {
    "name": "rakuten-classifier",
    "version": "3",
    "published_at": datetime(2026, 8, 20, tzinfo=UTC),
}


class StubModel:
    """
    Stands in for the MLflow pyfunc model.

    Returns the contract the routes unpack: four values for a single input, three
    for a batch. Records the array it was called with so tests can assert on shape.
    """

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def predict(self, x: np.ndarray) -> tuple:
        self.calls.append(x)
        distribution = {"2583": 0.87, "1140": 0.13}
        if x.shape[0] == 1:
            return ("2583", 0.87, distribution, 12.5)
        outputs = [("2583", 0.87, distribution) for _ in range(x.shape[0])]
        return (outputs, 25.0, 25.0 / x.shape[0])


@pytest.fixture
def model() -> StubModel:
    return StubModel()


@pytest.fixture
def client(service: Path, model: StubModel) -> Iterator[Any]:
    from fastapi.testclient import TestClient
    from main import build_api

    app = build_api()

    @asynccontextmanager
    async def stub_startup(running_app: Any):
        running_app.state.rakuten_model = model
        running_app.state.rakuten_model_info = MODEL_INFO
        yield

    # build_api() installs a lifespan that pulls the model from the MLflow registry.
    app.router.lifespan_context = stub_startup

    with TestClient(app) as test_client:
        yield test_client
