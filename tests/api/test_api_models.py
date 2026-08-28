"""
Model routes: who may call them, and what they answer.

`/models` reads the registry directly and stays reserved to admins. `/models/current`
asks the inference service which model it is actually serving, and is open to any
authenticated user — a seller needs to know whether a model is available before
filling in a listing.

Both routes call something outside the API, so both are driven here against a
stub. Without one, an authorised call fails on the network and answers 500,
which says nothing about the route being tested.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

CURRENT_MODEL = {
    "name": "rakuten-naive",
    "version": "3",
    "published_at": datetime(2026, 8, 20, tzinfo=timezone.utc).isoformat(),
}


class FakeResponse:
    """The answer of the inference service, as the route reads it."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> "FakeResponse":
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("erreur", request=None, response=self)  # type: ignore[arg-type]
        return self

    def json(self) -> Any:
        return self._payload


@pytest.fixture
def inference(monkeypatch):
    """
    Replace the HTTP client the route opens towards the inference service.

    Returns a setter: call it with a response, or with an exception to raise.
    """
    outcome: dict[str, Any] = {"response": FakeResponse(CURRENT_MODEL)}

    class FakeAsyncClient:
        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def get(self, url: str) -> FakeResponse:
            if isinstance(outcome["response"], BaseException):
                raise outcome["response"]
            return outcome["response"]

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: FakeAsyncClient())

    def answers(response: Any) -> None:
        outcome["response"] = response

    return answers


@pytest.fixture
def registry(monkeypatch):
    """Replace the MLflow client the /models route builds."""
    import routes.models as models_route

    class FakeRegistry:
        def search_registered_models(self) -> list[Any]:
            return [
                SimpleNamespace(
                    name="rakuten-naive",
                    latest_versions=[SimpleNamespace(version="3", creation_timestamp=1_756_000_000_000)],
                )
            ]

    monkeypatch.setattr(models_route, "MlflowClient", lambda *a, **k: FakeRegistry())


# ── /models — reserved to admins ──────────────────────────────────────────────


def test_listing_the_registry_requires_a_token(client) -> None:
    assert client.get("/models").status_code == 401


def test_listing_the_registry_is_refused_to_a_plain_user(as_user) -> None:
    assert as_user.get("/models").status_code == 401


def test_an_admin_lists_the_registered_models(as_admin, registry) -> None:
    response = as_admin.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["name"] == "rakuten-naive"
    assert body[0]["version"] == "3"


# ── /models/current — open to any authenticated user ──────────────────────────


def test_the_current_model_requires_a_token(client) -> None:
    assert client.get("/models/current").status_code == 401


def test_a_plain_user_reads_the_current_model(as_user, inference) -> None:
    """
    Open to every authenticated user since the route stopped requiring admin.

    A seller needs to know whether a model is serving before writing a listing.
    """
    response = as_user.get("/models/current")

    assert response.status_code == 200
    assert response.json()["name"] == "rakuten-naive"
    assert response.json()["version"] == "3"


def test_an_admin_also_reads_the_current_model(as_admin, inference) -> None:
    assert as_admin.get("/models/current").status_code == 200


def test_an_unreachable_inference_service_answers_500(as_user, inference) -> None:
    """
    The gateway cannot answer what the inference service alone knows.

    This is the state the API is in before the inference service has loaded its
    model, and what a caller sees while the stack is still warming up.
    """
    inference(httpx.ConnectError("connection refused"))

    response = as_user.get("/models/current")

    assert response.status_code == 500
    assert "inference service" in response.json()["detail"]["message"]


def test_an_error_from_the_inference_service_keeps_its_status(as_user, inference) -> None:
    inference(FakeResponse({"detail": "no model loaded"}, status_code=503))

    assert as_user.get("/models/current").status_code == 503
