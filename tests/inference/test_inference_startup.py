"""
Resolving which model version to serve, at startup.

The model URI comes from `RAKUTEN__INFERENCE__MLFLOW_MODEL_NAME` and takes one
of two shapes: `name/version` pins a version, `name@alias` follows an alias such
as `champion`. Both have to reach the registry with the right arguments — this
runs before the service can answer a single request.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

# The registry reports milliseconds; the service stores a datetime.
CREATED_AT_MS = 1_756_000_000_000


class FakeRegistry:
    """Records how the registry was queried and answers with a fixed version."""

    def __init__(self) -> None:
        self.by_version: list[tuple[str, str]] = []
        self.by_alias: list[tuple[str, str]] = []

    def _version(self, name: str, version: str) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            version=version,
            run_id="a-run",
            creation_timestamp=CREATED_AT_MS,
        )

    def get_model_version(self, name: str, version: str) -> SimpleNamespace:
        self.by_version.append((name, version))
        return self._version(name, version)

    def get_model_version_by_alias(self, name: str, alias: str) -> SimpleNamespace:
        self.by_alias.append((name, alias))
        return self._version(name, "7")


@pytest.fixture
def startup(service, monkeypatch):
    """Call init_mlflow_states with the registry and the loader replaced."""
    import main
    from _config import Settings, get_settings

    registry = FakeRegistry()
    monkeypatch.setattr(main.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(main.mlflow.pyfunc, "load_model", lambda uri: f"model at {uri}")
    monkeypatch.setattr(main.mlflow, "MlflowClient", lambda *a, **k: registry)

    def run(model_name: str) -> tuple[FastAPI, FakeRegistry]:
        settings = Settings(
            port=8000,
            mlflow_host="mlflow-service",
            mlflow_port=5000,
            mlflow_model_name=model_name,
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        app = FastAPI()
        main.init_mlflow_states(app)
        return app, registry

    get_settings.cache_clear()
    return run


def test_a_pinned_version_is_fetched_by_version(startup) -> None:
    app, registry = startup("rakuten-classifier/3")

    assert registry.by_version == [("rakuten-classifier", "3")]
    assert registry.by_alias == []
    assert app.state.rakuten_model_info["version"] == "3"


def test_an_alias_is_fetched_by_alias_under_the_model_name(startup) -> None:
    """
    `name@alias` used to unpack `split("@")[-1]`, a string, into two variables.

    That raised before the service could start, so an alias-based URI could
    never be served.
    """
    app, registry = startup("rakuten-classifier@champion")

    assert registry.by_alias == [("rakuten-classifier", "champion")]
    assert registry.by_version == []
    assert app.state.rakuten_model_info["name"] == "rakuten-classifier"


def test_the_loaded_model_is_the_one_the_uri_names(startup) -> None:
    app, _ = startup("rakuten-classifier/3")

    assert app.state.rakuten_model == "model at models:/rakuten-classifier/3"


def test_the_publication_date_is_converted_from_milliseconds(startup) -> None:
    app, _ = startup("rakuten-classifier/3")

    assert app.state.rakuten_model_info["published_at"] == datetime.fromtimestamp(
        CREATED_AT_MS / 1000
    )
