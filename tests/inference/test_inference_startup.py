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


# ── Which model the service asks for ──────────────────────────────────────────


def _registered_name() -> str:
    """The name the training pipeline registers under, from params.yaml."""
    import yaml

    from tests.conftest import ROOT

    params = yaml.safe_load((ROOT / "training" / "params.yaml").read_text(encoding="utf-8"))
    return params["champion_challenger"]["registered_model_name"]


def test_the_default_model_is_the_one_training_registers(service) -> None:
    """
    The two ends of the pipeline must name the same model.

    training/params.yaml decides what the champion is called; the inference
    service decides what to load. Renaming one and not the other leaves a
    service asking for a model nobody publishes, and nothing says so until
    startup fails.
    """
    from _config import DEFAULT_MODEL

    assert DEFAULT_MODEL.split("@")[0].split("/")[0] == _registered_name()


def test_the_default_follows_the_champion_alias(service) -> None:
    """Following the alias means a promotion needs no configuration change."""
    from _config import DEFAULT_MODEL

    assert DEFAULT_MODEL.endswith("@champion")


def test_a_service_with_no_model_configured_serves_the_champion(service, monkeypatch) -> None:
    """
    A fresh clone starts without anyone filling in a .env.

    `_env_file=None` ignores any .env sitting in the working directory, so the
    result does not depend on whose machine runs the suite.
    """
    from _config import DEFAULT_MODEL, Settings

    monkeypatch.delenv("RAKUTEN__INFERENCE__MLFLOW_MODEL_NAME", raising=False)
    settings = Settings(_env_file=None)

    assert settings.mlflow_model_uri == f"models:/{DEFAULT_MODEL}"


def test_an_empty_entry_is_treated_as_absent(service) -> None:
    """.env.example ships every key empty, so copying it must not break startup."""
    from _config import DEFAULT_MODEL, Settings

    assert Settings(mlflow_model_name="").mlflow_model_uri == f"models:/{DEFAULT_MODEL}"
    assert Settings(mlflow_model_name="   ").mlflow_model_uri == f"models:/{DEFAULT_MODEL}"


def test_an_explicit_model_still_wins(service) -> None:
    from _config import Settings

    settings = Settings(mlflow_model_name="rakuten-naive/7")

    assert settings.mlflow_model_uri == "models:/rakuten-naive/7"
