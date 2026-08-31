from datetime import datetime
import mlflow

from contextlib import asynccontextmanager
from fastapi import FastAPI
from mlflow.exceptions import MlflowException

from routes.health import router as health_router
from routes.metrics import router as metrics_router
from routes.models import router as models_router
from routes.predict import router as predict_router
from _config import get_settings
from _version import __version__


def _explain_missing_model(client, wanted: str, server: str) -> str:
    """
    Say what went wrong when the configured model cannot be loaded.

    Without this the service dies on a raw MLflow exception, and the reader has
    no way to tell a renamed model from an unreachable registry. Both happen:
    the name lives in training/params.yaml and in the environment, and the two
    drift apart on a rename.
    """
    try:
        available = sorted(model.name for model in client.search_registered_models())
    except MlflowException:
        return (
            f"Cannot load model {wanted!r}: the registry at {server} did not answer. "
            f"Check that the mlflow service is up and RAKUTEN__INFERENCE__MLFLOW_HOST "
            f"and _PORT point at it."
        )

    listed = ", ".join(available) if available else "none"
    return (
        f"Model {wanted!r} is not in the registry at {server}. Registered models: {listed}. "
        f"Set RAKUTEN__INFERENCE__MLFLOW_MODEL_NAME to one of them, as <name>@<alias> "
        f"or <name>/<version>, or run the training pipeline to publish one."
    )


def init_mlflow_states(app: FastAPI) -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_server_uri)
    client = mlflow.MlflowClient()
    try:
        app.state.rakuten_model = mlflow.pyfunc.load_model(settings.mlflow_model_uri)
    except MlflowException as error:
        raise RuntimeError(
            _explain_missing_model(client, settings.mlflow_model_name, settings.mlflow_server_uri)
        ) from error
    model_id = settings.mlflow_model_uri.removeprefix("models:/")
    if "@" in model_id:
        model_name, alias = model_id.split("@", 1)
        mv = client.get_model_version_by_alias(model_name, alias=alias)
    else:
        model_name, version = model_id.split("/", 1)
        mv = client.get_model_version(model_name, version=version)
    app.state.rakuten_model_info = {
        "name": mv.name,
        "version": mv.version,
        "published_at": datetime.fromtimestamp(mv.creation_timestamp / 1000)
    }
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_mlflow_states(app)
        yield
    finally:
        ...


def build_api() -> FastAPI:
    app: FastAPI = FastAPI(
        title="Rakuten Inference Service",
        version=__version__,
        lifespan=lifespan
    )

    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(models_router)
    app.include_router(predict_router)

    return app


app = build_api()
