from datetime import datetime
import mlflow

from contextlib import asynccontextmanager
from fastapi import FastAPI

from routes.health import router as health_router
from routes.metrics import router as metrics_router
from routes.models import router as models_router
from routes.predict import router as predict_router
from _config import get_settings
from _version import __version__


def init_mlflow_states(app: FastAPI) -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_server_uri)
    app.state.rakuten_model = mlflow.pyfunc.load_model(settings.mlflow_model_uri)
    client = mlflow.MlflowClient()
    model_id = settings.mlflow_model_uri.removeprefix("models:/")
    if "@" in model_id:
        model_name, alias = model_id.split("@")[-1]
        mv = client.get_model_version_by_alias(settings.mlflow_model_name, alias=alias)
    else:
        model_name, version = model_id.split("/", 2)
        mv = client.get_model_version(model_name, version=version)
    rid = mv.run_id if mv.run_id else ""
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
