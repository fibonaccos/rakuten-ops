from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text

from db.session import engine
from middlewares.prometheus import prometheus_middlware
from routes.auth import router as auth_router
from routes.metrics import router as metrics_router
from routes.models import router as models_router
from routes.predict import router as predict_router
from routes.sanity import router as health_router
from _version import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        yield
    finally:
        await engine.dispose()


def build_api() -> FastAPI:
    app: FastAPI = FastAPI(
        title="Rakuten Product Classification API",
        version=__version__,
        lifespan=lifespan
    )

    app.middleware("http")(prometheus_middlware)

    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(models_router)
    app.include_router(predict_router)

    return app


app = build_api()
