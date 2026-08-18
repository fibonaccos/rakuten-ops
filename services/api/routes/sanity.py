import httpx

from fastapi import APIRouter, Depends
from sqlalchemy import text

from db.models import User
from db.session import engine
from services.auth import ensure_user_is_admin_from_token
from _config import get_settings
from _version import __version__


router = APIRouter(prefix="", tags=["Sanity"])


@router.get(
    path="/health",
    summary="health",
    description="Check if the API can be called."
)
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "API is ready to use.",
        "api_version": __version__
    }


@router.get(
    path="/ready",
    summary="ready",
    description="Check if the backend services can be called from the API."
)
async def ready(user: User = Depends(ensure_user_is_admin_from_token)) -> dict[str, str]:
    try:
        async with engine.connect() as connection:
            _ = await connection.execute(text("SELECT 1"))
        database_readiness = "ready"
    except:
        database_readiness = "not ready"
    async with httpx.AsyncClient() as client:
        try:
            _ = await client.get(f"{get_settings().mlflow_server_uri}/health")
            mlflow_readiness = "ready"
        except:
            mlflow_readiness = "not_ready"
        try:
            _ = await client.get(f"{get_settings().inference_base_url}/health")
            inference_readiness = "ready"
        except:
            inference_readiness = "not_ready"
    return {
        "database": database_readiness,
        "model_registry": mlflow_readiness,
        "inference": inference_readiness
    }
