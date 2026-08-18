import httpx

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from mlflow import MlflowClient

from db.models import User
from schemas.inference import ModelInfo
from services.auth import ensure_user_is_admin_from_token
from _config import get_settings


router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    path="",
    summary="models",
    description="Get the available models in the registry.",
    response_model=list[ModelInfo]
)
def get_models(user: User = Depends(ensure_user_is_admin_from_token)) -> list[ModelInfo]:
    try:
        registered_models = MlflowClient(
            tracking_uri=get_settings().mlflow_server_uri
        ).search_registered_models()
    except Exception as e:
        message = f"Unable to retreive registered models. Error : {e}"
        raise HTTPException(status_code=500, detail=message)
    available_models: list[ModelInfo] = []
    for mdl in registered_models:
        name = mdl.name
        if mdl.latest_versions:
            for v in mdl.latest_versions:
                available_models.append(
                    ModelInfo(
                        name=name,
                        version=v.version,
                        published_at=datetime.fromtimestamp(v.creation_timestamp / 1000)
                    )
                )
        else:
            available_models.append(
                ModelInfo(
                    name=name,
                    version="",
                    published_at=datetime.fromtimestamp(0)
                )
            )
    return available_models


@router.get(
    path="/current",
    summary="current",
    description="Get the current model in production."
)
async def get_current_model(user: User = Depends(ensure_user_is_admin_from_token)) -> ModelInfo:
    try:
        async with httpx.AsyncClient() as client:
            result = await client.get(get_settings().inference_base_url + "/models/current")
            model_info = result.raise_for_status().json()
    except httpx.HTTPStatusError as e:
        detail = {
            "message": "An error occured in the inference service.",
            "error": dict(e.response.json())
        }
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.RequestError as e:
        detail = {
            "message": "An error occured while calling the inference service.",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=detail)
    except Exception as e:
        detail = {
            "message": "Unknown error occured while calling this endpoint.",
            "error": str(e)
        }
        raise HTTPException(status_code=500, detail=detail)
    return ModelInfo(**model_info)
