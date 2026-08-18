from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from mlflow import MlflowClient
from typing import Any

from services.app_state import _get_model_info
from _config import get_settings


router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/current")
def get_current_model(model_info: dict[str, Any] = Depends(_get_model_info)) -> dict[str, Any]:
    return model_info
