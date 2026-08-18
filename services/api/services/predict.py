import httpx

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

from db.models import Inference
from services.db import InferenceRepository
from _config import get_settings


_INFERENCE_RUNTIME_BASE_URL: str = get_settings().inference_base_url
_INFERENCE_PREDICT_SINGLE_ENDPOINT: str = _INFERENCE_RUNTIME_BASE_URL + "/predict/single"
_INFERENCE_PREDICT_BATCH_ENDPOINT: str = _INFERENCE_RUNTIME_BASE_URL + "/predict/batch"


async def request_predict_single(inputs: dict[str, str | None]) -> Any:
    try:
        async with httpx.AsyncClient() as client:
            response = (await client.post(
                url=_INFERENCE_PREDICT_SINGLE_ENDPOINT,
                json=inputs
            )).raise_for_status()
        result = response.json()
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
    return result


async def request_predict_batch(inputs: list[dict[str, str | None]]):
    try:
        async with httpx.AsyncClient() as client:
            response = (await client.post(
                url=_INFERENCE_PREDICT_BATCH_ENDPOINT,
                json={f"{i}": x for i, x in enumerate(inputs)}
            )).raise_for_status()
        result = response.json()
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
    return result


async def feed_inference_db(
    inference: Inference | list[Inference],
    session: AsyncSession
) -> Inference | list[Inference]:
    repository = InferenceRepository(session)
    if isinstance(inference, list):
        created = await repository.create_batch(inference)
    else:
        created = await repository.create(inference)
    return created
