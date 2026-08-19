import numpy as np

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from mlflow.pyfunc import PyFuncModel
from time import perf_counter
from typing import Any

from prom_metrics import INFERENCE_DURATION, INFERENCE_REQUESTS
from services.app_state import _get_model, _get_model_info


router = APIRouter(prefix="/predict")


@router.post("/single", summary="predict single")
async def predict_single(
    inputs: dict[str, str | None],
    model: PyFuncModel = Depends(_get_model),
    model_info: dict[str, Any] = Depends(_get_model_info)
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    start = perf_counter()
    elapsed_ms = -1
    try:
        if inputs.get("designation") is None:
            raise HTTPException(status_code=500, detail="Field 'designation' must be provided.")
        x = np.array([[inputs.get("designation"), inputs.get("description")]], dtype="object")
        category, confidence, distribution, elapsed_ms = await run_in_threadpool(
            model.predict,
            x
        )
        response = {}
        response["output"] = {
            "category": category,
            "confidence": confidence,
            "distribution": distribution
        }
        response["metadata"] = {
            "model_info": model_info,
            "timestamp": timestamp,
            "inference_time_ms": elapsed_ms,
            "on_gpu": False
        }

        INFERENCE_REQUESTS.labels(
            mode="single",
            model=model_info["name"] + "-" + model_info["version"],
            status="success"
        ).inc()
        return response
    except Exception:
        INFERENCE_REQUESTS.labels(
            mode="single",
            model=model_info["name"] + "-" + model_info["version"],
            status="failure"
        ).inc()
        raise
    finally:
        INFERENCE_DURATION.labels(
            mode="single",
            model=model_info["name"] + model_info["version"],
        ).observe(0.001 * elapsed_ms if elapsed_ms > 0 else (perf_counter() - start))


@router.post("/batch", summary="predict batch")
async def predict_batch(
    inputs: list[dict[str, str | None]],
    model: PyFuncModel = Depends(_get_model),
    model_info: dict[str, Any] = Depends(_get_model_info)
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    start = perf_counter()
    elapsed_ms = -1
    response = {"output": [], "metadata": ...}
    batch = []
    try:
        for inp in inputs:
            if inp.get("designation") is None:
                raise HTTPException(status_code=500, detail="Field 'designation' must be provided.")
            batch.append([inp.get("designation"), inp.get("description")])
        x = np.array(batch, dtype="object")
        y_batch, elapsed_ms, mean_elapsed_ms = await run_in_threadpool(
            model.predict,
            x
        )
        for category, confidence, distribution in y_batch:
            response["output"].append(
                {
                    "category": category,
                    "confidence": confidence,
                    "distribution": distribution
                }
            )
        response["metadata"] = {
            "model_info": model_info,
            "timestamp": timestamp,
            "mean_inference_time_ms": mean_elapsed_ms,
            "total_inference_time_ms": elapsed_ms,
            "on_gpu": False
        }

        INFERENCE_REQUESTS.labels(
            mode="batch",
            model=model_info["name"] + "-" + model_info["version"],
            status="success"
        ).inc()

        return response
    except Exception:
        INFERENCE_REQUESTS.labels(
            mode="batch",
            model=model_info["name"] + "-" + model_info["version"],
            status="failure"
        ).inc()
        raise
    finally:
        INFERENCE_DURATION.labels(
            mode="batch",
            model=model_info["name"] + "-" + model_info["version"],
        ).observe(0.001 * elapsed_ms if elapsed_ms > 0 else (perf_counter() - start))
