from fastapi import Request
from mlflow.pyfunc import PyFuncModel
from typing import Any


def _get_model(request: Request) -> PyFuncModel:
    return request.app.state.rakuten_model


def _get_model_info(request: Request) -> dict[str, Any]:
    return request.app.state.rakuten_model_info
