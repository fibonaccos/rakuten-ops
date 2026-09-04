"""Minimal HTTP client for the team's real FastAPI backend (see openapi.json)."""

import os
from typing import Any

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


class ApiError(Exception):
    """Raised when a call to the API fails (network issue or non-2xx response)."""


def _call(method: str, path: str, token: str | None = None, **kwargs) -> Any:
    """Send a request to the API and return the parsed JSON body.

    Args:
        method: HTTP verb, e.g. "GET" or "POST".
        path: API path starting with "/", e.g. "/auth/login".
        token: Optional bearer token to authenticate the request.
        **kwargs: Forwarded to requests.request (json=, data=, ...).

    Returns:
        The response body, parsed as JSON: a dict for most routes, a list for
        the ones that answer with a collection.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        response = requests.request(method, API_BASE_URL + path, headers=headers, **kwargs)
    except requests.exceptions.RequestException:
        raise ApiError(f"Impossible de joindre l'API sur {API_BASE_URL}.")

    if not response.ok:
        raise ApiError(f"Erreur API ({response.status_code}) sur {path}")

    return response.json()


def login(username: str, password: str) -> str:
    """Log in and return the access token."""
    data = _call("POST", "/auth/login", data={"username": username, "password": password})
    return data["access_token"]


def get_me(token: str) -> dict:
    """Return the currently authenticated user: {username, disabled, role}."""
    return _call("GET", "/auth/me", token=token)


def get_current_model(token: str) -> dict:
    """Return info about the model currently in production: {name, version, published_at}.

    Raises ApiError if no model is available yet (e.g. nothing trained/registered).
    """
    return _call("GET", "/models/current", token=token)


def predict(token: str, designation: str, description: str = "") -> dict:
    """Classify a product. Returns {output: {category, confidence, distribution}, metadata: {...}}."""
    payload = {"designation": designation, "description": description or None}
    return _call("POST", "/predict/single", token=token, json=payload)


def label_prediction(token: str, inference_id: int, labeled_category: str) -> dict:
    """Confirm or correct a previous single prediction.

    Send the predicted category itself to confirm it, or a different one to
    correct it. Only the user who made the original prediction can label it
    (enforced server-side). Returns {inference_id, labeled_category, matched_prediction}.
    """
    payload = {"labeled_category": labeled_category}
    return _call("PATCH", f"/predict/{inference_id}/label", token=token, json=payload)


def get_models(token: str) -> list[dict]:
    """Return every model in the registry: [{name, version, published_at}, ...].

    Admin only. Raises ApiError for a non-admin token or an unreachable registry.
    """
    return _call("GET", "/models", token=token)


def get_readiness(token: str) -> dict:
    """Return the state of each backend: {database, model_registry, inference}.

    Admin only. Each value is "ready" or "not ready".
    """
    return _call("GET", "/ready", token=token)
