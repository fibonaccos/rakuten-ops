"""Minimal HTTP client for the team's real FastAPI backend (see openapi.json)."""

import os

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


class ApiError(Exception):
    """Raised when a call to the API fails (network issue or non-2xx response)."""


def _call(method: str, path: str, token: str | None = None, **kwargs) -> dict:
    """Send a request to the API and return the parsed JSON body.

    Args:
        method: HTTP verb, e.g. "GET" or "POST".
        path: API path starting with "/", e.g. "/auth/login".
        token: Optional bearer token to authenticate the request.
        **kwargs: Forwarded to requests.request (json=, data=, ...).

    Returns:
        The response body, parsed as JSON.
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
