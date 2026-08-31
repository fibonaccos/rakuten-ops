"""
Fixtures for the Streamlit front-end.

Only the HTTP client is exercised here: it carries the contract with the gateway
API and is the one part of the front-end that runs without a browser session.
`requests.request` is replaced by a recorder so tests can assert on what was sent
and dictate what comes back.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import use_service


@pytest.fixture(autouse=True)
def service() -> Path:
    """Put `services/streamlit` on top of `sys.path` for the duration of one test."""
    return use_service("streamlit")


class FakeResponse:
    """Stands in for `requests.Response`."""

    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        return self._payload


@dataclass
class ApiHarness:
    """The API client module, plus the requests it sent and the replies it will get."""

    client: Any
    calls: list[dict] = field(default_factory=list)
    _replies: list[FakeResponse] = field(default_factory=list)

    def reply(self, status_code: int = 200, payload: Any = None) -> None:
        """Queue the next response the API will return."""
        self._replies.append(FakeResponse(status_code, payload))

    def fail_with(self, exception: BaseException) -> None:
        """Make the next call raise, as a network failure would."""
        self._replies.append(exception)  # type: ignore[arg-type]

    def next_reply(self) -> FakeResponse:
        if not self._replies:
            return FakeResponse()
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


@pytest.fixture
def api(service: Path, monkeypatch) -> ApiHarness:
    import requests

    from services import api_client

    harness = ApiHarness(client=api_client)

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        harness.calls.append({"method": method, "url": url, **kwargs})
        return harness.next_reply()

    monkeypatch.setattr(requests, "request", fake_request)
    return harness
