"""
Fixtures for the gateway API.

The API talks to PostgreSQL and to the inference service. Neither is available in a
test run, so the database session is replaced by an in-memory stub and the calls to
the inference service are patched per test. The startup hook, which opens a real
connection, is replaced by a no-op.
"""

import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import use_service

# Settings are read at import time by db.session, so the environment has to be
# complete before the service modules are loaded.
os.environ.update(
    {
        "RAKUTEN__API__JWT_ALGORITHM": "HS256",
        "RAKUTEN__API__JWT_SECRET": "test-secret-not-used-outside-the-test-suite",
        "RAKUTEN__API__JWT_EXPIRATION_IN_MINUTES": "60",
        "RAKUTEN__API__INFERENCE_HOST": "inference-service",
        "RAKUTEN__API__INFERENCE_PORT": "8000",
        "RAKUTEN__API__DATABASE_HOST": "database-service",
        "RAKUTEN__API__DATABASE_PORT": "5432",
        "RAKUTEN__API__DATABASE_NAME": "rakuten",
        "RAKUTEN__API__DATABASE_USER": "rakuten_api",
        "RAKUTEN__API__DATABASE_PASSWORD": "rakuten_api",
        "RAKUTEN__API__MLFLOW_HOST": "mlflow-service",
        "RAKUTEN__API__MLFLOW_PORT": "5000",
    }
)

@pytest.fixture(autouse=True)
def service() -> Path:
    """Put `services/api` on top of `sys.path` for the duration of one test."""
    return use_service("api")


class FakeResult:
    """Stands in for the object returned by `AsyncSession.execute`."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class FakeSession:
    """
    Minimal async session: returns a preset user and records what was written.

    Only the handful of calls the repositories make are implemented — `execute`,
    `add`, `commit` and `refresh`.
    """

    def __init__(self, user: Any = None) -> None:
        self.user = user
        self.added: list[Any] = []
        self.executed: list[Any] = []
        self.commits: int = 0

    async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> FakeResult:
        self.executed.append(statement)
        return FakeResult(self.user)

    def add(self, instance: Any) -> None:
        self.added.append(instance)

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, instance: Any) -> None:
        return None


@pytest.fixture
def user_factory(service: Path):
    """Build a persisted-looking User without touching the database."""
    from db.models import User, UserRole

    def _build(
        username: str = "strincal",
        password: str = "Strincal@123!",
        role: UserRole = UserRole.USER,
        user_id: int = 1,
    ) -> User:
        from services.auth import hash_password

        user = User()
        user.user_id = user_id
        user.username = username
        user.password_hash = hash_password(password)
        user.role = role
        return user

    return _build


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def client(service: Path, session: FakeSession) -> Iterator[Any]:
    """A TestClient wired to the fake session, with the database startup skipped."""
    from db.session import get_session
    from fastapi.testclient import TestClient
    from main import build_api

    app = build_api()

    @asynccontextmanager
    async def no_startup(_app: Any):
        yield

    # build_api() installs a lifespan that opens a real connection to PostgreSQL.
    app.router.lifespan_context = no_startup

    async def override_get_session():
        yield session

    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
