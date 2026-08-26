"""Password hashing, token issuance and the routes behind the token."""

from datetime import UTC

import pytest


@pytest.fixture(autouse=True)
def _auth_module():
    """Import the auth service once the service path is set up by the conftest."""
    from services import auth

    return auth


def test_a_password_never_equals_its_hash(_auth_module) -> None:
    hashed = _auth_module.hash_password("Strincal@123!")

    assert hashed != "Strincal@123!"
    assert hashed.startswith("$2b$")


def test_a_password_verifies_against_its_own_hash(_auth_module) -> None:
    hashed = _auth_module.hash_password("Strincal@123!")

    assert _auth_module.verify_password("Strincal@123!", hashed) is True
    assert _auth_module.verify_password("Rbanat@123!", hashed) is False


def test_hashing_the_same_password_twice_gives_two_hashes(_auth_module) -> None:
    """bcrypt salts each hash, so identical passwords must not share a hash."""
    first = _auth_module.hash_password("Alice@123!")
    second = _auth_module.hash_password("Alice@123!")

    assert first != second


def test_a_token_carries_the_username_and_an_expiry(_auth_module) -> None:
    from _config import get_settings
    from jose import jwt

    token = _auth_module.create_access_token(subject="strincal")
    payload = jwt.decode(
        token,
        get_settings().jwt_secret.get_secret_value(),
        algorithms=[get_settings().jwt_algorithm],
    )

    assert payload["sub"] == "strincal"
    assert payload["exp"] > 0


def test_login_returns_a_bearer_token(client, session, user_factory) -> None:
    session.user = user_factory(username="strincal", password="Strincal@123!")

    response = client.post(
        "/auth/login",
        data={"username": "strincal", "password": "Strincal@123!"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_rejects_a_wrong_password(client, session, user_factory) -> None:
    session.user = user_factory(username="strincal", password="Strincal@123!")

    response = client.post(
        "/auth/login",
        data={"username": "strincal", "password": "not-the-password"},
    )

    assert response.status_code == 401


def test_login_rejects_an_unknown_user(client, session) -> None:
    session.user = None

    response = client.post(
        "/auth/login",
        data={"username": "ghost", "password": "whatever"},
    )

    assert response.status_code == 401


def test_me_returns_the_authenticated_user(client, session, user_factory, _auth_module) -> None:
    session.user = user_factory(username="strincal")
    token = _auth_module.create_access_token(subject="strincal")

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["username"] == "strincal"
    assert response.json()["role"] == "user"


def test_me_rejects_a_missing_token(client) -> None:
    assert client.get("/auth/me").status_code == 401


def test_me_rejects_a_forged_token(client, session, user_factory) -> None:
    session.user = user_factory()

    response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-jwt"})

    assert response.status_code == 401


def test_me_rejects_a_token_signed_with_another_secret(client, session, user_factory) -> None:
    from datetime import datetime, timedelta

    from jose import jwt

    session.user = user_factory()
    token = jwt.encode(
        {"sub": "strincal", "exp": datetime.now(UTC) + timedelta(minutes=5)},
        key="another-secret",
        algorithm="HS256",
    )

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401


def test_me_rejects_a_token_whose_user_is_gone(client, session, _auth_module) -> None:
    session.user = None
    token = _auth_module.create_access_token(subject="deleted-user")

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
