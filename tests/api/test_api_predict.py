"""Prediction routes: authorisation, contract with the inference service, journaling."""

from datetime import UTC, datetime

import pytest


def _inference_reply(category: str = "2583", confidence: float = 0.87) -> dict:
    """A well-formed answer from the inference service, as the API expects it."""
    return {
        "output": {
            "category": category,
            "confidence": confidence,
            "distribution": {category: confidence, "1140": 1 - confidence},
        },
        "metadata": {
            "model_info": {
                "name": "rakuten-classifier",
                "version": "3",
                "published_at": datetime(2026, 8, 20, tzinfo=UTC).isoformat(),
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "inference_time_ms": 42.0,
            "on_gpu": False,
        },
    }


def _batch_inference_reply(size: int = 2) -> dict:
    reply = {
        "output": [
            {
                "category": "2583",
                "confidence": 0.87,
                "distribution": {"2583": 0.87, "1140": 0.13},
            }
            for _ in range(size)
        ],
        "metadata": {
            "model_info": {
                "name": "rakuten-classifier",
                "version": "3",
                "published_at": datetime(2026, 8, 20, tzinfo=UTC).isoformat(),
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "mean_inference_time_ms": 21.0,
            "total_inference_time_ms": 42.0,
            "on_gpu": False,
        },
    }
    return reply


@pytest.fixture
def as_user(client, session, user_factory):
    """Authenticate the client as a plain user."""
    from services.auth import create_access_token

    session.user = user_factory(username="strincal")
    token = create_access_token(subject="strincal")
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


@pytest.fixture
def as_admin(client, session, user_factory):
    """Authenticate the client as an admin."""
    from db.models import UserRole
    from services.auth import create_access_token

    session.user = user_factory(username="rmazoyer", role=UserRole.ADMIN)
    token = create_access_token(subject="rmazoyer")
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


def test_single_prediction_requires_a_token(client) -> None:
    response = client.post("/predict/single", json={"designation": "piscine gonflable"})

    assert response.status_code == 401


def test_single_prediction_rejects_a_missing_designation(as_user) -> None:
    response = as_user.post("/predict/single", json={"description": "sans designation"})

    assert response.status_code == 422


def test_single_prediction_returns_the_inference_output(as_user, monkeypatch) -> None:
    import routes.predict as predict_route

    async def fake_request(_inputs):
        return _inference_reply()

    monkeypatch.setattr(predict_route, "request_predict_single", fake_request)

    response = as_user.post(
        "/predict/single",
        json={"designation": "piscine gonflable", "description": "avec filtre"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"]["category"] == "2583"
    assert body["output"]["confidence"] == pytest.approx(0.87)
    assert body["metadata"]["model_info"]["name"] == "rakuten-classifier"


def test_single_prediction_is_journaled_for_the_caller(as_user, session, monkeypatch) -> None:
    import routes.predict as predict_route

    async def fake_request(_inputs):
        return _inference_reply()

    monkeypatch.setattr(predict_route, "request_predict_single", fake_request)

    as_user.post("/predict/single", json={"designation": "piscine gonflable"})

    assert len(session.added) == 1
    inference = session.added[0]
    assert inference.user_id == 1
    assert inference.batch is False
    assert inference.designation == "piscine gonflable"
    assert inference.predicted_category == "2583"


def test_single_prediction_forwards_the_description_as_sent(as_user, monkeypatch) -> None:
    import routes.predict as predict_route

    seen: dict = {}

    async def fake_request(inputs):
        seen.update(inputs)
        return _inference_reply()

    monkeypatch.setattr(predict_route, "request_predict_single", fake_request)

    as_user.post(
        "/predict/single",
        json={"designation": "carte dracaufeu", "description": "edition limitee"},
    )

    assert seen == {"designation": "carte dracaufeu", "description": "edition limitee"}


def test_batch_prediction_is_reserved_to_admins(as_user, monkeypatch) -> None:
    import routes.predict as predict_route

    async def fake_request(_inputs):
        return _batch_inference_reply()

    monkeypatch.setattr(predict_route, "request_predict_batch", fake_request)

    response = as_user.post("/predict/batch", json=[{"designation": "piscine"}])

    assert response.status_code == 401


def test_batch_prediction_returns_one_output_per_input(as_admin, monkeypatch) -> None:
    import routes.predict as predict_route

    async def fake_request(_inputs):
        return _batch_inference_reply(size=2)

    monkeypatch.setattr(predict_route, "request_predict_batch", fake_request)

    response = as_admin.post(
        "/predict/batch",
        json=[{"designation": "piscine gonflable"}, {"designation": "carte dracaufeu"}],
    )

    assert response.status_code == 200
    assert len(response.json()["output"]) == 2


def test_batch_prediction_journals_every_input(as_admin, session, monkeypatch) -> None:
    import routes.predict as predict_route

    async def fake_request(_inputs):
        return _batch_inference_reply(size=2)

    monkeypatch.setattr(predict_route, "request_predict_batch", fake_request)

    as_admin.post(
        "/predict/batch",
        json=[{"designation": "piscine gonflable"}, {"designation": "carte dracaufeu"}],
    )

    assert session.commits == 1
    assert len(session.executed) >= 1


def test_models_route_is_reserved_to_admins(as_user) -> None:
    assert as_user.get("/models").status_code == 401
    assert as_user.get("/models/current").status_code == 401
