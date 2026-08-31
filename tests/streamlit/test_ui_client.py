"""Contract between the Streamlit front-end and the gateway API."""

import pytest


def test_login_posts_the_credentials_as_a_form(api) -> None:
    api.reply(payload={"access_token": "a-token", "token_type": "bearer"})

    token = api.client.login("strincal", "Strincal@123!")

    assert token == "a-token"
    sent = api.calls[0]
    assert sent["method"] == "POST"
    assert sent["url"].endswith("/auth/login")
    # OAuth2PasswordRequestForm reads a form body, not JSON.
    assert sent["data"] == {"username": "strincal", "password": "Strincal@123!"}
    assert "json" not in sent


def test_login_sends_no_bearer_header(api) -> None:
    api.reply(payload={"access_token": "a-token"})

    api.client.login("strincal", "Strincal@123!")

    assert api.calls[0]["headers"] == {}


def test_authenticated_calls_carry_the_bearer_token(api) -> None:
    api.reply(payload={"username": "strincal", "role": "user", "disabled": False})

    api.client.get_me("a-token")

    assert api.calls[0]["headers"] == {"Authorization": "Bearer a-token"}


def test_predict_sends_designation_and_description(api) -> None:
    api.reply(payload={"output": {"category": "2583"}, "metadata": {}})

    api.client.predict("a-token", "piscine gonflable", "avec filtre")

    sent = api.calls[0]
    assert sent["url"].endswith("/predict/single")
    assert sent["json"] == {"designation": "piscine gonflable", "description": "avec filtre"}


def test_predict_turns_an_empty_description_into_null(api) -> None:
    """The API models description as `str | None`; an empty box must not send ""."""
    api.reply(payload={"output": {}, "metadata": {}})

    api.client.predict("a-token", "piscine gonflable")

    assert api.calls[0]["json"]["description"] is None


def test_predict_returns_the_body_untouched(api) -> None:
    body = {
        "output": {"category": "2583", "confidence": 0.87, "distribution": {"2583": 0.87}},
        "metadata": {"model_info": {"name": "rakuten-classifier", "version": "3"}},
    }
    api.reply(payload=body)

    assert api.client.predict("a-token", "piscine") == body


def test_an_http_error_is_reported_as_an_api_error(api) -> None:
    api.reply(status_code=401)

    with pytest.raises(api.client.ApiError) as failure:
        api.client.get_me("expired-token")

    assert "401" in str(failure.value)
    assert "/auth/me" in str(failure.value)


def test_an_unreachable_api_is_reported_as_an_api_error(api) -> None:
    import requests

    api.fail_with(requests.exceptions.ConnectionError("connection refused"))

    with pytest.raises(api.client.ApiError) as failure:
        api.client.get_me("a-token")

    assert "Impossible de joindre" in str(failure.value)


def test_a_missing_model_is_reported_as_an_api_error(api) -> None:
    """No model registered yet is the state the demo starts from."""
    api.reply(status_code=500)

    with pytest.raises(api.client.ApiError):
        api.client.get_current_model("a-token")


def test_the_base_url_comes_from_the_environment(api) -> None:
    api.reply(payload={"access_token": "a-token"})

    api.client.login("strincal", "Strincal@123!")

    assert api.calls[0]["url"].startswith(api.client.API_BASE_URL)
