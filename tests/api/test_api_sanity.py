"""Health, metrics and OpenAPI surface of the gateway API."""


def test_health_reports_the_api_version(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["api_version"]


def test_health_needs_no_authentication(client) -> None:
    assert "authorization" not in {k.lower() for k in client.headers}
    assert client.get("/health").status_code == 200


def test_ready_is_reserved_to_admins(client) -> None:
    assert client.get("/ready").status_code == 401


def test_metrics_are_exposed_in_the_prometheus_format(client) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_metrics_count_the_requests_that_went_through(client) -> None:
    client.get("/auth/me")

    body = client.get("/metrics").text

    assert "http_requests_total" in body
    assert "http_requests_duration_seconds" in body


def test_openapi_schema_lists_the_documented_routes(client) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert {"/health", "/ready", "/auth/login", "/auth/me"} <= set(paths)
    assert {"/predict/single", "/predict/batch"} <= set(paths)
    assert {"/models", "/models/current"} <= set(paths)
