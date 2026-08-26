"""Routes of the inference service, driven against a stub model."""


def test_health_reports_the_service_version(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["api_version"]


def test_metrics_are_exposed_in_the_prometheus_format(client) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_current_model_reports_what_was_loaded_at_startup(client) -> None:
    response = client.get("/models/current")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "rakuten-classifier"
    assert body["version"] == "3"


def test_single_prediction_returns_category_confidence_and_distribution(client) -> None:
    response = client.post(
        "/predict/single",
        json={"designation": "piscine gonflable", "description": "avec filtre"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"]["category"] == "2583"
    assert body["output"]["confidence"] == 0.87
    assert body["output"]["distribution"]["2583"] == 0.87


def test_single_prediction_reports_its_own_timing(client) -> None:
    body = client.post("/predict/single", json={"designation": "piscine"}).json()

    assert body["metadata"]["inference_time_ms"] == 12.5
    assert body["metadata"]["on_gpu"] is False
    assert body["metadata"]["model_info"]["name"] == "rakuten-classifier"


def test_single_prediction_passes_designation_and_description_to_the_model(
    client, model
) -> None:
    client.post(
        "/predict/single",
        json={"designation": "carte dracaufeu", "description": "edition limitee"},
    )

    assert model.calls[0].shape == (1, 2)
    assert list(model.calls[0][0]) == ["carte dracaufeu", "edition limitee"]


def test_single_prediction_accepts_a_missing_description(client, model) -> None:
    response = client.post("/predict/single", json={"designation": "piscine", "description": None})

    assert response.status_code == 200
    assert model.calls[0][0][1] is None


def test_single_prediction_refuses_a_missing_designation(client) -> None:
    response = client.post("/predict/single", json={"description": "sans designation"})

    assert response.status_code == 500
    assert "designation" in response.json()["detail"]


def test_batch_prediction_returns_one_output_per_input(client) -> None:
    response = client.post(
        "/predict/batch",
        json=[{"designation": "piscine gonflable"}, {"designation": "carte dracaufeu"}],
    )

    assert response.status_code == 200
    assert len(response.json()["output"]) == 2


def test_batch_prediction_reports_total_and_mean_timings(client) -> None:
    body = client.post(
        "/predict/batch",
        json=[{"designation": "piscine"}, {"designation": "carte"}],
    ).json()

    assert body["metadata"]["total_inference_time_ms"] == 25.0
    assert body["metadata"]["mean_inference_time_ms"] == 12.5


def test_batch_prediction_builds_one_row_per_input(client, model) -> None:
    client.post(
        "/predict/batch",
        json=[{"designation": "piscine"}, {"designation": "carte"}, {"designation": "livre"}],
    )

    assert model.calls[0].shape == (3, 2)


def test_batch_prediction_refuses_a_batch_with_a_missing_designation(client) -> None:
    response = client.post(
        "/predict/batch",
        json=[{"designation": "piscine"}, {"description": "sans designation"}],
    )

    assert response.status_code == 500


def test_inference_metrics_are_incremented_by_a_prediction(client) -> None:
    client.post("/predict/single", json={"designation": "piscine"})

    body = client.get("/metrics").text

    assert "inference_requests_total" in body
    assert "inference_duration_seconds" in body
