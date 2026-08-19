def test_feedback_requires_api_key(client, sample_property_payload):
    r = client.post("/api/v1/predictions", json=sample_property_payload)
    prediction_id = r.json()["prediction_id"]

    r = client.post("/api/v1/feedback", json={"prediction_id": prediction_id, "actual_price": 100})
    assert r.status_code in (401, 422)  # 422 quando o header nem está presente (FastAPI valida antes)


def test_feedback_then_error_matrix(client, sample_property_payload, api_key_headers):
    r = client.post("/api/v1/predictions", json=sample_property_payload)
    prediction_id = r.json()["prediction_id"]

    r = client.post(
        "/api/v1/feedback",
        json={"prediction_id": prediction_id, "actual_price": 410000},
        headers=api_key_headers,
    )
    assert r.status_code == 200

    r = client.get("/api/v1/performance/error-matrix")
    assert r.status_code == 200
    assert r.json()["n"] >= 1


def test_feedback_unknown_prediction_404(client, api_key_headers):
    r = client.post(
        "/api/v1/feedback",
        json={"prediction_id": "does-not-exist", "actual_price": 100},
        headers=api_key_headers,
    )
    assert r.status_code == 404


def test_performance_summary_reflects_predictions(client, sample_property_payload):
    client.post("/api/v1/predictions", json=sample_property_payload)
    r = client.get("/api/v1/performance/summary")
    assert r.status_code == 200
    assert r.json()["total_predictions"] >= 1


def test_performance_endpoints_filter_by_model_version(client, sample_property_payload, api_key_headers):
    r = client.post("/api/v1/predictions", json=sample_property_payload)
    prediction_id = r.json()["prediction_id"]
    model_version = r.json()["model_version"]
    client.post(
        "/api/v1/feedback",
        json={"prediction_id": prediction_id, "actual_price": 415000},
        headers=api_key_headers,
    )

    r = client.get("/api/v1/performance/model-versions")
    assert r.status_code == 200
    assert model_version in r.json()

    r = client.get("/api/v1/performance/summary", params={"model_version": model_version})
    assert r.status_code == 200
    assert r.json()["total_predictions"] >= 1

    r = client.get("/api/v1/performance/summary", params={"model_version": "does-not-exist"})
    assert r.status_code == 200
    assert r.json()["total_predictions"] == 0

    r = client.get("/api/v1/performance/error-matrix", params={"model_version": model_version})
    assert r.status_code == 200
    assert r.json()["n"] >= 1

    r = client.get("/api/v1/performance/error-matrix", params={"model_version": "does-not-exist"})
    assert r.status_code == 200
    assert r.json()["n"] == 0

    r = client.get("/api/v1/performance/predictions", params={"model_version": model_version})
    assert r.status_code == 200
    assert prediction_id in [p["id"] for p in r.json()]

    r = client.get("/api/v1/performance/predictions", params={"model_version": "does-not-exist"})
    assert r.status_code == 200
    assert prediction_id not in [p["id"] for p in r.json()]


def test_list_predictions_returns_recent(client, sample_property_payload):
    r = client.post("/api/v1/predictions", json=sample_property_payload)
    prediction_id = r.json()["prediction_id"]

    r = client.get("/api/v1/performance/predictions")
    assert r.status_code == 200
    ids = [p["id"] for p in r.json()]
    assert prediction_id in ids
    assert r.json()[0]["actual_price"] is None


def test_list_predictions_with_feedback_filter(client, sample_property_payload, api_key_headers):
    r = client.post("/api/v1/predictions", json=sample_property_payload)
    prediction_id = r.json()["prediction_id"]

    r = client.get("/api/v1/performance/predictions?with_feedback=true")
    assert r.status_code == 200
    assert prediction_id not in [p["id"] for p in r.json()]

    client.post(
        "/api/v1/feedback",
        json={"prediction_id": prediction_id, "actual_price": 420000},
        headers=api_key_headers,
    )

    r = client.get("/api/v1/performance/predictions?with_feedback=true")
    assert r.status_code == 200
    by_id = {p["id"]: p for p in r.json()}
    assert prediction_id in by_id
    assert by_id[prediction_id]["actual_price"] == 420000
