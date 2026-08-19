def test_drift_evaluate_requires_api_key(client):
    r = client.post("/api/v1/drift/evaluate")
    assert r.status_code in (401, 422)


def test_drift_evaluate_and_read_reports(client, api_key_headers):
    r = client.post("/api/v1/drift/evaluate", headers=api_key_headers)
    assert r.status_code == 200
    assert r.json()["overall_status"] in {"insufficient_data", "stable", "moderate", "significant"}

    r = client.get("/api/v1/drift/reports")
    assert r.status_code == 200
    assert len(r.json()) >= 1

    r = client.get("/api/v1/drift/reports/latest")
    assert r.status_code == 200


def test_drift_report_includes_prediction_drift_with_enough_samples(
    client, sample_property_payload, api_key_headers
):
    payload = {"properties": [sample_property_payload] * 30}
    client.post("/api/v1/predictions/batch", json=payload)

    r = client.post("/api/v1/drift/evaluate", headers=api_key_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["overall_status"] != "insufficient_data"
    pred_drift = body["prediction_drift"]
    assert pred_drift is not None
    assert set(pred_drift) == {
        "train_mean", "train_median", "recent_mean", "recent_median", "mean_change_pct", "psi",
    }
