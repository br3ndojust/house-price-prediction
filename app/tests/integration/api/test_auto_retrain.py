def test_auto_retrain_config_defaults(client):
    r = client.get("/api/v1/training/auto-retrain-config")
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] >= 1
    assert isinstance(body["enabled"], bool)


def test_auto_retrain_config_update_requires_api_key(client):
    r = client.put("/api/v1/training/auto-retrain-config", json={"threshold": 10})
    assert r.status_code in (401, 422)


def test_auto_retrain_config_update_rejects_invalid_threshold(client, api_key_headers):
    r = client.put("/api/v1/training/auto-retrain-config", json={"threshold": 0}, headers=api_key_headers)
    assert r.status_code == 422


def test_auto_retrain_triggers_on_threshold_and_evaluates_against_active(
    client, sample_property_payload, api_key_headers
):
    r = client.put("/api/v1/training/auto-retrain-config", json={"threshold": 2}, headers=api_key_headers)
    assert r.status_code == 200
    assert r.json()["threshold"] == 2

    prediction_ids = []
    for _ in range(2):
        r = client.post("/api/v1/predictions", json=sample_property_payload)
        prediction_ids.append(r.json()["prediction_id"])

    triggered_job_id = None
    for pid in prediction_ids:
        r = client.post(
            "/api/v1/feedback", json={"prediction_id": pid, "actual_price": 410000}, headers=api_key_headers
        )
        assert r.status_code == 200
        if r.json()["auto_retrain_job_id"]:
            triggered_job_id = r.json()["auto_retrain_job_id"]
    assert triggered_job_id is not None

    r = client.get(f"/api/v1/training/jobs/{triggered_job_id}")
    assert r.json()["status"] == "done"

    r = client.get("/api/v1/model/versions")
    version = next(v for v in r.json() if v["training_job_id"] == triggered_job_id)
    assert version["triggered_by"] == "auto:volume_threshold"
    assert version["recommended_for_promotion"] in (True, False)
    assert version["promotion_eval"] is not None
    assert "criterion" in version["promotion_eval"]

    # reseta o limite alto de volta pra não interferir em outros testes que rodem depois na mesma sessão
    client.put("/api/v1/training/auto-retrain-config", json={"threshold": 100000}, headers=api_key_headers)


def test_auto_retrain_disabled_never_triggers(client, sample_property_payload, api_key_headers):
    client.put("/api/v1/training/auto-retrain-config", json={"enabled": False}, headers=api_key_headers)
    r = client.post("/api/v1/predictions", json=sample_property_payload)
    pid = r.json()["prediction_id"]
    r = client.post("/api/v1/feedback", json={"prediction_id": pid, "actual_price": 410000}, headers=api_key_headers)
    assert r.json()["auto_retrain_job_id"] is None
    client.put("/api/v1/training/auto-retrain-config", json={"enabled": True}, headers=api_key_headers)
