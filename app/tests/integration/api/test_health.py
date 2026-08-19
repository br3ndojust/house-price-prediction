def test_liveness(client):
    r = client.get("/api/v1/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}


def test_readiness_model_loaded(client):
    r = client.get("/api/v1/health/ready")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_detailed_health(client):
    r = client.get("/api/v1/health/detailed")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["active_model_version"] == "model_final"


def test_metrics_exposes_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text
