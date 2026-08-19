def test_predict_returns_plausible_price(client, sample_property_payload):
    r = client.post("/api/v1/predictions", json=sample_property_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_price"] > 0
    assert body["price_band"] in {"Entry", "Standard", "Premium", "Luxury"}
    assert "prediction_id" in body


def test_predict_rejects_invalid_grade(client, sample_property_payload):
    payload = {**sample_property_payload, "grade": 99}
    r = client.post("/api/v1/predictions", json=payload)
    assert r.status_code == 422


def test_predict_batch(client, sample_property_payload):
    payload = {"properties": [sample_property_payload, sample_property_payload]}
    r = client.post("/api/v1/predictions/batch", json=payload)
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_explain_returns_shap_contributions(client, sample_property_payload):
    r = client.post("/api/v1/predictions/explain", json=sample_property_payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["contributions"]) == 20
    assert "approx_dollar_note" in body


def test_feature_importance_ranks_20_features(client):
    r = client.get("/api/v1/model/feature-importance")
    assert r.status_code == 200
    assert len(r.json()) == 20


def test_feature_statistics_has_correlation_and_quartiles_for_all_features(client):
    r = client.get("/api/v1/model/feature-statistics")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 20
    stat = body["sqft_living"]
    assert -1 <= stat["correlation_with_price"] <= 1
    assert stat["min"] <= stat["q1"] <= stat["median"] <= stat["q3"] <= stat["max"]


def test_feature_sample_returns_requested_size(client):
    r = client.get("/api/v1/model/feature-sample", params={"sample_size": 50})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 50
    assert "price" in body[0]
    assert "sqft_living" in body[0]
