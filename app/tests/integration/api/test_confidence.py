def test_prediction_includes_confidence_fields(client, sample_property_payload):
    r = client.post("/api/v1/predictions", json=sample_property_payload)
    assert r.status_code == 200
    body = r.json()
    assert "confidence_score" in body and "confidence_category" in body
    if body["confidence_score"] is not None:
        assert 0 <= body["confidence_score"] <= 100


def test_prediction_confidence_detail_endpoint(client, sample_property_payload):
    r = client.post("/api/v1/predictions/confidence", json=sample_property_payload)
    assert r.status_code in (200, 409)  # 409 só se a calibração não tiver sido gerada
    if r.status_code == 200:
        body = r.json()
        assert set(body) >= {
            "score", "category", "expected_ape", "expected_abs_error", "feature_space_distance",
            "coverage_bucket", "segment_n", "low_sample", "hard_segment", "segment_level_used",
        }
        assert body["coverage_bucket"] in {"HIGH", "MEDIUM", "LOW"}


def test_confidence_calibration_summary(client):
    r = client.get("/api/v1/model/confidence-calibration")
    assert r.status_code == 200
    body = r.json()
    if body["ready"]:
        assert set(body["category_thresholds"]) == {"alta_min", "moderada_min", "baixa_min"}
        assert (
            body["category_thresholds"]["alta_min"]
            > body["category_thresholds"]["moderada_min"]
            > body["category_thresholds"]["baixa_min"]
        )


def test_performance_summary_breaks_down_by_confidence_category(client, sample_property_payload):
    client.post("/api/v1/predictions", json=sample_property_payload)
    r = client.get("/api/v1/performance/summary")
    assert r.status_code == 200
    assert "by_confidence_category" in r.json()
