import joblib


def test_training_job_with_feature_subset(client, api_key_headers):
    subset = ["bedrooms", "bathrooms", "sqft_living", "grade", "comps_knn_price"]
    r = client.post("/api/v1/training/jobs", json={"feature_cols": subset}, headers=api_key_headers)
    assert r.status_code == 200
    job_id = r.json()["id"]

    r = client.get(f"/api/v1/training/jobs/{job_id}")
    assert r.json()["status"] == "done"
    assert r.json()["result"]["feature_cols_used"] == subset

    candidate_path = r.json()["result"]["candidate_artifact_path"]
    bundle = joblib.load(candidate_path)
    assert bundle["feature_cols"] == subset


def test_training_job_without_feature_cols_uses_official_set(client, api_key_headers):
    r = client.post("/api/v1/training/jobs", headers=api_key_headers)
    assert r.status_code == 200
    job_id = r.json()["id"]

    r = client.get(f"/api/v1/training/jobs/{job_id}")
    assert r.json()["result"]["feature_cols_used"] is None


def test_training_job_rejects_unknown_feature(client, api_key_headers):
    r = client.post(
        "/api/v1/training/jobs", json={"feature_cols": ["not_a_real_feature"]}, headers=api_key_headers
    )
    assert r.status_code == 422
    assert "not_a_real_feature" in r.text


def test_training_job_from_upload_accepts_feature_cols_form_field(
    client, sample_property_payload, api_key_headers
):
    import io

    import pandas as pd

    row = {**sample_property_payload, "price": 420000}
    df = pd.DataFrame([row])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    files = {"file": ("lote.csv", buf.getvalue().encode("utf-8"), "text/csv")}

    r = client.post(
        "/api/v1/training/jobs/from-upload",
        files=files,
        data={"value_column": "price", "feature_cols": "bedrooms,bathrooms,sqft_living,grade"},
        headers=api_key_headers,
    )
    assert r.status_code == 200
    job_id = r.json()["id"]

    r = client.get(f"/api/v1/training/jobs/{job_id}")
    assert r.json()["status"] == "done"
    assert r.json()["result"]["feature_cols_used"] == ["bedrooms", "bathrooms", "sqft_living", "grade"]
