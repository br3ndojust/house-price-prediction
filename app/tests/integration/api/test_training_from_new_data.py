import io

import pandas as pd


def test_from_feedback_requires_api_key(client):
    r = client.post("/api/v1/training/jobs/from-feedback")
    assert r.status_code in (401, 422)


def test_from_feedback_lifecycle(client, sample_property_payload, api_key_headers):
    r = client.post("/api/v1/predictions", json=sample_property_payload)
    prediction_id = r.json()["prediction_id"]
    client.post(
        "/api/v1/feedback",
        json={"prediction_id": prediction_id, "actual_price": 415000},
        headers=api_key_headers,
    )

    # suíte roda com um SQLite compartilhado por sessão (app/tests/conftest.py) — outros testes podem
    # já ter registrado feedback antes deste, então só garante "pelo menos a linha que este teste criou"
    r = client.post("/api/v1/training/jobs/from-feedback", headers=api_key_headers)
    assert r.status_code == 200
    n_staged = r.json()["result"]["n_extra_rows_staged"]
    assert n_staged >= 1
    job_id = r.json()["id"]

    r = client.get(f"/api/v1/training/jobs/{job_id}")
    assert r.json()["status"] == "done"
    assert r.json()["result"]["n_extra_rows_from_new_data"] == n_staged

    r = client.get("/api/v1/model/versions")
    version = next(v for v in r.json() if v["training_job_id"] == job_id)
    assert version["n_train_rows"] == 17744 + n_staged


def test_from_upload_requires_api_key(client, tmp_path):
    csv_bytes = b"bedrooms,price\n3,400000\n"
    files = {"file": ("t.csv", csv_bytes, "text/csv")}
    r = client.post("/api/v1/training/jobs/from-upload", files=files, data={"value_column": "price"})
    assert r.status_code in (401, 422)


def test_from_upload_missing_columns_returns_422(client, api_key_headers):
    csv_bytes = b"bedrooms,price\n3,400000\n"
    files = {"file": ("t.csv", csv_bytes, "text/csv")}
    r = client.post(
        "/api/v1/training/jobs/from-upload",
        files=files, data={"value_column": "price"}, headers=api_key_headers,
    )
    assert r.status_code == 422
    assert "bathrooms" in r.text


def test_from_upload_lifecycle(client, sample_property_payload, api_key_headers):
    row = {**sample_property_payload, "price": 420000}
    df = pd.DataFrame([row, row])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    files = {"file": ("lote.csv", buf.getvalue().encode("utf-8"), "text/csv")}

    r = client.post(
        "/api/v1/training/jobs/from-upload",
        files=files, data={"value_column": "price"}, headers=api_key_headers,
    )
    assert r.status_code == 200
    assert r.json()["result"]["n_extra_rows_staged"] == 2
    job_id = r.json()["id"]

    r = client.get(f"/api/v1/training/jobs/{job_id}")
    assert r.json()["status"] == "done"


def test_from_upload_test_size_controls_split(client, sample_property_payload, api_key_headers):
    row = {**sample_property_payload, "price": 420000}
    df = pd.DataFrame([row] * 6)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    files = {"file": ("lote.csv", buf.getvalue().encode("utf-8"), "text/csv")}

    r = client.post(
        "/api/v1/training/jobs/from-upload",
        files=files, data={"value_column": "price", "test_size": "1.0"}, headers=api_key_headers,
    )
    assert r.status_code == 200
    job_id = r.json()["id"]

    r = client.get(f"/api/v1/training/jobs/{job_id}")
    assert r.json()["result"]["n_extra_rows_test_split"] == 6

    buf2 = io.StringIO()
    df.to_csv(buf2, index=False)
    files2 = {"file": ("lote2.csv", buf2.getvalue().encode("utf-8"), "text/csv")}
    r = client.post(
        "/api/v1/training/jobs/from-upload",
        files=files2, data={"value_column": "price", "test_size": "0.0"}, headers=api_key_headers,
    )
    job_id2 = r.json()["id"]
    r = client.get(f"/api/v1/training/jobs/{job_id2}")
    assert r.json()["result"]["n_extra_rows_test_split"] == 0


def test_from_upload_test_size_out_of_range_returns_422(client, sample_property_payload, api_key_headers):
    row = {**sample_property_payload, "price": 420000}
    df = pd.DataFrame([row])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    files = {"file": ("t.csv", buf.getvalue().encode("utf-8"), "text/csv")}

    r = client.post(
        "/api/v1/training/jobs/from-upload",
        files=files, data={"value_column": "price", "test_size": "1.5"}, headers=api_key_headers,
    )
    assert r.status_code == 422


def test_from_upload_invalid_row_value_returns_422(client, sample_property_payload, api_key_headers):
    row = {**sample_property_payload, "grade": 99, "price": 420000}  # grade fora do intervalo 1-13
    df = pd.DataFrame([row])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    files = {"file": ("lote.csv", buf.getvalue().encode("utf-8"), "text/csv")}

    r = client.post(
        "/api/v1/training/jobs/from-upload",
        files=files, data={"value_column": "price"}, headers=api_key_headers,
    )
    assert r.status_code == 422
    assert any(err.get("campo") == "grade" for err in r.json()["detail"])


def test_from_upload_with_column_order_override(client, sample_property_payload, api_key_headers):
    ordered_keys = list(sample_property_payload.keys()) + ["price"]
    row = [sample_property_payload[k] for k in ordered_keys[:-1]] + [420000]
    csv_bytes = ",".join(str(v) for v in row).encode("utf-8") + b"\n"
    files = {"file": ("no_header.csv", csv_bytes, "text/csv")}

    r = client.post(
        "/api/v1/training/jobs/from-upload",
        files=files,
        data={"value_column": "price", "column_order": ",".join(ordered_keys), "has_header": "false"},
        headers=api_key_headers,
    )
    assert r.status_code == 200
    assert r.json()["result"]["n_extra_rows_staged"] == 1
