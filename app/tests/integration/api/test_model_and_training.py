def test_model_info_returns_contract(client):
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    assert r.json()["status"] == "RECOMMENDED_FOR_PROMOTION"


def test_baseline_model_final_is_registered_and_selectable_for_rollback(client):
    r = client.get("/api/v1/model/versions")
    assert r.status_code == 200
    versions = {v["version"]: v for v in r.json()}
    assert "model_final" in versions
    baseline = versions["model_final"]
    assert baseline["status"] in {"active", "candidate", "superseded"}
    assert baseline["artifact_path"].endswith("model_final.pkl")
    assert baseline["n_features"] and baseline["n_features"] > 0
    assert baseline["cv_mae"] and baseline["test_mae"] and baseline["val_mae"]


def test_baseline_model_final_has_feature_snapshot(client):
    r = client.get("/api/v1/model/feature-snapshots")
    assert r.status_code == 200
    versions = {s["version"] for s in r.json()}
    assert "model_final" in versions

    r = client.get("/api/v1/model/feature-snapshots/model_final")
    assert r.status_code == 200
    body = r.json()
    assert len(body["features"]) > 0
    stat = next(iter(body["features"].values()))
    assert "importance" in stat and "correlation_with_price" in stat and "median" in stat


def test_feature_snapshot_unknown_version_404(client):
    r = client.get("/api/v1/model/feature-snapshots/does-not-exist")
    assert r.status_code == 404


def test_promote_requires_api_key(client):
    r = client.post("/api/v1/model/promote", json={"version": "whatever"})
    assert r.status_code in (401, 422)


def test_promote_unknown_version_404(client, api_key_headers):
    r = client.post("/api/v1/model/promote", json={"version": "does-not-exist"}, headers=api_key_headers)
    assert r.status_code == 404


def test_training_job_requires_api_key(client):
    r = client.post("/api/v1/training/jobs")
    assert r.status_code in (401, 422)


def test_training_job_lifecycle(client, api_key_headers):
    r = client.post("/api/v1/training/jobs", headers=api_key_headers)
    assert r.status_code == 200
    job_id = r.json()["id"]

    r = client.get(f"/api/v1/training/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "done"
    result = r.json()["result"]
    assert result["val_check"]["global"]["mae"] > 0
    assert result["test_check"]["global"]["mae"] > 0
    assert result["test_check"]["split_used"] == "test"

    r = client.get("/api/v1/model/versions")
    versions = r.json()
    trained = next(v for v in versions if v["training_job_id"] == job_id)
    assert trained["n_features"] and trained["n_features"] > 0
    assert trained["cv_mae"] and trained["test_mae"] and trained["val_mae"]

    r = client.get(f"/api/v1/model/feature-snapshots/{trained['version']}")
    assert r.status_code == 200
    assert len(r.json()["features"]) == trained["n_features"]

    pm = trained["performance_metrics"]
    for split in ("train", "test", "val"):
        for metric in ("mae", "rmse", "mape", "r2_log"):
            assert pm[split][metric] is not None, f"{split}.{metric} não deveria ser None"
    assert 0 < pm["train"]["mape"] < 1  # normalizado como fração, não pontos percentuais
