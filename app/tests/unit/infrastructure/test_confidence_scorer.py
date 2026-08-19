from pathlib import Path

import pandas as pd
import pytest

from app.infrastructure.confidence.confidence_scorer import ConfidenceScorer
from app.core.config import settings


def test_not_ready_when_calibration_artifact_missing(tmp_path):
    scorer = ConfidenceScorer(artifacts_dir=tmp_path, data_dir=settings.data_dir)
    assert scorer.is_ready() is False
    assert scorer.calibration_summary() == {"ready": False}


def test_score_raises_when_not_ready(tmp_path):
    scorer = ConfidenceScorer(artifacts_dir=tmp_path, data_dir=settings.data_dir)
    dummy = pd.DataFrame([{"grade": 7}])
    with pytest.raises(ValueError):
        scorer.score(dummy, "Standard", 0, 0, 7)


@pytest.mark.skipif(
    not (Path("artifacts") / "confidence_calibration.json").exists(),
    reason="requer artifacts/confidence_calibration.json (python src/scripts/build_confidence_matrix.py)",
)
def test_score_returns_plausible_confidence_with_real_calibration():
    from app.infrastructure.model.artifact_model_repository import ArtifactModelRepository

    models = ArtifactModelRepository()
    scorer = ConfidenceScorer()
    assert scorer.is_ready() is True

    raw = pd.DataFrame([{
        "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800, "sqft_lot": 5000, "floors": 1.0,
        "waterfront": 0, "view": 0, "condition": 3, "grade": 7, "sqft_above": 1800,
        "sqft_basement": 0, "yr_built": 1990, "yr_renovated": 0, "zipcode": 98042, "lat": 47.6,
        "long": -122.3, "sqft_living15": 1800, "sqft_lot15": 5000,
    }])
    model_features = models.build_model_features(raw)

    confidence = scorer.score(model_features, "Standard", 0, waterfront=0, grade=7)
    assert 0 <= confidence.score <= 100
    assert confidence.category in {
        "Alta confiança", "Confiança moderada", "Baixa confiança", "Muito baixa confiança / revisão",
    }
    assert confidence.coverage_bucket in {"HIGH", "MEDIUM", "LOW"}
    assert confidence.segment_n > 0


def test_hard_segment_gets_worse_or_equal_expected_error_than_normal():
    if not (Path("artifacts") / "confidence_calibration.json").exists():
        pytest.skip("requer artifacts/confidence_calibration.json")
    from app.infrastructure.model.artifact_model_repository import ArtifactModelRepository

    models = ArtifactModelRepository()
    scorer = ConfidenceScorer()
    raw = pd.DataFrame([{
        "bedrooms": 5, "bathrooms": 4.0, "sqft_living": 5000, "sqft_lot": 20000, "floors": 2.0,
        "waterfront": 1, "view": 4, "condition": 4, "grade": 12, "sqft_above": 4000,
        "sqft_basement": 1000, "yr_built": 2005, "yr_renovated": 0, "zipcode": 98039, "lat": 47.62,
        "long": -122.24, "sqft_living15": 4500, "sqft_lot15": 18000,
    }])
    model_features = models.build_model_features(raw)
    confidence = scorer.score(model_features, "Luxury", 2, waterfront=1, grade=12)
    assert confidence.hard_segment is True
