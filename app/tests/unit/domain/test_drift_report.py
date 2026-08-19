from app.domain.entities.drift_report import DriftReport


def test_classify_stable():
    assert DriftReport.classify(0.05) == "stable"


def test_classify_moderate():
    assert DriftReport.classify(0.15) == "moderate"


def test_classify_significant():
    assert DriftReport.classify(0.3) == "significant"
