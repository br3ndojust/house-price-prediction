from src.evaluation.promotion_criterion import check_promotion_criterion


def _band(price_band, mae, n=50):
    return {"price_band": price_band, "n": n, "mae": mae}


def test_check_promotion_criterion_recommends_when_not_worse():
    candidate = [_band("Standard", 60000), _band("Luxury", 210000)]
    active = [_band("Standard", 61000), _band("Luxury", 215000)]
    out = check_promotion_criterion(candidate, active, tolerance=0.02)
    assert out["recommended"] is True
    assert out["violations"] == []


def test_check_promotion_criterion_rejects_when_segment_worse_beyond_tolerance():
    candidate = [_band("Standard", 60000), _band("Luxury", 300000)]
    active = [_band("Standard", 61000), _band("Luxury", 215000)]
    out = check_promotion_criterion(candidate, active, tolerance=0.02)
    assert out["recommended"] is False
    assert len(out["violations"]) == 1
    assert out["violations"][0]["price_band"] == "Luxury"


def test_check_promotion_criterion_ignores_low_sample_active_segment():
    candidate = [_band("Luxury", 300000)]
    active = [_band("Luxury", 215000, n=5)]  # abaixo de LOW_SAMPLE_THRESHOLD
    out = check_promotion_criterion(candidate, active, tolerance=0.02)
    assert out["recommended"] is True
    assert out["violations"] == []
