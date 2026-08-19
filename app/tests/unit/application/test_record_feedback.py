import pytest

from app.application.use_cases.record_feedback import RecordFeedback
from app.domain.entities.prediction import Prediction
from app.domain.entities.property import Property


class FakeFeedbackRepository:
    def __init__(self):
        self.recorded = []

    def record(self, prediction_id, actual_price):
        self.recorded.append((prediction_id, actual_price))


class FakePredictionLogger:
    def __init__(self, existing: dict):
        self._existing = existing

    def get(self, prediction_id):
        return self._existing.get(prediction_id)

    def log(self, prediction):
        raise NotImplementedError

    def list_recent(self, limit=100):
        raise NotImplementedError

    def list_with_feedback(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


def _make_prediction() -> Prediction:
    prop = Property(
        bedrooms=3, bathrooms=2.0, sqft_living=1800, sqft_lot=5000, floors=1.0, waterfront=0,
        view=0, condition=3, grade=7, sqft_above=1800, sqft_basement=0, yr_built=1990,
        yr_renovated=0, zipcode=98101, lat=47.6, long=-122.3, sqft_living15=1800, sqft_lot15=5000,
    )
    return Prediction(property=prop, predicted_price=400000, price_band="Standard",
                       property_cluster=0, model_version="model_final", id="pred-1")


def test_record_feedback_success():
    feedback_repo = FakeFeedbackRepository()
    logger = FakePredictionLogger({"pred-1": _make_prediction()})
    use_case = RecordFeedback(feedback_repo, logger)

    use_case.execute("pred-1", 420000.0)

    assert feedback_repo.recorded == [("pred-1", 420000.0)]


def test_record_feedback_unknown_prediction_raises():
    use_case = RecordFeedback(FakeFeedbackRepository(), FakePredictionLogger({}))
    with pytest.raises(ValueError):
        use_case.execute("missing", 100.0)


def test_record_feedback_non_positive_price_raises():
    logger = FakePredictionLogger({"pred-1": _make_prediction()})
    use_case = RecordFeedback(FakeFeedbackRepository(), logger)
    with pytest.raises(ValueError):
        use_case.execute("pred-1", -5.0)
