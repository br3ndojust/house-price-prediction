from datetime import datetime, timedelta

from app.application.use_cases.list_predictions import ListPredictions
from app.domain.entities.prediction import Prediction
from app.domain.entities.property import Property

PROP = Property(
    bedrooms=3, bathrooms=2.0, sqft_living=1800, sqft_lot=5000, floors=1.0, waterfront=0,
    view=0, condition=3, grade=7, sqft_above=1800, sqft_basement=0, yr_built=1990,
    yr_renovated=0, zipcode=98101, lat=47.6, long=-122.3, sqft_living15=1800, sqft_lot15=5000,
)


class FakeLogger:
    def __init__(self, recent, with_feedback):
        self._recent = recent
        self._with_feedback = with_feedback

    def list_recent(self, limit=100, model_version=None):
        recent = self._recent
        if model_version:
            recent = [p for p in recent if p.model_version == model_version]
        return recent[:limit]

    def list_with_feedback(self, model_version=None):
        if model_version:
            return [p for p in self._with_feedback if p.model_version == model_version]
        return self._with_feedback


def test_list_predictions_default_uses_recent():
    recent = [Prediction(property=PROP, predicted_price=1, price_band="Entry", property_cluster=0,
                          model_version="v1", id="p1")]
    use_case = ListPredictions(FakeLogger(recent, []))
    assert use_case.execute() == recent


def test_list_predictions_with_feedback_sorted_desc_and_limited():
    older = Prediction(property=PROP, predicted_price=1, price_band="Entry", property_cluster=0,
                        model_version="v1", id="p1", actual_price=100,
                        created_at=datetime.utcnow() - timedelta(hours=2))
    newer = Prediction(property=PROP, predicted_price=1, price_band="Entry", property_cluster=0,
                        model_version="v1", id="p2", actual_price=200,
                        created_at=datetime.utcnow())
    use_case = ListPredictions(FakeLogger([], [older, newer]))

    result = use_case.execute(with_feedback=True, limit=1)
    assert result == [newer]
