from app.application.use_cases.get_error_matrix import GetErrorMatrix
from app.domain.entities.prediction import Prediction
from app.domain.entities.property import Property

PROP = Property(
    bedrooms=3, bathrooms=2.0, sqft_living=1800, sqft_lot=5000, floors=1.0, waterfront=0,
    view=0, condition=3, grade=7, sqft_above=1800, sqft_basement=0, yr_built=1990,
    yr_renovated=0, zipcode=98101, lat=47.6, long=-122.3, sqft_living15=1800, sqft_lot15=5000,
)


class FakeLogger:
    def __init__(self, predictions):
        self._predictions = predictions

    def list_with_feedback(self, model_version=None):
        if model_version:
            return [p for p in self._predictions if p.model_version == model_version]
        return self._predictions


def test_error_matrix_empty_when_no_feedback():
    out = GetErrorMatrix(FakeLogger([])).execute()
    assert out["n"] == 0
    assert out["global"] is None


def test_error_matrix_computes_metrics_with_feedback():
    predictions = [
        Prediction(property=PROP, predicted_price=400000, price_band="Standard", property_cluster=0,
                   model_version="v1", id="p1", actual_price=420000),
        Prediction(property=PROP, predicted_price=800000, price_band="Luxury", property_cluster=2,
                   model_version="v1", id="p2", actual_price=750000),
    ]
    out = GetErrorMatrix(FakeLogger(predictions)).execute()
    assert out["n"] == 2
    assert out["global"]["mae"] == 35000
    bands = {row["price_band"] for row in out["by_price_band"]}
    assert bands == {"Standard", "Luxury"}
