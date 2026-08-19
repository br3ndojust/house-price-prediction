import pytest

from app.domain.entities.prediction import Prediction
from app.domain.entities.property import Property
from app.infrastructure.persistence.db import create_all_tables
from app.infrastructure.persistence.prediction_repository import SqlFeedbackRepository, SqlPredictionRepository

PROP = Property(
    bedrooms=3, bathrooms=2.0, sqft_living=1800, sqft_lot=5000, floors=1.0, waterfront=0,
    view=0, condition=3, grade=7, sqft_above=1800, sqft_basement=0, yr_built=1990,
    yr_renovated=0, zipcode=98101, lat=47.6, long=-122.3, sqft_living15=1800, sqft_lot15=5000,
)


@pytest.fixture(autouse=True)
def _tables():
    create_all_tables()


def test_log_and_get_roundtrip():
    repo = SqlPredictionRepository()
    prediction = Prediction(property=PROP, predicted_price=400000, price_band="Standard",
                             property_cluster=0, model_version="model_final")
    prediction_id = repo.log(prediction)

    fetched = repo.get(prediction_id)
    assert fetched is not None
    assert fetched.predicted_price == 400000
    assert fetched.property.zipcode == 98101
    assert fetched.actual_price is None


def test_feedback_marks_prediction_with_actual_price():
    pred_repo = SqlPredictionRepository()
    feedback_repo = SqlFeedbackRepository()
    prediction = Prediction(property=PROP, predicted_price=400000, price_band="Standard",
                             property_cluster=0, model_version="model_final")
    prediction_id = pred_repo.log(prediction)

    feedback_repo.record(prediction_id, 415000.0)

    fetched = pred_repo.get(prediction_id)
    assert fetched.actual_price == 415000.0
    assert fetched.has_feedback is True
    assert pred_repo.list_with_feedback() != []


def test_summary_counts_by_band():
    repo = SqlPredictionRepository()
    for band in ("Entry", "Entry", "Luxury"):
        repo.log(Prediction(property=PROP, predicted_price=100, price_band=band, property_cluster=0,
                             model_version="v1"))
    summary = repo.summary()
    assert summary["by_price_band"]["Entry"] == 2
    assert summary["by_price_band"]["Luxury"] == 1
