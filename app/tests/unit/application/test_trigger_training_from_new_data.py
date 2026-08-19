import numpy as np
import pandas as pd
import pytest

from app.application.use_cases.trigger_training_from_new_data import (
    TriggerTrainingFromFeedback,
    TriggerTrainingFromUpload,
)
from app.domain.entities.property import Property
from app.domain.entities.training_job import TrainingJob


class FakeModelRepository:
    def training_materials(self) -> dict:
        return {
            "demographics": None, "spatial_index": None,
            "price_breakpoints": None, "cluster_scaler": None, "cluster_kmeans": None,
        }


class FakeTrainingService:
    def __init__(self):
        self.staged = {}

    def create_job(self, triggered_by):
        return TrainingJob(id="job-1", triggered_by=triggered_by)

    def stage_extra_rows(self, job_id, extra_rows):
        self.staged[job_id] = extra_rows


class FakeEmptyLogger:
    def list_with_feedback(self):
        return []


def test_from_feedback_raises_when_no_feedback_exists():
    use_case = TriggerTrainingFromFeedback(FakeEmptyLogger(), FakeModelRepository(), FakeTrainingService())
    with pytest.raises(ValueError):
        use_case.execute()


def test_from_feedback_raises_when_segmentation_artifacts_missing():
    prop = Property(
        bedrooms=3, bathrooms=2.0, sqft_living=1800, sqft_lot=5000, floors=1.0, waterfront=0,
        view=0, condition=3, grade=7, sqft_above=1800, sqft_basement=0, yr_built=1990,
        yr_renovated=0, zipcode=98101, lat=47.6, long=-122.3, sqft_living15=1800, sqft_lot15=5000,
    )

    class FakePrediction:
        property = prop
        actual_price = 400000.0

    class FakeLoggerWithFeedback:
        def list_with_feedback(self):
            return [FakePrediction()]

    use_case = TriggerTrainingFromFeedback(FakeLoggerWithFeedback(), FakeModelRepository(), FakeTrainingService())
    with pytest.raises(ValueError, match="segmenta"):
        use_case.execute()


def test_from_upload_raises_on_empty_dataframe():
    use_case = TriggerTrainingFromUpload(FakeModelRepository(), FakeTrainingService())
    with pytest.raises(ValueError):
        use_case.execute(pd.DataFrame(), np.array([]))


def test_from_upload_raises_on_invalid_property_row():
    bad_row = pd.DataFrame([{
        "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800, "sqft_lot": 5000, "floors": 1.0,
        "waterfront": 0, "view": 0, "condition": 3, "grade": 99, "sqft_above": 1800,
        "sqft_basement": 0, "yr_built": 1990, "yr_renovated": 0, "zipcode": 98101, "lat": 47.6,
        "long": -122.3, "sqft_living15": 1800, "sqft_lot15": 5000,
    }])
    use_case = TriggerTrainingFromUpload(FakeModelRepository(), FakeTrainingService())
    with pytest.raises(ValueError):
        use_case.execute(bad_row, np.array([400000.0]))
