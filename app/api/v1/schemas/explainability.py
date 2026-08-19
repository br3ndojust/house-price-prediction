from __future__ import annotations

from pydantic import BaseModel


class FeatureImportanceOut(BaseModel):
    feature: str
    importance: float
    rank: int
    layer: str | None
    hypothesis: str | None
    used_by_active_model: bool = True


class FeatureContributionOut(BaseModel):
    feature: str
    value: float
    shap_value_log: float
    pct_of_total_abs_contribution: float
    approx_dollar_contribution: float


class PredictionExplanationOut(BaseModel):
    base_value_log: float
    predicted_value_log: float
    predicted_price_dollar: float
    contributions: list[FeatureContributionOut]
    approx_dollar_note: str
