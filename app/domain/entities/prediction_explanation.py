"""PredictionExplanation — contribuição local (SHAP) por feature para uma predição específica.

Nota de honestidade técnica (P4, ver docs/02_execution_plan_api.md): os valores de SHAP são calculados
no espaço do alvo do modelo (`price_log`). A contribuição relativa (ranking + %) é exata nesse espaço;
a conversão aproximada para dólar via `expm1` no ponto de referência NÃO é perfeitamente aditiva
(não-linearidade do `expm1`) — o campo `approx_dollar_note` explicita isso em toda resposta.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    feature: str
    value: float
    shap_value_log: float
    pct_of_total_abs_contribution: float
    approx_dollar_contribution: float


@dataclass(frozen=True, slots=True)
class PredictionExplanation:
    base_value_log: float
    predicted_value_log: float
    predicted_price_dollar: float
    contributions: list[FeatureContribution]
    approx_dollar_note: str = (
        "A conversão para dólar é aproximada (expm1 no ponto de referência) e não é perfeitamente "
        "aditiva devido à não-linearidade do expm1; use o ranking e o percentual (espaço price_log) "
        "como a leitura primária de importância local."
    )
