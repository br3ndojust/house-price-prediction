from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionConfidenceOut(BaseModel):
    score: float = Field(..., description="Confidence Score 0-100 (100 = melhor)")
    category: str = Field(..., description="Alta confiança / Confiança moderada / Baixa confiança / Muito baixa confiança / revisão")
    expected_ape: float = Field(..., description="Erro percentual absoluto esperado (mediana do segmento em TEST)")
    expected_abs_error: float = Field(..., description="Erro absoluto esperado em dólares (mediana do segmento em TEST)")
    feature_space_distance: float = Field(..., description="Distância (z-score) ao vizinho mais próximo em TRAIN — reusa fase 04")
    coverage_bucket: str = Field(..., description="HIGH/MEDIUM/LOW — tercis da distância em TEST")
    segment_n: int = Field(..., description="Tamanho da amostra em TEST usada para estimar o erro esperado deste segmento")
    low_sample: bool = Field(..., description="True se nem o backoff mais grosseiro atingiu a amostra mínima (P4)")
    hard_segment: bool = Field(..., description="waterfront=1 ou grade>=10 — segmento com erro 2-4x maior (fase 11/13)")
    segment_level_used: str = Field(..., description="level0 (mais fino) / level1 / level2 / global — qual nível de backoff foi usado")


class ConfidenceCalibrationSummaryOut(BaseModel):
    ready: bool
    methodology: str | None = None
    category_thresholds: dict | None = None
    category_labels_high_to_low: list[str] | None = None
    ape_breakpoints_from_test: dict | None = None
    min_segment_n: int | None = None
    coverage_bucket_edges: list[float] | None = None
