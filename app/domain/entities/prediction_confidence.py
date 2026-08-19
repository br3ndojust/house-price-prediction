"""PredictionConfidence — Matriz/Score de Confiança (0-100) calibrado em TEST, verificado em VAL uma
única vez. Ver `src/evaluation/confidence.py` (metodologia) e `docs/09_confidence_matrix.md`
(explicação end-to-end + números desta execução, incluindo a limitação honesta encontrada na
verificação em VAL)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionConfidence:
    score: float
    category: str
    expected_ape: float
    expected_abs_error: float
    feature_space_distance: float
    coverage_bucket: str
    segment_n: int
    low_sample: bool
    hard_segment: bool
    segment_level_used: str
