from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class FeatureSnapshot:
    """Importância (gain-based) + correlação com `price` + quartis, por feature, de um modelo
    registrado específico — calculado sobre o próprio `feature_cols` daquele modelo (nunca o conjunto
    oficial completo, P4: honestidade sobre o que aquele modelo de fato usou). Gravado uma vez, na
    criação do `ModelVersion` (fase 16/API) — histórico consultável mesmo se o `.pkl` do candidato for
    removido depois."""

    version: str
    stats: dict[str, dict]  # feature -> {importance, rank, correlation_with_price, min, q1, median, q3, max, layer, hypothesis}
    n_train_rows: int
    created_at: datetime = field(default_factory=datetime.utcnow)
