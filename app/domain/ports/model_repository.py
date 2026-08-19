"""Port — acesso ao modelo ativo e artefatos derivados (P6: promoção troca ponteiro, nunca sobrescreve)."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class ModelRepository(Protocol):
    def predict_raw(self, raw_df: pd.DataFrame) -> "pd.Series":
        """Retorna preço em dólar por linha, reusando src/pipelines/inference.py::predict_price."""
        ...

    def predict_from_features(self, features_df: pd.DataFrame) -> "pd.Series":
        """Prevê a partir de um `build_model_features(...)` já calculado — evita reconstruir a
        engenharia de features quando o chamador já precisa do dataframe de features por outro
        motivo (banda/cluster/confiança)."""
        ...

    def predict_price_band(self, price_log: "pd.Series") -> list[str]:
        ...

    def predict_property_cluster(self, features_df: pd.DataFrame) -> list[int]:
        ...

    def build_model_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Constrói o dataframe no espaço de 20 features oficiais (paridade treino/produção, P5)."""
        ...

    def active_model_bundle(self) -> dict:
        ...

    def active_contract(self) -> dict:
        ...

    def active_version(self) -> str:
        ...

    def is_ready(self) -> bool:
        ...

    def reload(self, artifact_path: str | None = None) -> None:
        """Recarrega o modelo ativo sem restart do processo (P6)."""
        ...

    def training_materials(self) -> dict:
        """Demografia, índice espacial, cortes de banda/cluster — para construir dado novo de treino
        (fase 16, `src/pipelines/training_data.py`)."""
        ...
