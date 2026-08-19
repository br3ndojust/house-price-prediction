"""Snapshot persistido (importância + correlação com `price` + quartis) por versão de modelo — grava
uma vez, na criação do `ModelVersion` (seed do modelo original ou job de treino via API), sobre o
próprio `feature_cols` daquele modelo (nunca o conjunto oficial completo). Histórico consultável na
tela Importância de Features do portal, inclusive pra comparar modelos entre si, mesmo que o `.pkl`
do candidato seja removido depois (P5: cálculo centralizado em `src/evaluation/feature_statistics.py`,
nunca duplicado aqui)."""
from __future__ import annotations

import logging

import joblib
import pandas as pd

from src.evaluation.feature_statistics import compute_feature_statistics, compute_gain_importance

from app.domain.entities.feature_snapshot import FeatureSnapshot
from app.infrastructure.explainability.shap_explainer import load_feature_registry

logger = logging.getLogger("app.feature_snapshot")


class FeatureSnapshotService:
    def __init__(self, repository):
        self._repo = repository
        self._feature_registry = load_feature_registry()

    def build_and_persist(self, version: str, model, feature_cols: list[str],
                           train_df: pd.DataFrame) -> None:
        if self._repo.get(version) is not None:
            return  # idempotente — nunca recalcula/sobrescreve um snapshot já gravado

        raw_importance = compute_gain_importance(model, feature_cols)
        total = sum(raw_importance.values()) or 1.0
        correlations = compute_feature_statistics(train_df, feature_cols, price_col="price")
        ranked = sorted(raw_importance.items(), key=lambda kv: kv[1], reverse=True)

        reg = self._feature_registry
        stats: dict[str, dict] = {}
        for rank, (feature, value) in enumerate(ranked, start=1):
            row = reg[reg["feature"] == feature]
            corr = correlations.get(feature, {})
            stats[feature] = {
                "importance": round(value / total, 6),
                "rank": rank,
                "correlation_with_price": corr.get("correlation_with_price"),
                "min": corr.get("min"), "q1": corr.get("q1"), "median": corr.get("median"),
                "q3": corr.get("q3"), "max": corr.get("max"),
                "layer": row["layer"].iloc[0] if len(row) else None,
                "hypothesis": row["hypothesis"].iloc[0] if len(row) else None,
            }

        self._repo.add(FeatureSnapshot(version=version, stats=stats, n_train_rows=len(train_df)))

    def build_and_persist_from_artifact(self, version: str, artifact_path: str,
                                         features_parquet: str) -> None:
        """Carrega o `.pkl` do modelo + o TRAIN daquela execução — nunca lança, um snapshot é
        bookkeeping best-effort, nunca pode derrubar um job de treino ou o startup da API."""
        try:
            bundle = joblib.load(artifact_path)
            df = pd.read_parquet(features_parquet)
            train = df[df["split"] == "train"].reset_index(drop=True)
            self.build_and_persist(version, bundle["model"], bundle["feature_cols"], train)
        except Exception:
            logger.exception(f"falha ao gravar feature snapshot da versão {version} — seguindo sem ele")
