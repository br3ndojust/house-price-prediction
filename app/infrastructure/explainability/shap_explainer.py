"""Adapter do port ExplainabilityService — importância global (gain-based, nativo do XGBoost) +
local via `shap.TreeExplainer` (nova dependência, ver docs/02_execution_plan_api.md).

`shap.TreeExplainer` é cacheado no startup/reload (never recriado por request) — reconstruído só
quando `refresh()` é chamado após uma promoção de modelo (P6: ponteiro do modelo ativo pode trocar).

Nota de honestidade técnica: o SHAP local é calculado no espaço `price_log` (o alvo do modelo); a
conversão para dólar é aproximada (ver `PredictionExplanation.approx_dollar_note`).
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pandas as pd
import shap

from src.evaluation.feature_statistics import (
    compute_feature_statistics,
    compute_gain_importance,
    sample_feature_price_pairs,
)

from app.core.config import settings
from app.domain.entities.feature_importance import FeatureImportance
from app.domain.entities.prediction_explanation import FeatureContribution, PredictionExplanation
from app.domain.ports.model_repository import ModelRepository


def load_feature_registry() -> pd.DataFrame:
    """Camada/hipótese de cada feature (`data/processing/feature_registry.csv`) — única fonte (P5),
    reusada pelo ranking do modelo ativo e pelo `FeatureSnapshotService` (snapshot por modelo)."""
    path = settings.data_dir / "processing" / "feature_registry.csv"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "layer", "hypothesis"])
    return pd.read_csv(path, encoding="utf-8")[["feature", "layer", "hypothesis"]]


def load_official_feature_cols() -> list[str]:
    path = settings.data_dir / "trusted" / "feature_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))["model_features"]


class ShapExplainabilityService:
    def __init__(self, model_repository: ModelRepository):
        self._models = model_repository
        self._lock = threading.RLock()
        self._explainer = None
        self._explainer_version: str | None = None
        self._feature_registry = load_feature_registry()
        self._train_df: pd.DataFrame | None = None

    def _train_reference(self) -> pd.DataFrame:
        if self._train_df is None:
            df = pd.read_parquet(settings.data_dir / "trusted" / "features_contextual.parquet")
            self._train_df = df[df["split"] == "train"].reset_index(drop=True)
        return self._train_df

    def _get_explainer(self):
        with self._lock:
            active_version = self._models.active_version()
            if self._explainer is None or self._explainer_version != active_version:
                model = self._models.active_model_bundle()["model"]
                if not hasattr(model, "get_booster"):
                    raise ValueError(
                        "shap.TreeExplainer só é suportado para modelos baseados em árvore (XGBoost). "
                        f"Modelo ativo atual: {type(model).__name__}."
                    )
                self._explainer = shap.TreeExplainer(model)
                self._explainer_version = active_version
            return self._explainer

    def global_importance(self) -> list[FeatureImportance]:
        """Importância calculada sobre o modelo ativo, mas reportada para o conjunto OFICIAL completo
        de features (`data/trusted/feature_metadata.json`, travado desde o primeiro treino) — não só
        as do modelo ativo. Um candidato pode ter sido treinado com um subconjunto (retraining com
        features desmarcadas na tela Treino); isso não pode fazer a feature "sumir" das opções de
        seleção para o próximo retraining (`used_by_active_model=False` sinaliza esse caso, com
        importância 0.0 por não ter sido usada por este modelo)."""
        bundle = self._models.active_model_bundle()
        model = bundle["model"]
        active_feature_cols = set(bundle["feature_cols"])

        raw = compute_gain_importance(model, bundle["feature_cols"])
        total = sum(raw.values()) or 1.0
        all_features = load_official_feature_cols()
        combined = {f: raw.get(f, 0.0) for f in all_features}
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)

        reg = self._feature_registry
        out = []
        for rank, (feature, value) in enumerate(ranked, start=1):
            row = reg[reg["feature"] == feature]
            layer = row["layer"].iloc[0] if len(row) else None
            hypothesis = row["hypothesis"].iloc[0] if len(row) else None
            out.append(FeatureImportance(
                feature=feature, importance=round(value / total, 6), rank=rank,
                layer=layer, hypothesis=hypothesis,
                used_by_active_model=feature in active_feature_cols,
            ))
        return out

    def explain_local(self, model_features_row: pd.DataFrame, predicted_price: float) -> PredictionExplanation:
        bundle = self._models.active_model_bundle()
        feature_cols = bundle["feature_cols"]
        explainer = self._get_explainer()

        X = model_features_row[feature_cols]
        shap_values = explainer.shap_values(X)[0]
        base_value_log = float(np.asarray(explainer.expected_value).ravel()[0])
        predicted_value_log = float(base_value_log + shap_values.sum())

        total_abs = float(np.sum(np.abs(shap_values))) or 1.0
        contributions = []
        for feature, shap_value in zip(feature_cols, shap_values):
            shap_value = float(shap_value)
            without_feature_log = predicted_value_log - shap_value
            approx_dollar = float(np.expm1(predicted_value_log) - np.expm1(without_feature_log))
            contributions.append(FeatureContribution(
                feature=feature,
                value=float(X[feature].iloc[0]),
                shap_value_log=shap_value,
                pct_of_total_abs_contribution=round(abs(shap_value) / total_abs * 100, 2),
                approx_dollar_contribution=round(approx_dollar, 2),
            ))
        contributions.sort(key=lambda c: abs(c.shap_value_log), reverse=True)

        return PredictionExplanation(
            base_value_log=base_value_log,
            predicted_value_log=predicted_value_log,
            predicted_price_dollar=predicted_price,
            contributions=contributions,
        )

    def refresh(self) -> None:
        with self._lock:
            self._explainer = None
            self._explainer_version = None

    def feature_statistics(self) -> dict[str, dict]:
        train = self._train_reference()
        official = load_official_feature_cols()
        return compute_feature_statistics(train, official, price_col="price")

    def feature_sample(self, sample_size: int = 400) -> list[dict]:
        train = self._train_reference()
        official = load_official_feature_cols()
        sample = sample_feature_price_pairs(train, official, price_col="price", sample_size=sample_size)
        return sample.to_dict(orient="records")
