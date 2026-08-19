"""Adapter do port ModelRepository — carrega artifacts/*.pkl + production_contract.yaml.

Reusa src/pipelines/inference.py::predict_price (mesma função da demo de inferência da fase 14, P5:
paridade treino/produção por reuso de código). Suporta `reload()` sem restart — promoção (P6) troca o
ponteiro para o artefato ativo (`artifacts/candidates/model_candidate_<job_id>.pkl`), nunca sobrescreve
`artifacts/model_final.pkl`.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data.merge import merge_demographics
from src.features.comparable import apply_comps_knn_price
from src.features.neighborhood import add_dist_to_seattle_center
from src.features.raw import add_property_age
from src.features.relative_position import apply_local_grade_percentile
from src.segmentation.price_bands import SEMANTIC_LABELS, apply_price_quartile
from src.segmentation.property_clusters import PHYSICAL_PROFILE_COLUMNS, apply_property_cluster

from app.core.config import settings


class ArtifactModelRepository:
    def __init__(self, artifacts_dir: Path | None = None, data_dir: Path | None = None):
        self._artifacts_dir = artifacts_dir or settings.artifacts_dir
        self._data_dir = data_dir or settings.data_dir
        self._lock = threading.RLock()

        self._model_bundle: dict | None = None
        self._spatial_index = None
        self._demographics: pd.DataFrame | None = None
        self._price_breakpoints: np.ndarray | None = None
        self._cluster_scaler = None
        self._cluster_kmeans = None
        self._contract: dict = {}
        self._active_version: str = "unloaded"
        self._active_artifact_path: str = ""

        self.reload()

    # ---- carregamento / promoção ----------------------------------------------------------
    def reload(self, artifact_path: str | None = None) -> None:
        with self._lock:
            model_path = Path(artifact_path) if artifact_path else self._artifacts_dir / "model_final.pkl"
            self._model_bundle = joblib.load(model_path)
            self._active_artifact_path = str(model_path)
            self._active_version = model_path.stem if artifact_path else "model_final"

            spatial = joblib.load(self._artifacts_dir / "spatial_index.pkl")
            self._spatial_index = spatial["index"] if isinstance(spatial, dict) else spatial

            self._demographics = pd.read_csv(self._data_dir / "raw" / "zipcode_demographics.csv")

            quartiles_path = self._data_dir / "trusted" / "price_quartiles.json"
            if quartiles_path.exists():
                self._price_breakpoints = np.array(json.loads(quartiles_path.read_text())["breakpoints"])

            cluster_path = self._artifacts_dir / "property_cluster_model.pkl"
            if cluster_path.exists():
                cluster_bundle = joblib.load(cluster_path)
                self._cluster_scaler = cluster_bundle["scaler"]
                self._cluster_kmeans = cluster_bundle["kmeans"]

            contract_path = self._artifacts_dir / "production_contract.yaml"
            if contract_path.exists():
                self._contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
            self._contract = {
                **self._contract,
                "active_model_version": self._active_version,
                "active_artifact_path": self._active_artifact_path,
                "reloaded_at": datetime.utcnow().isoformat(),
            }

    def is_ready(self) -> bool:
        return self._model_bundle is not None

    def active_model_bundle(self) -> dict:
        return self._model_bundle

    def active_contract(self) -> dict:
        return self._contract

    def active_version(self) -> str:
        return self._active_version

    # ---- feature building (paridade treino/produção, P5) -----------------------------------
    def _build_feature_frame(self, raw_df: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
        as_of = as_of_date or datetime.now().date()
        df = raw_df.copy()
        df["date"] = as_of.strftime("%Y%m%dT000000")

        df = merge_demographics(df, self._demographics)
        df = add_property_age(df)
        df = add_dist_to_seattle_center(df)
        df["comps_knn_price"] = apply_comps_knn_price(df, self._spatial_index)
        df["local_grade_percentile"] = apply_local_grade_percentile(df, self._spatial_index)
        return df

    def build_model_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return self._build_feature_frame(raw_df)

    def predict_from_features(self, features_df: pd.DataFrame) -> pd.Series:
        """Prevê a partir de um `build_model_features(...)` já calculado — evita reconstruir a
        engenharia de features quando o chamador já precisa do dataframe de features por outro
        motivo (banda/cluster/confiança), P5."""
        X = features_df[self._model_bundle["feature_cols"]]
        pred_log = self._model_bundle["model"].predict(X)
        return pd.Series(np.expm1(pred_log), index=features_df.index)

    def predict_raw(self, raw_df: pd.DataFrame) -> pd.Series:
        df = self._build_feature_frame(raw_df)
        return self.predict_from_features(df)

    def predict_price_band(self, price_log: pd.Series) -> list[str]:
        if self._price_breakpoints is None:
            return ["unknown"] * len(price_log)
        quartile = apply_price_quartile(price_log, self._price_breakpoints)
        return [SEMANTIC_LABELS[str(q)] for q in quartile]

    def predict_property_cluster(self, features_df: pd.DataFrame) -> list[int]:
        if self._cluster_scaler is None or self._cluster_kmeans is None:
            return [-1] * len(features_df)
        labels = apply_property_cluster(
            features_df[PHYSICAL_PROFILE_COLUMNS], self._cluster_scaler, self._cluster_kmeans
        )
        return [int(c) for c in labels]

    # ---- materiais de segmentação/features para construir dado novo de treino (fase 16) --------
    def training_materials(self) -> dict:
        """Expõe os artefatos já carregados (demografia, índice espacial, cortes de banda/cluster)
        para `src/pipelines/training_data.py::build_labeled_rows` — reuso, não recarrega nada (P5)."""
        return {
            "demographics": self._demographics,
            "spatial_index": self._spatial_index,
            "price_breakpoints": self._price_breakpoints,
            "cluster_scaler": self._cluster_scaler,
            "cluster_kmeans": self._cluster_kmeans,
        }
