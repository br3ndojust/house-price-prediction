"""Análise pós-projeto: previsões do modelo final sobre `val`, erro por região/tipo de imóvel/
conjunto de features, e bandas de confiança empírica (conformal simples).

Não é uma fase de decisão (P4) — não muda nenhum artefato de modelo/feature. É leitura sobre
artefatos já travados (fases 09-14), para reporte visual. Metodologia da banda de confiança:

- Calibração: `model_candidate.pkl` (treinado só em TRAIN) aplicado em TEST (dado nunca visto por
  esse modelo) -> resíduos assinados (`y_true - y_pred`) por banda de preço geram quantis empíricos
  (5/10/25/50/75/90/95%).
- Validação: os quantis calibrados em TEST são aplicados às previsões de `model_final.pkl` (treinado
  em TRAIN+TEST) sobre `val` -> mede-se a taxa de cobertura real em `val` (dado nunca usado para
  calibrar o intervalo) para checar se o intervalo declarado é confiável (honesto, não circular).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.error_matrix import compute_metrics, error_matrix_by

CLUSTER_LABELS = {
    0: "Antigo / Compacto",
    1: "Moderno / Amplo",
    2: "Luxo físico (waterfront/view)",
}
BAND_ORDER = ["Entry", "Standard", "Premium", "Luxury"]


def predict(bundle_path: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    bundle = joblib.load(bundle_path)
    feature_cols = bundle["feature_cols"]
    target = bundle["target"]
    y_true = np.expm1(df[target].to_numpy())
    y_pred = np.expm1(bundle["model"].predict(df[feature_cols]))
    return y_true, y_pred, feature_cols


def quantile_bands(residual: np.ndarray) -> dict:
    qs = [5, 10, 25, 50, 75, 90, 95]
    return {f"q{q}": float(np.percentile(residual, q)) for q in qs}


def coverage(y_true: np.ndarray, y_pred: np.ndarray, q_lo: float, q_hi: float) -> float:
    lower = y_pred + q_lo
    upper = y_pred + q_hi
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def price_buckets(y_pred: np.ndarray, y_true: np.ndarray, n_buckets: int = 8) -> list[dict]:
    edges = np.quantile(y_pred, np.linspace(0, 1, n_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bucket_idx = np.digitize(y_pred, edges[1:-1])
    rows = []
    for b in range(n_buckets):
        mask = bucket_idx == b
        if mask.sum() == 0:
            continue
        err = y_true[mask] - y_pred[mask]
        abs_err = np.abs(err)
        ape = abs_err / np.maximum(y_true[mask], 1)
        lo_edge = edges[b] if np.isfinite(edges[b]) else float(np.min(y_pred[mask]))
        hi_edge = edges[b + 1] if np.isfinite(edges[b + 1]) else float(np.max(y_pred[mask]))
        rows.append({
            "bucket": b,
            "pred_range_low": float(lo_edge),
            "pred_range_high": float(hi_edge),
            "pred_median": float(np.median(y_pred[mask])),
            "n": int(mask.sum()),
            "median_abs_error": float(np.median(abs_err)),
            "p80_abs_error": float(np.percentile(abs_err, 80)),
            "p90_abs_error": float(np.percentile(abs_err, 90)),
            "median_ape": float(np.median(ape)),
            "mean_bias": float(np.mean(err)),
        })
    return rows


def run_val_prediction_analysis_pipeline(
    features_parquet: str = "data/trusted/features_contextual.parquet",
    candidate_artifact_path: str = "artifacts/model_candidate.pkl",
    final_artifact_path: str = "artifacts/model_final.pkl",
    ablation_report_path: str = "reports/ablation_fase09.json",
    output_report_path: str = "reports/prediction_confidence_analysis.json",
) -> dict:
    """Análise pós-projeto, refatorada em função chamável (P5) — reusada tanto por `main()` (CLI)
    quanto pelo notebook `13_final_refit_and_val_check.ipynb` (passo 2, após `finalize_model.py`).

    Não é uma fase de decisão (P4) — não muda nenhum artefato de modelo/feature. É leitura sobre
    artefatos já travados (fases 09-14), para reporte visual."""
    df = pd.read_parquet(features_parquet)
    test_df = df[df["split"] == "test"].copy()
    val_df = df[df["split"] == "val"].copy()

    # calibração: model_candidate (so viu TRAIN) aplicado em TEST
    cal_true, cal_pred, _ = predict(candidate_artifact_path, test_df)
    test_df["y_true"] = cal_true
    test_df["y_pred"] = cal_pred
    test_df["residual"] = test_df["y_true"] - test_df["y_pred"]

    band_quantiles = {}
    for band in BAND_ORDER:
        sub = test_df[test_df["price_band"] == band]
        band_quantiles[band] = quantile_bands(sub["residual"].to_numpy())
    global_quantiles = quantile_bands(test_df["residual"].to_numpy())

    # previsoes finais: model_final (TRAIN+TEST) aplicado em VAL, tocado uma vez (P1)
    val_true, val_pred, feature_cols = predict(final_artifact_path, val_df)
    val_df["y_true"] = val_true
    val_df["y_pred"] = val_pred
    val_df["abs_error"] = np.abs(val_df["y_true"] - val_df["y_pred"])
    val_df["pct_error"] = (val_df["y_pred"] - val_df["y_true"]) / val_df["y_true"]
    val_df["ape"] = np.abs(val_df["pct_error"])

    global_metrics = compute_metrics(val_true, val_pred)

    # cobertura do intervalo calibrado em TEST, checada em VAL (honesto: nao foi usado p/ calibrar)
    coverage_rows = []
    for band in BAND_ORDER:
        sub = val_df[val_df["price_band"] == band]
        if len(sub) == 0:
            continue
        bq = band_quantiles[band]
        cov80 = coverage(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), bq["q10"], bq["q90"])
        cov90 = coverage(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), bq["q5"], bq["q95"])
        coverage_rows.append({
            "price_band": band, "n": int(len(sub)),
            "interval80_low_offset": bq["q10"], "interval80_high_offset": bq["q90"],
            "interval80_target": 0.80, "interval80_observed": cov80,
            "interval90_low_offset": bq["q5"], "interval90_high_offset": bq["q95"],
            "interval90_target": 0.90, "interval90_observed": cov90,
        })
    cov80_all = coverage(val_df["y_true"].to_numpy(), val_df["y_pred"].to_numpy(),
                          global_quantiles["q10"], global_quantiles["q90"])
    cov90_all = coverage(val_df["y_true"].to_numpy(), val_df["y_pred"].to_numpy(),
                          global_quantiles["q5"], global_quantiles["q95"])

    by_band = error_matrix_by(val_df, "price_band", "y_true", "y_pred")
    by_cluster = error_matrix_by(val_df, "property_cluster", "y_true", "y_pred")
    by_zip = error_matrix_by(val_df, "zipcode", "y_true", "y_pred")
    zip_coords = val_df.groupby("zipcode")[["lat", "long"]].mean().reset_index()
    by_zip = by_zip.merge(zip_coords, on="zipcode", how="left")

    feature_set_report = json.loads(Path(ablation_report_path).read_text())["ablation"]

    buckets = price_buckets(val_df["y_pred"].to_numpy(), val_df["y_true"].to_numpy(), n_buckets=8)

    scatter_sample = val_df[["y_true", "y_pred", "price_band", "zipcode"]].copy()
    scatter_sample = scatter_sample.rename(columns={"y_true": "actual", "y_pred": "predicted"})

    out = {
        "n_val": int(len(val_df)),
        "model": {
            "algorithm": "xgboost", "hyperparameters": {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05},
            "trained_on": "train+test (refit final, fase 13)", "artifact": "artifacts/model_final.pkl",
            "n_features": len(feature_cols),
        },
        "global_metrics_val": global_metrics,
        "by_price_band_val": by_band.to_dict(orient="records"),
        "by_property_cluster_val": [
            {**row, "label": CLUSTER_LABELS.get(int(row["property_cluster"]), str(row["property_cluster"]))}
            for row in by_cluster.to_dict(orient="records")
        ],
        "by_zipcode_val": by_zip.to_dict(orient="records"),
        "feature_set_comparison": feature_set_report,
        "confidence_calibration": {
            "methodology": "quantis de residuo (model_candidate em TEST, nao visto no treino) "
                            "aplicados como banda em torno da previsao de model_final em VAL "
                            "(nao usado para calibrar) - checagem de cobertura honesta.",
            "global_quantiles_from_test": global_quantiles,
            "band_quantiles_from_test": band_quantiles,
            "coverage_check_on_val": {
                "overall": {"interval80_target": 0.80, "interval80_observed": cov80_all,
                             "interval90_target": 0.90, "interval90_observed": cov90_all},
                "by_band": coverage_rows,
            },
        },
        "price_buckets_val": buckets,
        "scatter_val": scatter_sample.to_dict(orient="records"),
    }
    Path(output_report_path).write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    out = run_val_prediction_analysis_pipeline()
    print(json.dumps({k: v for k, v in out.items() if k != "scatter_val"}, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
