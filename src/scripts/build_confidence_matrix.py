"""Matriz/Score de Confiança — calibra em TEST, verifica uma única vez em VAL.

Não é uma fase de decisão de modelo/feature (P4) — não muda `model_final.pkl` nem o conjunto de
features. Produz `artifacts/confidence_calibration.json` (artefato congelado, consumido pela API para
anexar confiança a cada predição nova, ver `app/infrastructure/confidence/`) e
`reports/confidence_matrix_report.json` (relatório humano: calibração em TEST + verificação em VAL).

Ver `src/evaluation/confidence.py` para a metodologia completa (docstring do módulo) e
`docs/09_confidence_matrix.md` para a explicação end-to-end com os números desta execução.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.confidence import (
    apply_coverage_bucket,
    build_confidence_matrix_2x2,
    build_ecdf_reference,
    build_segment_lookup,
    calibrated_ape_breakpoints,
    categorize,
    compute_feature_space_distance,
    confidence_score,
    derive_category_thresholds,
    expected_error_for_row,
    fit_coverage_bucket_edges,
    fit_isotonic_calibration,
    isotonic_curve_points,
    segment_keys,
)
from src.evaluation.error_matrix import LOW_SAMPLE_THRESHOLD


def _predict(bundle_path: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    bundle = joblib.load(bundle_path)
    feature_cols = bundle["feature_cols"]
    target = bundle["target"]
    y_true = np.expm1(df[target].to_numpy())
    y_pred = np.expm1(bundle["model"].predict(df[feature_cols]))
    return y_true, y_pred, feature_cols


def _with_error_cols(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["y_true"], df["y_pred"] = y_true, y_pred
    df["abs_error"] = np.abs(y_true - y_pred)
    df["ape"] = df["abs_error"] / np.maximum(y_true, 1)
    return df


def run_confidence_matrix_pipeline(
    features_parquet: str = "data/trusted/features_contextual.parquet",
    candidate_artifact_path: str = "artifacts/model_candidate.pkl",
    final_artifact_path: str = "artifacts/model_final.pkl",
    calibration_artifact_path: str = "artifacts/confidence_calibration.json",
    report_path: str = "reports/confidence_matrix_report.json",
) -> dict:
    """Matriz/Score de Confiança, refatorada em função chamável (P5) — reusada tanto por `main()`
    (CLI) quanto pelo notebook `15_confidence_matrix.ipynb`.

    Não é uma fase de decisão de modelo/feature (P4) — não muda `model_final.pkl` nem o conjunto de
    features. Produz `artifacts/confidence_calibration.json` (artefato congelado, consumido pela API
    para anexar confiança a cada predição nova, ver `app/infrastructure/confidence/`) e
    `reports/confidence_matrix_report.json` (relatório humano: calibração em TEST + verificação em
    VAL). Ver `src/evaluation/confidence.py` para a metodologia completa (docstring do módulo) e
    `docs/09_confidence_matrix.md` para a explicação end-to-end com os números desta execução."""
    df = pd.read_parquet(features_parquet)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    # ---- CALIBRAÇÃO EM TEST (model_candidate: nunca viu TEST) ----------------------------------
    cal_true, cal_pred, feature_cols = _predict(candidate_artifact_path, test_df)
    test_df = _with_error_cols(test_df, cal_true, cal_pred)

    test_distance = compute_feature_space_distance(train_df, test_df, feature_cols)
    coverage_edges = fit_coverage_bucket_edges(test_distance)
    test_df["coverage_bucket"] = apply_coverage_bucket(test_distance, coverage_edges)
    test_df["feature_space_distance"] = test_distance

    keys = segment_keys(test_df, test_df["coverage_bucket"].to_numpy())
    test_df = pd.concat([test_df.reset_index(drop=True), keys.reset_index(drop=True)], axis=1)

    lookup = build_segment_lookup(test_df, min_n=LOW_SAMPLE_THRESHOLD)
    expected = pd.DataFrame([
        expected_error_for_row(r.level0, r.level1, r.level2, lookup)
        for r in test_df.itertuples()
    ])
    test_df["expected_ape"] = expected["median_ape"].to_numpy()
    test_df["expected_abs_error"] = expected["median_abs_error"].to_numpy()
    test_df["segment_n"] = expected["n"].to_numpy()
    test_df["segment_level_used"] = expected["level_used"].to_numpy()
    test_df["segment_low_sample"] = expected["low_sample"].to_numpy()

    ecdf_reference = build_ecdf_reference(test_df["expected_ape"].to_numpy())
    test_df["confidence_score"] = confidence_score(test_df["expected_ape"].to_numpy(), ecdf_reference)

    isotonic_model = fit_isotonic_calibration(test_df["confidence_score"].to_numpy(), test_df["ape"].to_numpy())
    curve_points = isotonic_curve_points(isotonic_model)

    ape_breakpoints = calibrated_ape_breakpoints(isotonic_model, test_df["confidence_score"].to_numpy())
    thresholds = derive_category_thresholds(curve_points, ape_breakpoints)
    test_df["confidence_category"] = [categorize(s, thresholds) for s in test_df["confidence_score"]]

    distance_median_test = float(np.median(test_distance))
    expected_ape_median_test = float(np.median(test_df["expected_ape"]))
    matrix_test = build_confidence_matrix_2x2(
        test_distance, test_df["expected_ape"].to_numpy(), test_df["ape"].to_numpy(),
        test_df["abs_error"].to_numpy(), distance_median_test, expected_ape_median_test,
    )

    calibration_by_category_test = (
        test_df.groupby("confidence_category", observed=True)
        .agg(n=("ape", "size"), median_ape=("ape", "median"), median_abs_error=("abs_error", "median"))
        .reindex(["Alta confiança", "Confiança moderada", "Baixa confiança",
                  "Muito baixa confiança / revisão"])
        .reset_index()
    )

    # ---- VERIFICAÇÃO EM VAL (model_final: refit TRAIN+TEST; calibração acima fica congelada) ----
    val_true, val_pred, _ = _predict(final_artifact_path, val_df)
    val_df = _with_error_cols(val_df, val_true, val_pred)

    val_distance = compute_feature_space_distance(train_df, val_df, feature_cols)
    val_df["coverage_bucket"] = apply_coverage_bucket(val_distance, coverage_edges)  # edges congelados
    val_df["feature_space_distance"] = val_distance

    val_keys = segment_keys(val_df, val_df["coverage_bucket"].to_numpy())
    val_df = pd.concat([val_df.reset_index(drop=True), val_keys.reset_index(drop=True)], axis=1)

    val_expected = pd.DataFrame([
        expected_error_for_row(r.level0, r.level1, r.level2, lookup)  # lookup congelado (TEST)
        for r in val_df.itertuples()
    ])
    val_df["expected_ape"] = val_expected["median_ape"].to_numpy()
    val_df["segment_n"] = val_expected["n"].to_numpy()
    val_df["segment_low_sample"] = val_expected["low_sample"].to_numpy()

    val_df["confidence_score"] = confidence_score(val_df["expected_ape"].to_numpy(), ecdf_reference)  # ecdf congelado
    val_df["confidence_category"] = [categorize(s, thresholds) for s in val_df["confidence_score"]]

    validation_by_category_val = (
        val_df.groupby("confidence_category", observed=True)
        .agg(n=("ape", "size"), observed_median_ape=("ape", "median"),
             observed_mae=("abs_error", "median"))
        .reindex(["Alta confiança", "Confiança moderada", "Baixa confiança",
                  "Muito baixa confiança / revisão"])
        .reset_index()
    )
    validation_by_category_val["promised_median_ape_max_from_test"] = [
        ape_breakpoints["p25"], ape_breakpoints["p50"], ape_breakpoints["p75"], None
    ]
    validation_by_category_val["promise_held"] = [
        bool(row.observed_median_ape <= row.promised_median_ape_max_from_test)
        if row.promised_median_ape_max_from_test is not None and pd.notna(row.observed_median_ape)
        else None
        for row in validation_by_category_val.itertuples()
    ]

    # Checagem honesta (P4): as 4 categorias só entregam o que prometem se o erro observado em VAL
    # for realmente crescente de "Alta confiança" para "Muito baixa" — checa isso explicitamente em
    # vez de só reportar "dentro do teto prometido" (um teto largo pode "passar" mesmo fora de ordem).
    observed_series = validation_by_category_val.set_index("confidence_category")["observed_median_ape"]
    ordered_categories = ["Alta confiança", "Confiança moderada", "Baixa confiança",
                           "Muito baixa confiança / revisão"]
    observed_ordered = [float(observed_series[c]) for c in ordered_categories if pd.notna(observed_series[c])]
    is_monotonic_on_val = bool(all(a <= b for a, b in zip(observed_ordered, observed_ordered[1:])))

    matrix_val = build_confidence_matrix_2x2(
        val_distance, val_df["expected_ape"].to_numpy(), val_df["ape"].to_numpy(),
        val_df["abs_error"].to_numpy(), distance_median_test, expected_ape_median_test,
        # cortes de distance/expected_ape CONGELADOS de TEST, nunca recalculados em VAL
    )

    # ---- artefato congelado consumido pela API --------------------------------------------------
    calibration_artifact = {
        "methodology": (
            "TEST calibra (model_candidate, nunca viu TEST): tabela de erro esperado por segmento "
            "(price_band x property_cluster x cobertura no espaco de features x waterfront/grade>=10, "
            "com backoff por amostra pequena), Confidence Score = rank empirico do erro esperado do "
            "segmento vs. distribuicao de TEST, curva score->erro calibrada por regressao isotonica, "
            "cortes de categoria = onde a curva cruza os quartis REAIS de APE em TEST (P25/P50/P75). "
            "VAL verifica uma unica vez, sem recalcular nada (model_final, calibracao 100% congelada)."
        ),
        "feature_cols": feature_cols,
        "coverage_bucket_edges": coverage_edges,
        "min_segment_n": LOW_SAMPLE_THRESHOLD,
        "segment_lookup": lookup,
        "ecdf_reference": ecdf_reference.tolist(),
        "isotonic_curve": curve_points,
        "ape_breakpoints_from_test": ape_breakpoints,
        "category_thresholds": thresholds,
        "category_labels_high_to_low": [
            "Alta confiança", "Confiança moderada", "Baixa confiança", "Muito baixa confiança / revisão",
        ],
        "matrix_2x2_distance_median_from_test": distance_median_test,
        "matrix_2x2_expected_ape_median_from_test": expected_ape_median_test,
    }
    Path(calibration_artifact_path).parent.mkdir(parents=True, exist_ok=True)
    Path(calibration_artifact_path).write_text(
        json.dumps(calibration_artifact, indent=2, default=str), encoding="utf-8"
    )

    report = {
        "n_test": int(len(test_df)),
        "n_val": int(len(val_df)),
        "category_thresholds": thresholds,
        "ape_breakpoints_from_test": ape_breakpoints,
        "calibration_on_test": {
            "by_category": calibration_by_category_test.to_dict(orient="records"),
            "matrix_2x2": matrix_test,
        },
        "verification_on_val": {
            "by_category": validation_by_category_val.to_dict(orient="records"),
            "matrix_2x2": matrix_val,
            "is_monotonic_by_category": is_monotonic_on_val,
            "monotonicity_note": (
                "True = erro mediano observado em VAL cresce (ou empata) de 'Alta confianca' para "
                "'Muito baixa confianca / revisao', na mesma ordem prometida pela calibracao em TEST. "
                "False = a ORDEM das categorias nao se sustentou em VAL, mesmo que cada categoria "
                "tenha ficado dentro do teto prometido (promise_held) - ver docs/09_confidence_matrix.md "
                "secao 'Limitacao observada' para a leitura honesta deste resultado."
            ),
        },
        "segment_lookup_summary": {
            "n_level0_segments": len(lookup["level0"]),
            "n_level1_segments": len(lookup["level1"]),
            "n_level2_segments": len(lookup["level2"]),
        },
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> None:
    report = run_confidence_matrix_pipeline()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
