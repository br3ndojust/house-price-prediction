"""Matriz/Score de Confiança — pós-projeto (não é fase de decisão de modelo/feature, P4).

Metodologia (mesma disciplina de `src/scripts/analyze_val_predictions.py` e `src/models/train.py`):
**TEST calibra, VAL verifica uma única vez.**

    TEST (model_candidate, só viu TRAIN)
      │
      ├── mede erro real por segmento (price_band × property_cluster × cobertura no espaço de
      │    features × waterfront/grade≥10) — nunca um único MAE global (P3)
      ├── constrói o Confidence Score (0-100) como rank empírico do erro esperado do segmento
      │    contra a distribuição de erro esperado de todo o TEST
      ├── calibra a curva score → erro esperado via regressão isotônica (monótona, sem pesos
      │    escolhidos à mão) e deriva os cortes de categoria dos quartis REAIS de erro em TEST
      │    (P25/P50/P75), não de um número "que parece bom"
      └── tudo isso fica congelado em `artifacts/confidence_calibration.json`
              │
              ▼
             VAL (model_final, tocado nesta checagem)
              │
              └── aplica a calibração congelada (sem refazer nada) e mede: a confiança prometida
                   em TEST realmente se sustenta em dado nunca usado para calibrar?

Segmentos com poucas amostras em TEST (`n < LOW_SAMPLE_THRESHOLD`, mesmo limiar de
`src/evaluation/error_matrix.py`) fazem *backoff* para uma chave mais grosseira — nunca reporta erro
esperado a partir de uma amostra estatisticamente frágil sem avisar (P4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.evaluation.error_matrix import LOW_SAMPLE_THRESHOLD
from src.evaluation.geographic_coverage import feature_space_coverage

COVERAGE_LABELS = ["HIGH", "MEDIUM", "LOW"]  # HIGH = perto de TRAIN no espaço de features (melhor)
GLOBAL_KEY = "GLOBAL"


# ---------------------------------------------------------------------------------------------
# Cobertura no espaço de features (reusa src/evaluation/geographic_coverage.py, fase 04 — P5)
# ---------------------------------------------------------------------------------------------
def compute_feature_space_distance(train_df: pd.DataFrame, other_df: pd.DataFrame,
                                    feature_cols: list[str]) -> np.ndarray:
    coverage = feature_space_coverage(train_df, other_df.reset_index(drop=True), feature_cols)
    return coverage["nearest_train_neighbor_distance"].to_numpy()


def fit_coverage_bucket_edges(distances: np.ndarray) -> list[float]:
    """Tercis da distância em TEST — mesmo estilo de balde por quantil da fase 04, aplicado à
    distância no espaço de features em vez de volume de zipcode."""
    edges = np.quantile(distances, [1 / 3, 2 / 3])
    return [float(e) for e in edges]


def apply_coverage_bucket(distances: np.ndarray, edges: list[float]) -> np.ndarray:
    idx = np.digitize(distances, edges)  # 0: <edge0 (HIGH) | 1: entre (MEDIUM) | 2: >edge1 (LOW)
    return np.array(COVERAGE_LABELS)[idx]


# ---------------------------------------------------------------------------------------------
# Chave de segmento (price_band x cluster x cobertura x segmento conhecido de erro alto)
# ---------------------------------------------------------------------------------------------
def hard_segment_flag(df: pd.DataFrame) -> pd.Series:
    """waterfront=1 ou grade>=10 — segmentos com erro 2-4x maior, já documentados no Error Matrix
    (fase 11/13, `src/evaluation/error_matrix.py::characteristic_cuts`)."""
    return (df["waterfront"] == 1) | (df["grade"] >= 10)


def _hard_label(flag: pd.Series) -> pd.Series:
    return np.where(flag, "HARD", "NORMAL")


def segment_keys(df: pd.DataFrame, coverage_bucket: np.ndarray) -> pd.DataFrame:
    """Retorna as 3 chaves de segmento (fina -> grosseira) usadas no backoff."""
    hard = _hard_label(hard_segment_flag(df))
    band = df["price_band"].astype(str).to_numpy()
    cluster = df["property_cluster"].astype(str).to_numpy()
    return pd.DataFrame({
        "level0": [f"{b}|{c}|{cv}|{h}" for b, c, cv, h in zip(band, cluster, coverage_bucket, hard)],
        "level1": [f"{b}|{cv}|{h}" for b, cv, h in zip(band, coverage_bucket, hard)],
        "level2": [f"{cv}|{h}" for cv, h in zip(coverage_bucket, hard)],
    })


# ---------------------------------------------------------------------------------------------
# Tabela de erro esperado por segmento, calculada em TEST, com backoff por amostra pequena (P4)
# ---------------------------------------------------------------------------------------------
def _group_stats(df: pd.DataFrame, key_col: str, ape_col: str, ae_col: str) -> dict:
    out = {}
    for key, sub in df.groupby(key_col, observed=True):
        out[key] = {
            "n": int(len(sub)),
            "median_ape": float(sub[ape_col].median()),
            "median_abs_error": float(sub[ae_col].median()),
        }
    return out


def build_segment_lookup(test_df: pd.DataFrame, min_n: int = LOW_SAMPLE_THRESHOLD) -> dict:
    """`test_df` precisa ter as colunas `level0`/`level1`/`level2` (ver `segment_keys`) e `ape`/`abs_error`."""
    level0 = _group_stats(test_df, "level0", "ape", "abs_error")
    level1 = _group_stats(test_df, "level1", "ape", "abs_error")
    level2 = _group_stats(test_df, "level2", "ape", "abs_error")
    glob = {
        "n": int(len(test_df)),
        "median_ape": float(test_df["ape"].median()),
        "median_abs_error": float(test_df["abs_error"].median()),
    }
    return {"level0": level0, "level1": level1, "level2": level2, "global": glob, "min_n": min_n}


def expected_error_for_row(level0_key: str, level1_key: str, level2_key: str, lookup: dict) -> dict:
    """Backoff level0 -> level1 -> level2 -> global, parando no primeiro nível com `n >= min_n`."""
    min_n = lookup["min_n"]
    for level_name, key in (("level0", level0_key), ("level1", level1_key), ("level2", level2_key)):
        stats = lookup[level_name].get(key)
        if stats is not None and stats["n"] >= min_n:
            return {**stats, "level_used": level_name, "segment_key": key, "low_sample": False}
    stats = lookup["global"]
    return {**stats, "level_used": "global", "segment_key": GLOBAL_KEY, "low_sample": stats["n"] < min_n}


def assign_expected_error(df_with_keys: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    rows = [
        expected_error_for_row(r.level0, r.level1, r.level2, lookup)
        for r in df_with_keys.itertuples()
    ]
    return pd.DataFrame(rows, index=df_with_keys.index)


# ---------------------------------------------------------------------------------------------
# Confidence Score (0-100) — rank empírico do erro esperado do segmento vs. toda a distribuição TEST
# ---------------------------------------------------------------------------------------------
def build_ecdf_reference(expected_ape_test: np.ndarray) -> np.ndarray:
    return np.sort(expected_ape_test)


def confidence_score(expected_ape: np.ndarray | float, ecdf_reference: np.ndarray) -> np.ndarray:
    expected_ape = np.atleast_1d(expected_ape)
    n = len(ecdf_reference)
    idx = np.searchsorted(ecdf_reference, expected_ape, side="right")
    ecdf = idx / n
    return 100.0 * (1.0 - ecdf)


# ---------------------------------------------------------------------------------------------
# Calibração isotônica score -> erro esperado real, e derivação EMPÍRICA dos cortes de categoria
# ---------------------------------------------------------------------------------------------
def fit_isotonic_calibration(scores_test: np.ndarray, actual_ape_test: np.ndarray) -> IsotonicRegression:
    """Monótona (score maior nunca implica erro calibrado maior) — sem peso escolhido à mão."""
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(scores_test, actual_ape_test)
    return model


def isotonic_curve_points(model: IsotonicRegression, n_points: int = 101) -> list[list[float]]:
    grid = np.linspace(0, 100, n_points)
    calibrated = model.predict(grid)
    return [[float(s), float(a)] for s, a in zip(grid, calibrated)]


def calibrated_ape_breakpoints(isotonic_model: IsotonicRegression, scores_test: np.ndarray) -> dict:
    """P25/P50/P75 do erro CALIBRADO (`isotonic.predict(score)`) de cada linha de TEST — não do APE
    bruto individual. O APE bruto individual tem cauda mais larga que qualquer segmento consegue
    prometer (o score é uma média de segmento, sempre mais suave que o pior/melhor caso individual);
    usar o quartil bruto como corte deixava a categoria "Alta confiança" praticamente inatingível
    (nenhum segmento chega tão baixo quanto o melhor caso individual isolado). Cortar nos quartis da
    própria distribuição calibrada garante que os 4 cortes são sempre atingíveis pela curva."""
    calibrated = isotonic_model.predict(scores_test)
    return {
        "p25": float(np.percentile(calibrated, 25)),
        "p50": float(np.percentile(calibrated, 50)),
        "p75": float(np.percentile(calibrated, 75)),
    }


def derive_category_thresholds(curve_points: list[list[float]], ape_breakpoints: dict) -> dict:
    """Inverte a curva score->APE calibrada: acha o score onde o erro calibrado cruza cada quartil
    da distribuição CALIBRADA em TEST (`calibrated_ape_breakpoints`, P25/P50/P75) — os cortes de
    categoria vêm do dado, não de "parecer bom" (pedido explícito: 80/60/40 são ilustrativos,
    precisam ser calibrados)."""
    grid = np.array(curve_points)
    scores, ape = grid[:, 0], grid[:, 1]
    order = np.argsort(ape)  # np.interp exige xp crescente
    ape_sorted, scores_sorted = ape[order], scores[order]

    def score_at_ape(bp: float) -> float:
        return float(np.clip(np.interp(bp, ape_sorted, scores_sorted), 0.0, 100.0))

    alta_min = score_at_ape(ape_breakpoints["p25"])
    moderada_min = score_at_ape(ape_breakpoints["p50"])
    baixa_min = score_at_ape(ape_breakpoints["p75"])
    return {"alta_min": alta_min, "moderada_min": moderada_min, "baixa_min": baixa_min}


CATEGORY_LABELS = ["Alta confiança", "Confiança moderada", "Baixa confiança", "Muito baixa confiança / revisão"]


def categorize(score: float, thresholds: dict) -> str:
    if score >= thresholds["alta_min"]:
        return CATEGORY_LABELS[0]
    if score >= thresholds["moderada_min"]:
        return CATEGORY_LABELS[1]
    if score >= thresholds["baixa_min"]:
        return CATEGORY_LABELS[2]
    return CATEGORY_LABELS[3]


# ---------------------------------------------------------------------------------------------
# Matriz de Confiança 2x2 simplificada (cobertura x erro esperado) — versão didática do score
# ---------------------------------------------------------------------------------------------
def build_confidence_matrix_2x2(distance: np.ndarray, expected_ape: np.ndarray, actual_ape: np.ndarray,
                                 abs_error: np.ndarray, distance_median: float,
                                 expected_ape_median: float) -> list[dict]:
    """Os dois eixos (`distance`, `expected_ape`) precisam ser conhecidos ANTES de saber o resultado
    real (senão não é confiança, é diagnóstico retrospectivo) — por isso o corte "erro esperado
    baixo/alto" usa `expected_ape` (lookup de segmento, fase de calibração), nunca `actual_ape`.
    `actual_ape`/`abs_error` só entram para reportar o que de fato aconteceu dentro de cada célula."""
    coverage_high = distance <= distance_median
    error_low = expected_ape <= expected_ape_median
    ape = actual_ape
    cells = [
        ("cobertura_alta__erro_baixo", coverage_high & error_low, "ALTA"),
        ("cobertura_alta__erro_alto", coverage_high & ~error_low, "MEDIA"),
        ("cobertura_baixa__erro_baixo", ~coverage_high & error_low, "MEDIA"),
        ("cobertura_baixa__erro_alto", ~coverage_high & ~error_low, "BAIXA"),
    ]
    out = []
    for name, mask, confidence_level in cells:
        n = int(mask.sum())
        out.append({
            "cell": name,
            "confidence_level": confidence_level,
            "n": n,
            "median_ape": float(np.median(ape[mask])) if n else None,
            "median_abs_error": float(np.median(abs_error[mask])) if n else None,
            "low_sample": n < LOW_SAMPLE_THRESHOLD,
        })
    return out
