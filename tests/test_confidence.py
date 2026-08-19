import sys
from pathlib import Path

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
    confidence_score,
    derive_category_thresholds,
    expected_error_for_row,
    fit_coverage_bucket_edges,
    fit_isotonic_calibration,
    hard_segment_flag,
    isotonic_curve_points,
    segment_keys,
)


def test_hard_segment_flag_waterfront_or_high_grade():
    df = pd.DataFrame({"waterfront": [1, 0, 0], "grade": [5, 11, 5]})
    assert list(hard_segment_flag(df)) == [True, True, False]


def test_coverage_bucket_edges_split_into_tercis():
    distances = np.arange(300.0)
    edges = fit_coverage_bucket_edges(distances)
    buckets = apply_coverage_bucket(distances, edges)
    counts = pd.Series(buckets).value_counts()
    assert set(counts.index) == {"HIGH", "MEDIUM", "LOW"}
    assert counts["HIGH"] == 100 and counts["LOW"] == 100


def _toy_test_df(n_per_segment: int = 30) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    rows = []
    # 2 bandas x 2 clusters x 1 cobertura, erro bem separado por banda (Entry barato/bom, Luxury caro/ruim)
    for band, base_ape in [("Entry", 0.05), ("Luxury", 0.30)]:
        for cluster in [0, 1]:
            for _ in range(n_per_segment):
                rows.append({
                    "price_band": band, "property_cluster": cluster,
                    "waterfront": 0, "grade": 7,
                    "ape": max(0.01, rng.normal(base_ape, 0.02)),
                    "abs_error": max(100.0, rng.normal(base_ape * 400000, 5000)),
                })
    return pd.DataFrame(rows)


def test_segment_lookup_and_expected_error_backoff():
    df = _toy_test_df()
    coverage_bucket = np.array(["HIGH"] * len(df))
    keys = segment_keys(df, coverage_bucket)
    df = pd.concat([df, keys], axis=1)

    lookup = build_segment_lookup(df, min_n=20)
    entry_row = df[df["price_band"] == "Entry"].iloc[0]
    luxury_row = df[df["price_band"] == "Luxury"].iloc[0]

    entry_expected = expected_error_for_row(entry_row.level0, entry_row.level1, entry_row.level2, lookup)
    luxury_expected = expected_error_for_row(luxury_row.level0, luxury_row.level1, luxury_row.level2, lookup)

    assert entry_expected["low_sample"] is False
    assert entry_expected["median_ape"] < luxury_expected["median_ape"]


def test_expected_error_backoff_to_global_when_segment_too_small():
    df = _toy_test_df(n_per_segment=5)  # abaixo do min_n -> força backoff
    coverage_bucket = np.array(["HIGH"] * len(df))
    keys = segment_keys(df, coverage_bucket)
    df = pd.concat([df, keys], axis=1)

    lookup = build_segment_lookup(df, min_n=20)
    row = df.iloc[0]
    result = expected_error_for_row(row.level0, row.level1, row.level2, lookup)
    assert result["level_used"] in {"level1", "level2", "global"}


def test_confidence_score_higher_for_lower_expected_error():
    ecdf_ref = build_ecdf_reference(np.array([0.05, 0.10, 0.15, 0.20, 0.30]))
    good_score = confidence_score(0.05, ecdf_ref)[0]
    bad_score = confidence_score(0.30, ecdf_ref)[0]
    assert good_score > bad_score
    assert 0 <= good_score <= 100 and 0 <= bad_score <= 100


def test_isotonic_calibration_is_monotonic_non_increasing():
    rng = np.random.RandomState(0)
    scores = rng.uniform(0, 100, 200)
    actual_ape = 0.3 - 0.002 * scores + rng.normal(0, 0.01, 200)  # score maior -> erro menor, com ruido
    model = fit_isotonic_calibration(scores, actual_ape)
    curve = isotonic_curve_points(model, n_points=11)
    values = [v for _, v in curve]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))  # nao-crescente em score


def test_derive_category_thresholds_are_ordered_and_bounded():
    rng = np.random.RandomState(0)
    scores = rng.uniform(0, 100, 300)
    actual_ape = 0.3 - 0.002 * scores + rng.normal(0, 0.01, 300)
    model = fit_isotonic_calibration(scores, actual_ape)
    curve = isotonic_curve_points(model)
    breakpoints = calibrated_ape_breakpoints(model, scores)

    thresholds = derive_category_thresholds(curve, breakpoints)
    assert thresholds["alta_min"] > thresholds["moderada_min"] > thresholds["baixa_min"]
    assert all(0 <= v <= 100 for v in thresholds.values())


def test_categorize_maps_score_to_expected_label():
    thresholds = {"alta_min": 80, "moderada_min": 60, "baixa_min": 40}
    assert categorize(95, thresholds) == "Alta confiança"
    assert categorize(70, thresholds) == "Confiança moderada"
    assert categorize(50, thresholds) == "Baixa confiança"
    assert categorize(10, thresholds) == "Muito baixa confiança / revisão"


def test_confidence_matrix_2x2_uses_expected_not_actual_for_split():
    # erro esperado (conhecido antes) nao bate com o erro real (conhecido soh depois) de proposito -
    # a celula tem que refletir o corte por ESPERADO, reportando o REAL dentro dela.
    distance = np.array([1.0, 1.0, 5.0, 5.0])
    expected_ape = np.array([0.05, 0.05, 0.05, 0.05])  # todo mundo "esperado baixo"
    actual_ape = np.array([0.05, 0.40, 0.05, 0.40])  # metade surpreende com erro real alto
    abs_error = actual_ape * 100000

    matrix = build_confidence_matrix_2x2(distance, expected_ape, actual_ape, abs_error,
                                          distance_median=3.0, expected_ape_median=0.1)
    by_cell = {row["cell"]: row for row in matrix}
    # como expected_ape_median=0.1 e todo expected=0.05, TODAS as linhas caem no lado "erro_baixo"
    assert by_cell["cobertura_alta__erro_baixo"]["n"] == 2
    assert by_cell["cobertura_baixa__erro_baixo"]["n"] == 2
    assert by_cell["cobertura_alta__erro_alto"]["n"] == 0
    # dentro da celula "esperado baixo", o real reportado mistura 0.05 e 0.40 (media = 0.225)
    assert by_cell["cobertura_alta__erro_baixo"]["median_ape"] == 0.225
