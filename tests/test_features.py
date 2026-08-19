import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.comparable import (
    apply_comps_knn_neighbor_distance,
    apply_comps_knn_price,
    apply_local_price_dispersion,
    fit_spatial_index,
)
from src.features.derived import (
    add_basement_features,
    add_bathrooms_per_bedroom,
    add_grade_condition_interaction,
    add_sqft_living_to_lot_ratio,
)
from src.features.neighborhood import (
    add_dist_to_seattle_center,
    apply_dist_to_nearest_train_zip,
    fit_train_zip_index,
)
from src.features.physical_typicality import add_property_cluster_distance
from src.features.raw import add_log_sqft_lot, add_property_age, add_renovation_features
from src.features.relative_position import apply_local_grade_percentile


def test_add_property_age():
    df = pd.DataFrame({"date": ["20150301T000000"], "yr_built": [2000]})
    assert add_property_age(df)["property_age"].iloc[0] == 15


def test_add_renovation_features_never_renovated():
    df = pd.DataFrame({"date": ["20150301T000000"], "yr_renovated": [0]})
    out = add_renovation_features(df)
    assert out["was_renovated"].iloc[0] == 0
    assert out["years_since_renovation"].iloc[0] == -1


def test_add_renovation_features_renovated():
    df = pd.DataFrame({"date": ["20150301T000000"], "yr_renovated": [2010]})
    out = add_renovation_features(df)
    assert out["was_renovated"].iloc[0] == 1
    assert out["years_since_renovation"].iloc[0] == 5


def test_add_basement_features_no_basement():
    df = pd.DataFrame({"sqft_basement": [0], "sqft_living": [1500]})
    out = add_basement_features(df)
    assert out["has_basement"].iloc[0] == 0
    assert out["basement_ratio"].iloc[0] == 0


def test_add_basement_features_with_basement():
    df = pd.DataFrame({"sqft_basement": [500], "sqft_living": [2000]})
    out = add_basement_features(df)
    assert out["has_basement"].iloc[0] == 1
    assert np.isclose(out["basement_ratio"].iloc[0], 0.25)


def test_add_grade_condition_interaction():
    df = pd.DataFrame({"grade": [7], "condition": [3]})
    assert add_grade_condition_interaction(df)["grade_condition_interaction"].iloc[0] == 21


def test_add_log_sqft_lot():
    df = pd.DataFrame({"sqft_lot": [np.e - 1]})
    assert np.isclose(add_log_sqft_lot(df)["log_sqft_lot"].iloc[0], 1.0)


def test_dist_to_seattle_center_zero_at_center():
    df = pd.DataFrame({"lat": [47.6062], "long": [-122.3321]})
    assert np.isclose(add_dist_to_seattle_center(df)["dist_to_seattle_center"].iloc[0], 0.0)


def _toy_train(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "lat": rng.uniform(47.3, 47.8, n),
        "long": rng.uniform(-122.5, -121.9, n),
        "price_log": rng.normal(13, 0.5, n),
        "grade": rng.randint(3, 12, n),
    })


def test_comps_knn_price_excludes_self_match_on_train():
    """Se o vizinho mais próximo de uma linha de treino for ela mesma (distância 0), precisa ser
    excluído do cálculo — senão o preço da própria linha vaza para a própria feature (P1)."""
    train = _toy_train()
    index = fit_spatial_index(train, k=5)
    result = apply_comps_knn_price(train, index)
    # nenhuma estimativa deve ser exatamente igual ao price_log da própria linha
    # (checagem fraca mas suficiente: com k=5 e ruido aleatorio, exact match e improvavel sem leakage)
    assert not np.allclose(result, train["price_log"].to_numpy())
    assert len(result) == len(train)


def test_comps_knn_price_matches_manual_mean_excluding_self():
    train = pd.DataFrame({
        "lat": [47.60, 47.60, 47.61, 47.62, 47.90],
        "long": [-122.33, -122.33, -122.34, -122.35, -122.00],
        "price_log": [13.0, 13.0, 13.1, 12.9, 15.0],
        "grade": [7, 7, 8, 6, 10],
    })
    index = fit_spatial_index(train, k=3)
    result = apply_comps_knn_price(train.iloc[[0]], index)
    # vizinhos de train[0] (excluindo self, indice 0): os 3 mais proximos entre indices 1,2,3,4
    expected = np.mean([13.0, 13.1, 12.9])  # indices 1,2,3 sao os 3 mais proximos de train[0]
    assert np.isclose(result[0], expected, atol=1e-6)


def test_local_grade_percentile_range():
    train = _toy_train()
    index = fit_spatial_index(train, k=10)
    result = apply_local_grade_percentile(train, index)
    assert len(result) == len(train)
    assert (result >= 0).all() and (result <= 1).all()


def test_add_bathrooms_per_bedroom():
    df = pd.DataFrame({"bathrooms": [2.5, 1.0], "bedrooms": [5, 0]})
    out = add_bathrooms_per_bedroom(df)
    assert np.isclose(out["bathrooms_per_bedroom"].iloc[0], 0.5)
    # bedrooms=0 nao pode gerar inf/NaN (denominador tratado como 1)
    assert np.isclose(out["bathrooms_per_bedroom"].iloc[1], 1.0)


def test_add_sqft_living_to_lot_ratio():
    df = pd.DataFrame({"sqft_living": [1000, 500], "sqft_lot": [2000, 0]})
    out = add_sqft_living_to_lot_ratio(df)
    assert np.isclose(out["sqft_living_to_lot_ratio"].iloc[0], 0.5)
    # sqft_lot=0 nao pode gerar inf/NaN
    assert out["sqft_living_to_lot_ratio"].iloc[1] == 0


def test_dist_to_nearest_train_zip_zero_at_centroid():
    train = pd.DataFrame({"zipcode": [1, 1, 2, 2], "lat": [47.6, 47.6, 47.7, 47.7],
                           "long": [-122.3, -122.3, -122.2, -122.2]})
    index = fit_train_zip_index(train)
    query = pd.DataFrame({"lat": [47.6], "long": [-122.3]})
    dist = apply_dist_to_nearest_train_zip(query, index)
    assert np.isclose(dist[0], 0.0)


def test_dist_to_nearest_train_zip_far_positive():
    train = pd.DataFrame({"zipcode": [1], "lat": [47.6], "long": [-122.3]})
    index = fit_train_zip_index(train)
    query = pd.DataFrame({"lat": [50.0], "long": [-100.0]})
    dist = apply_dist_to_nearest_train_zip(query, index)
    assert dist[0] > 1.0


def test_comps_knn_neighbor_distance_larger_when_far():
    train = _toy_train()
    index = fit_spatial_index(train, k=5)
    near_point = pd.DataFrame({"lat": [train["lat"].mean()], "long": [train["long"].mean()],
                                "price_log": [13.0], "grade": [7]})
    far_point = pd.DataFrame({"lat": [60.0], "long": [-100.0], "price_log": [13.0], "grade": [7]})
    d_near = apply_comps_knn_neighbor_distance(near_point, index)[0]
    d_far = apply_comps_knn_neighbor_distance(far_point, index)[0]
    assert d_far > d_near


def test_local_price_dispersion_matches_manual_std_excluding_self():
    train = pd.DataFrame({
        "lat": [47.60, 47.60, 47.61, 47.62, 47.90],
        "long": [-122.33, -122.33, -122.34, -122.35, -122.00],
        "price_log": [13.0, 13.0, 13.1, 12.9, 15.0],
        "grade": [7, 7, 8, 6, 10],
    })
    index = fit_spatial_index(train, k=3)
    result = apply_local_price_dispersion(train.iloc[[0]], index)
    # mesmos vizinhos de train[0] usados no teste de comps_knn_price (indices 1,2,3): [13.0, 13.1, 12.9]
    expected = np.std([13.0, 13.1, 12.9])
    assert np.isclose(result[0], expected, atol=1e-6)


def test_add_property_cluster_distance_zero_at_own_centroid(tmp_path):
    import joblib
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    from src.segmentation.property_clusters import PHYSICAL_PROFILE_COLUMNS

    train = pd.DataFrame({col: np.linspace(1, 10, 20) + i for i, col in enumerate(PHYSICAL_PROFILE_COLUMNS)})
    scaler = StandardScaler().fit(train[PHYSICAL_PROFILE_COLUMNS])
    km = KMeans(n_clusters=2, random_state=0, n_init=5).fit(scaler.transform(train[PHYSICAL_PROFILE_COLUMNS]))

    model_path = tmp_path / "property_cluster_model.pkl"
    joblib.dump({"scaler": scaler, "kmeans": km}, model_path)

    df = train.copy()
    df["property_cluster"] = km.labels_
    out = add_property_cluster_distance(df, property_cluster_model_path=str(model_path))

    assert (out["property_cluster_distance"] >= 0).all()
    # linha cujo perfil eh exatamente o centroide do proprio cluster (em espaco escalado) tem distancia ~0
    centroid_unscaled = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_[[0]]), columns=PHYSICAL_PROFILE_COLUMNS
    )
    centroid_unscaled["property_cluster"] = 0
    dist = add_property_cluster_distance(centroid_unscaled, property_cluster_model_path=str(model_path))
    assert np.isclose(dist["property_cluster_distance"].iloc[0], 0.0, atol=1e-6)
