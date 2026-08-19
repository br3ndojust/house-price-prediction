from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, HTTPException

from app.core.container import Container, get_container

from ..schemas.explainability import FeatureImportanceOut, PredictionExplanationOut
from ..schemas.inference import PropertyIn
from .inference import _to_property

router = APIRouter(tags=["Explicabilidade"])


@router.get(
    "/model/feature-importance",
    response_model=list[FeatureImportanceOut],
    summary="Ranking global de importância (gain-based, XGBoost)",
    description="As 20 features oficiais, ligadas à camada/hipótese de `data/processing/feature_registry.csv`.",
)
def feature_importance(container: Container = Depends(get_container)) -> list[FeatureImportanceOut]:
    return [FeatureImportanceOut(**dataclasses.asdict(f)) for f in container.get_global_feature_importance.execute()]


@router.get(
    "/model/feature-statistics",
    summary="Correlação com `price` + quartis de cada feature oficial (TRAIN)",
    description=(
        "Correlação de Pearson bruta com `price` (não `price_log`, mais interpretável) e quartis "
        "(min/q1/mediana/q3/max) de cada uma das 20 features oficiais, sobre TRAIN. Usado pelos "
        "gráficos de correlação/quartis do portal (telas Importância de Features e Treino)."
    ),
)
def feature_statistics(container: Container = Depends(get_container)) -> dict[str, dict]:
    return container.get_feature_statistics.execute()


@router.get(
    "/model/feature-sample",
    summary="Amostra de TRAIN (feature + price) para gráfico de dispersão",
    description=(
        "Amostra aleatória de até `sample_size` linhas de TRAIN com os valores das 20 features "
        "oficiais + `price` — só para o gráfico de dispersão do portal, nunca usada para decisão de "
        "modelo (isso é fase 10, sobre o dataset inteiro)."
    ),
)
def feature_sample(sample_size: int = 400, container: Container = Depends(get_container)) -> list[dict]:
    return container.get_feature_sample.execute(sample_size)


@router.get(
    "/model/feature-snapshots",
    summary="Modelos registrados com snapshot de importância/correlação gravado",
    description=(
        "Um snapshot é gravado uma vez, na criação do `ModelVersion` (seed do modelo original ou job "
        "de treino via API) — importância (gain-based) + correlação com `price` + quartis, sobre o "
        "próprio `feature_cols` daquele modelo (nunca o conjunto oficial completo). Fonte do seletor "
        "de modelo da tela Importância de Features do portal — só modelos aqui podem ser analisados, "
        "não só o modelo ativo agora."
    ),
)
def feature_snapshots(container: Container = Depends(get_container)) -> list[dict]:
    return [
        {
            "version": s.version, "created_at": s.created_at,
            "n_train_rows": s.n_train_rows, "n_features": len(s.stats),
        }
        for s in container.list_feature_snapshots.execute()
    ]


@router.get(
    "/model/feature-snapshots/{version}",
    summary="Snapshot completo de importância/correlação/quartis de um modelo registrado",
    description="Só as features que aquele modelo de fato usou (`feature_cols` da versão) — nunca o conjunto oficial completo.",
)
def feature_snapshot(version: str, container: Container = Depends(get_container)) -> dict:
    snapshot = container.get_feature_snapshot.execute(version)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"sem snapshot gravado para a versão '{version}' — modelos anteriores a essa "
                   "funcionalidade não têm histórico; retreine para gerar um novo.",
        )
    return {
        "version": snapshot.version, "created_at": snapshot.created_at,
        "n_train_rows": snapshot.n_train_rows, "features": snapshot.stats,
    }


@router.post(
    "/predictions/explain",
    response_model=PredictionExplanationOut,
    summary="Explica uma predição individual (SHAP local)",
    description=(
        "Mesmo input de POST /predictions + contribuição local por feature (SHAP), calculada no "
        "espaço `price_log` — ver `approx_dollar_note` na resposta para a ressalva sobre a conversão "
        "para dólar."
    ),
)
def explain(data: PropertyIn, container: Container = Depends(get_container)) -> PredictionExplanationOut:
    try:
        explanation = container.explain_prediction.execute(_to_property(data))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return PredictionExplanationOut(
        base_value_log=explanation.base_value_log,
        predicted_value_log=explanation.predicted_value_log,
        predicted_price_dollar=explanation.predicted_price_dollar,
        contributions=[dataclasses.asdict(c) for c in explanation.contributions],
        approx_dollar_note=explanation.approx_dollar_note,
    )
