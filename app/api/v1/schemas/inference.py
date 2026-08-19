from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_PROPERTY_EXAMPLE = {
    "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800, "sqft_lot": 5000, "floors": 1.0,
    "waterfront": 0, "view": 0, "condition": 3, "grade": 7, "sqft_above": 1800,
    "sqft_basement": 0, "yr_built": 1990, "yr_renovated": 0, "zipcode": 98101, "lat": 47.6,
    "long": -122.3, "sqft_living15": 1800, "sqft_lot15": 5000,
}


class PropertyIn(BaseModel):
    """Mesmo schema de `future_unseen_examples.csv` — sem `id`/`date`/`price` (o serviço não exige
    histórico de venda, só atributos físicos + zipcode; `property_age` é calculado a partir da data
    da requisição)."""

    model_config = ConfigDict(json_schema_extra={"example": _PROPERTY_EXAMPLE})

    bedrooms: int = Field(..., ge=0, le=30, description="Número de quartos")
    bathrooms: float = Field(..., ge=0, le=10, description="Número de banheiros (pode ser fracionário)")
    sqft_living: int = Field(..., gt=0, description="Área construída (pés²)")
    sqft_lot: int = Field(..., gt=0, description="Área do terreno (pés²)")
    floors: float = Field(..., ge=0, le=5, description="Número de andares")
    waterfront: int = Field(..., ge=0, le=1, description="1 se possui frente d'água")
    view: int = Field(..., ge=0, le=4, description="Qualidade da vista (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Estado de conservação (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Qualidade de construção/design (1-13)")
    sqft_above: int = Field(..., ge=0, description="Área acima do solo (pés²)")
    sqft_basement: int = Field(..., ge=0, description="Área abaixo do solo/porão (pés²)")
    yr_built: int = Field(..., ge=1800, le=2100, description="Ano de construção")
    yr_renovated: int = Field(0, ge=0, description="Ano da última reforma (0 se nunca reformado)")
    zipcode: int = Field(..., description="CEP (Seattle) — só usado como chave espacial/demográfica")
    lat: float = Field(..., description="Latitude")
    long: float = Field(..., description="Longitude")
    sqft_living15: int = Field(..., gt=0, description="Área construída média dos 15 vizinhos mais próximos")
    sqft_lot15: int = Field(..., gt=0, description="Área de terreno média dos 15 vizinhos mais próximos")


class PredictionOut(BaseModel):
    prediction_id: str = Field(..., description="Id da predição — usado depois em POST /feedback")
    predicted_price: float = Field(..., description="Preço previsto em dólares")
    price_band: str = Field(..., description="Banda de preço prevista (Entry/Standard/Premium/Luxury)")
    property_cluster: int = Field(..., description="Cluster de perfil físico previsto (0, 1, 2, ...)")
    model_version: str = Field(..., description="Versão do modelo ativo que gerou esta predição")
    confidence_score: float | None = Field(
        None, description="Confidence Score 0-100 (calibrado em TEST, verificado em VAL — ver GET "
                           "/model/confidence-calibration). null se a calibração ainda não foi gerada."
    )
    confidence_category: str | None = Field(
        None, description="Alta confiança / Confiança moderada / Baixa confiança / Muito baixa "
                           "confiança / revisão — ver docs/09_confidence_matrix.md"
    )


class BatchPredictionIn(BaseModel):
    properties: list[PropertyIn] = Field(..., min_length=1, max_length=500)


class BatchPredictionOut(BaseModel):
    predictions: list[PredictionOut]
