"""Entidade Property — mesmo schema de `future_unseen_examples.csv` (sem id/date/price, P1).

Validação de invariante alinhada a `configs/data_contract.yaml` (range_checks fase 00) — a API rejeita
entrada fisicamente implausível antes de chegar no pipeline de inferência, nunca depois.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Property:
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not (0 <= self.bathrooms <= 10):
            errors.append("bathrooms fora do intervalo [0, 10]")
        if not (0 <= self.floors <= 5):
            errors.append("floors fora do intervalo [0, 5]")
        if not (1 <= self.grade <= 13):
            errors.append("grade fora do intervalo [1, 13]")
        if not (1 <= self.condition <= 5):
            errors.append("condition fora do intervalo [1, 5]")
        if self.waterfront not in (0, 1):
            errors.append("waterfront deve ser 0 ou 1")
        if not (0 <= self.view <= 4):
            errors.append("view fora do intervalo [0, 4]")
        for field_name in ("sqft_living", "sqft_lot", "bedrooms", "sqft_above", "sqft_basement"):
            if getattr(self, field_name) < 0:
                errors.append(f"{field_name} não pode ser negativo")
        if self.yr_built < 1800 or self.yr_built > 2100:
            errors.append("yr_built implausível")
        if errors:
            raise ValueError("; ".join(errors))
