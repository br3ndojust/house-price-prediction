from __future__ import annotations

from enum import StrEnum


class PriceBand(StrEnum):
    """Bandas semânticas de preço — quartis fit em TRAIN (fase 03/06), P3."""

    ENTRY = "Entry"
    STANDARD = "Standard"
    PREMIUM = "Premium"
    LUXURY = "Luxury"
