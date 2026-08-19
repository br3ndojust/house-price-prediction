from __future__ import annotations

from typing import Protocol


class FeedbackRepository(Protocol):
    def record(self, prediction_id: str, actual_price: float) -> None:
        ...
