"""Exporta o schema OpenAPI (Swagger) estático da API para `docs/api/openapi.json`.

A API já serve Swagger UI interativo em `/docs` (e ReDoc em `/redoc`) via FastAPI — este script só
gera uma cópia estática, útil para revisar o contrato sem subir o servidor (ex: em PR review).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from app.main import app

    out_path = Path("docs/api/openapi.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OpenAPI schema exportado para {out_path}")


if __name__ == "__main__":
    main()
