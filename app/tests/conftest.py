"""Isola cada execução de teste num SQLite temporário — nunca escreve em `app_data/app.db` real.

Precisa rodar ANTES de qualquer `import app.*` (settings é lido uma vez no import de
`app.core.config`), por isso as variáveis de ambiente são setadas no topo do módulo, e o `conftest.py`
é sempre importado pelo pytest antes dos módulos de teste do mesmo pacote.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_TEST_DIR = Path(tempfile.mkdtemp(prefix="house_pricing_api_tests_"))
os.environ["APP_DATABASE_URL"] = f"sqlite:///{(_TEST_DIR / 'test_app.db').as_posix()}"
os.environ["APP_APP_DATA_DIR"] = str(_TEST_DIR)
os.environ["APP_API_KEY"] = "test-api-key"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_dir():
    yield
    shutil.rmtree(_TEST_DIR, ignore_errors=True)


@pytest.fixture()
def api_key_headers() -> dict:
    return {"X-API-Key": "test-api-key"}


@pytest.fixture()
def sample_property_payload() -> dict:
    return {
        "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800, "sqft_lot": 5000, "floors": 1.0,
        "waterfront": 0, "view": 0, "condition": 3, "grade": 7, "sqft_above": 1800,
        "sqft_basement": 0, "yr_built": 1990, "yr_renovated": 0, "zipcode": 98042, "lat": 47.6,
        "long": -122.3, "sqft_living15": 1800, "sqft_lot15": 5000,
    }
