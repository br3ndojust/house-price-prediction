from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.infrastructure.persistence.models import Base

_IS_SQLITE = settings.database_url.startswith("sqlite")

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
)

if _IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record) -> None:
        # WAL + synchronous=NORMAL: cada commit deixa de forçar um fsync completo (modo DELETE
        # padrão do SQLite) — cada `POST /predictions` chama `session.commit()` uma vez, e um lote
        # de 500 imóveis (`PredictBatch`) fazia 500 commits individuais; em modo DELETE isso sozinho
        # já passava de 20s (o timeout real reportado em lotes >1000 linhas, não só o índice de
        # cobertura refeito por linha — ver ConfidenceScorer). WAL ainda é durável (o writer só perde
        # a última transação não sincronizada em caso de crash do SO, não corrompe o banco).
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def create_all_tables() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
