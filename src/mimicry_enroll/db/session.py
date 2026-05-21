from __future__ import annotations

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session

from mimicry_enroll.config import settings
from mimicry_enroll.db.models import Base


def _fix_url(url: str) -> str:
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.split("://", 1)[1]
    return url


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_fix_url(settings.database_url), pool_pre_ping=True)
    return _engine


REQUIRED_COLUMNS = {"display_name", "main_emotion", "key_type"}


def init_db() -> None:
    """Создать таблицы. Если схема устарела (отсутствуют новые колонки) — пересоздать."""
    engine = get_engine()
    inspector = inspect(engine)
    if inspector.has_table("enrolled_users"):
        existing = {col["name"] for col in inspector.get_columns("enrolled_users")}
        if not REQUIRED_COLUMNS.issubset(existing):
            Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()
