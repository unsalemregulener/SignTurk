"""Database configuration: PostgreSQL from DATABASE_URL, SQLite otherwise."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


BASE_DIR = Path(__file__).resolve().parent

# Credentials are read from the environment (or a local .env); none live in code.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{(BASE_DIR / 'local_dev.db').as_posix()}"


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        # FastAPI serves requests from a thread pool
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def database_name() -> str:
    """Return a credential-free label for the active database."""
    return "SQLite (local)" if DATABASE_URL.startswith("sqlite") else "PostgreSQL"


def get_db():
    """FastAPI dependency that yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
