import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Veritabani baglantisi ortam degiskeninden okunur — gizli bilgi koda gomulmez.
# Kendi .env dosyana DATABASE_URL ekle (ornek icin .env.example).
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL ayarli degil. Proje kokune bir .env olusturup "
        "DATABASE_URL=... ekle (bkz. .env.example)."
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session, auto-closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()