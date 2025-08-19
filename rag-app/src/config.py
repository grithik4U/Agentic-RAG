import os
from pathlib import Path


def getenv_str(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return str(value)


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


OPENAI_API_KEY = getenv_str("OPENAI_API_KEY", "")

EMBED_MODEL = getenv_str("EMBED_MODEL", "text-embedding-3-small")
GENERATE_MODEL = getenv_str("GENERATE_MODEL", "gpt-4o-mini")

CHUNK_SIZE = getenv_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = getenv_int("CHUNK_OVERLAP", 150)

K = getenv_int("K", 5)
RERANK_TOP_M = getenv_int("RERANK_TOP_M", 20)
CONFIDENCE_THRESHOLD = getenv_float("CONFIDENCE_THRESHOLD", 0.22)

DB_PATH = Path(getenv_str("DB_PATH", "data/rag.db"))
INDEX_PATH = Path(getenv_str("INDEX_PATH", "data/index"))
CACHE_PATH = Path(getenv_str("CACHE_PATH", "data/cache"))

UPLOADS_PATH = Path("data/uploads")
EMBED_CACHE_PATH = CACHE_PATH / "embeddings"

HOST = getenv_str("HOST", "127.0.0.1")
PORT = getenv_int("PORT", 8000)

# Ensure directories
for p in [INDEX_PATH, CACHE_PATH, UPLOADS_PATH, EMBED_CACHE_PATH]:
    p.mkdir(parents=True, exist_ok=True)

