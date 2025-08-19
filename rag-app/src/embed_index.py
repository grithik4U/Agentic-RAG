import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import faiss  # type: ignore
except Exception as e:  # noqa: BLE001
    faiss = None  # type: ignore
import numpy as np
from openai import OpenAI

from .config import EMBED_CACHE_PATH, EMBED_MODEL, INDEX_PATH, OPENAI_API_KEY
from .utils import compute_sha256_bytes


def get_openai_client() -> OpenAI:
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    return OpenAI(api_key=api_key)


def embedding_dimension(model: str) -> int:
    # Known dims for popular OpenAI models
    if model == "text-embedding-3-small":
        return 1536
    if model == "text-embedding-3-large":
        return 3072
    # Fallback to 1536 which works for many models
    return 1536


def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    client = get_openai_client()
    # Caching per text using content hash keyed by model
    vectors: List[np.ndarray] = []
    EMBED_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    dim = embedding_dimension(model)
    batch_texts: List[str] = []
    batch_indices: List[int] = []
    cached: Dict[int, np.ndarray] = {}
    for i, t in enumerate(texts):
        key = f"{model}_{compute_sha256_bytes(t.encode('utf-8'))}.npy"
        fp = EMBED_CACHE_PATH / key
        if fp.exists():
            vec = np.load(fp)
            if vec.shape[0] != dim:
                fp.unlink(missing_ok=True)
            else:
                cached[i] = vec
                continue
        batch_texts.append(t)
        batch_indices.append(i)

    if batch_texts:
        resp = client.embeddings.create(model=model, input=batch_texts)
        for i, data in zip(batch_indices, resp.data):
            vec = np.array(data.embedding, dtype=np.float32)
            key = f"{model}_{compute_sha256_bytes(texts[i].encode('utf-8'))}.npy"
            fp = EMBED_CACHE_PATH / key
            np.save(fp, vec)
            cached[i] = vec

    # Order results to original order
    for i in range(len(texts)):
        vectors.append(cached[i])
    return np.vstack(vectors)


def build_faiss_index(vectors: np.ndarray):
    # Normalize for cosine similarity via inner product
    if faiss is None:
        raise RuntimeError("FAISS not available; please install faiss-cpu or use Python < 3.13.")
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index, meta: Dict[str, str], path: Path = INDEX_PATH) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if faiss is None:
        raise RuntimeError("FAISS not available")
    faiss.write_index(index, str(path / "index.faiss"))
    (path / "manifest.json").write_text(json.dumps(meta, indent=2))


def load_index(path: Path = INDEX_PATH) -> Tuple[Optional[object], Optional[Dict[str, str]]]:
    index_file = path / "index.faiss"
    manifest_file = path / "manifest.json"
    if not index_file.exists() or not manifest_file.exists():
        return None, None
    if faiss is None:
        return None, None
    index = faiss.read_index(str(index_file))
    meta = json.loads(manifest_file.read_text())
    return index, meta

