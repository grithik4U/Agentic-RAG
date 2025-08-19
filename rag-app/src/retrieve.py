from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore
import numpy as np

from . import db
from .config import EMBED_MODEL, K, RERANK_TOP_M
from .embed_index import embed_texts, load_index


@dataclass
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_id: int
    page: Optional[int]
    text: str
    score: float


def _load_index_or_build() -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    index, meta = load_index()
    return index, meta


def retrieve(query: str, k: int = K, m: int = RERANK_TOP_M) -> List[RetrievedChunk]:
    index, meta = _load_index_or_build()
    if index is None:
        return []

    q_vec = embed_texts([query], EMBED_MODEL).astype(np.float32)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, m)
    scores = D[0]
    indices = I[0]
    rows = db.all_chunks_with_embeddings()
    if not rows:
        return []

    # Map index position to chunk row
    retrieved: List[Tuple[float, int]] = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(rows):
            continue
        retrieved.append((float(score), idx))

    # Rerank by cosine score (already cosine via normalized IP)
    retrieved.sort(key=lambda x: x[0], reverse=True)
    top = retrieved[:k]

    results: List[RetrievedChunk] = []
    # Preload doc filename map
    docs = {r["id"]: r for r in db.list_documents()}
    for score, idx in top:
        r = rows[idx]
        doc = docs.get(r["document_id"]) or db.get_document(r["document_id"])  # fallback
        filename = doc["filename"] if doc else r["document_id"]
        results.append(
            RetrievedChunk(
                document_id=r["document_id"],
                filename=filename,
                chunk_id=int(r["chunk_id"]),
                page=int(r["page"]) if r["page"] is not None else None,
                text=r["text"],
                score=float(score),
            )
        )

    return results

