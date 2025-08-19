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
	q_vec = embed_texts([query], EMBED_MODEL).astype(np.float32)
	if faiss is not None:
		faiss.normalize_L2(q_vec)

	index, meta = _load_index_or_build()
	rows = db.all_chunks_with_embeddings()
	if not rows:
		return []

	retrieved: List[Tuple[float, int]] = []
	if index is not None and faiss is not None:
		D, I = index.search(q_vec, m)
		scores = D[0]
		indices = I[0]
		for score, idx in zip(scores, indices):
			if idx < 0 or idx >= len(rows):
				continue
			retrieved.append((float(score), idx))
	else:
		# Fallback brute-force cosine similarity
		mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
		mat_norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
		mat_n = mat / mat_norms
		q = q_vec[0]
		qn = q / (np.linalg.norm(q) + 1e-12)
		sims = mat_n @ qn
		order = np.argsort(-sims)[:m]
		for idx in order.tolist():
			retrieved.append((float(sims[idx]), int(idx)))

	retrieved.sort(key=lambda x: x[0], reverse=True)
	top = retrieved[:k]

	results: List[RetrievedChunk] = []
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