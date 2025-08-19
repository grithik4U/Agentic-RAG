import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from pathlib import Path

from . import db
from .chunk import chunk_document
from .config import EMBED_MODEL
from .embed_index import build_faiss_index, embed_texts, save_index
from .text_extract import extract_text_and_pages, extract_pdf_pages
from .utils import new_id


@dataclass
class JobStatus:
    job_id: str
    state: str = "queued"  # queued | processing | done | error
    queued_docs: List[str] = field(default_factory=list)
    processed_docs: List[str] = field(default_factory=list)
    total_chunks: int = 0
    embedded_chunks: int = 0
    error: Optional[str] = None


_jobs: Dict[str, JobStatus] = {}
_lock = threading.Lock()


def start_processing(doc_ids: Optional[List[str]] = None) -> str:
    if doc_ids is None:
        # Find pending docs
        docs = db.list_documents()
        doc_ids = [d["id"] for d in docs if d["status"] in ("PENDING", "NEEDS_PROCESSING")]

    job_id = new_id("job")
    status = JobStatus(job_id=job_id, state="queued", queued_docs=list(doc_ids))
    with _lock:
        _jobs[job_id] = status

    t = threading.Thread(target=_run_job, args=(job_id, doc_ids), daemon=True)
    t.start()
    return job_id


def get_status(job_id: str) -> Optional[JobStatus]:
    with _lock:
        return _jobs.get(job_id)


def _run_job(job_id: str, doc_ids: List[str]) -> None:
    status = get_status(job_id)
    if status is None:
        return
    status.state = "processing"

    try:
        # Extract, chunk, embed for each doc
        for doc_id in doc_ids:
            d = db.get_document(doc_id)
            if not d:
                continue
            if d["status"] == "DUPLICATE":
                status.processed_docs.append(doc_id)
                continue
            try:
                path = Path(d["path"])
                ext = d["ext"].lower()
                rows = []
                cid_counter = 0
                if ext == "pdf":
                    for page_num, page_text in extract_pdf_pages(path):
                        for offset, ctext in chunk_document(page_text):
                            rows.append({
                                "id": f"chunk_{doc_id}_{cid_counter}",
                                "document_id": doc_id,
                                "chunk_id": cid_counter,
                                "text": ctext,
                                "page": page_num,
                                "embedding": None,
                            })
                            cid_counter += 1
                else:
                    text, _ = extract_text_and_pages(path, ext)
                    for offset, ctext in chunk_document(text):
                        rows.append({
                            "id": f"chunk_{doc_id}_{cid_counter}",
                            "document_id": doc_id,
                            "chunk_id": cid_counter,
                            "text": ctext,
                            "page": None,
                            "embedding": None,
                        })
                        cid_counter += 1
            except Exception as e:
                db.update_document_status(doc_id, f"ERROR: {e}")
                continue

            db.insert_chunks_bulk(rows)

            # Fetch all chunks for this doc and embed missing
            doc_chunks = [r for r in db.chunks_for_document(doc_id)]
            missing = [r for r in doc_chunks if r["embedding"] is None]
            status.total_chunks += len(missing)
            if missing:
                vecs = embed_texts([r["text"] for r in missing])
                for r, vec in zip(missing, vecs):
                    db.insert_chunk(
                        {
                            "id": r["id"],
                            "document_id": r["document_id"],
                            "chunk_id": r["chunk_id"],
                            "text": r["text"],
                            "page": r["page"],
                            "embedding": vec.tobytes(),
                        }
                    )
                    status.embedded_chunks += 1

            db.update_document_status(doc_id, "READY")
            status.processed_docs.append(doc_id)

        # Rebuild index from all embeddings
        all_rows = db.all_chunks_with_embeddings()
        if all_rows:
            vectors = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in all_rows])
            index = build_faiss_index(vectors)
            meta = {"model": EMBED_MODEL, "dim": str(vectors.shape[1])}
            save_index(index, meta)

        status.state = "done"
    except Exception as e:  # noqa: BLE001
        status.state = "error"
        status.error = f"{e}\n{traceback.format_exc()}"

from pathlib import Path

