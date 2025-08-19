import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from . import db
from .config import HOST, PORT, UPLOADS_PATH
from .generate import generate_answer
from .retrieve import retrieve
from .utils import compute_sha256_bytes, new_id, now_iso, safe_filename
from .worker import get_status, start_processing
from .embed_index import build_faiss_index, save_index
from .config import EMBED_MODEL
import numpy as np


load_dotenv()

ALLOWED_EXTS = {"pdf", "docx", "txt", "md"}
MAX_SIZE = 20 * 1024 * 1024

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

app = FastAPI()


@app.on_event("startup")
def _init_db() -> None:
    db.init_db()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    docs = db.list_documents()
    return templates.TemplateResponse("index.html", {"request": request, "documents": docs})


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    results = []

    for f in files:
        name = f.filename or ""
        ext = name.split(".")[-1].lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")
        data = await f.read()
        if len(data) > MAX_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large (>20MB): {name}")
        sha = compute_sha256_bytes(data)
        existing = db.get_document_by_sha256(sha)
        if existing:
            # Return existing record marked as duplicate
            results.append({
                "id": existing["id"],
                "filename": existing["filename"],
                "ext": existing["ext"],
                "path": existing["path"],
                "size_bytes": existing["size_bytes"],
                "sha256": existing["sha256"],
                "status": "DUPLICATE",
                "created_at": existing["created_at"],
            })
            continue

        doc_id = new_id("doc")
        safe_name = safe_filename(name)
        save_path = UPLOADS_PATH / f"{doc_id}_{safe_name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as out:
            out.write(data)
        rec = {
            "id": doc_id,
            "filename": safe_name,
            "ext": ext,
            "path": str(save_path),
            "size_bytes": len(data),
            "sha256": sha,
            "status": "PENDING",
            "created_at": now_iso(),
        }
        db.upsert_document(rec)
        results.append(rec)
    return JSONResponse(results)


@app.post("/process")
async def process(doc_ids: Optional[List[str]] = Form(None)) -> JSONResponse:
    job_id = start_processing(doc_ids)
    st = get_status(job_id)
    return JSONResponse({"job_id": job_id, "status": st.state if st else "queued"})


@app.get("/status")
def status(job_id: str = Query(...)) -> JSONResponse:
    st = get_status(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({
        "job_id": st.job_id,
        "state": st.state,
        "queued_docs": st.queued_docs,
        "processed_docs": st.processed_docs,
        "total_chunks": st.total_chunks,
        "embedded_chunks": st.embedded_chunks,
        "error": st.error,
    })


@app.get("/documents")
def documents() -> JSONResponse:
    docs = db.list_documents_with_counts()
    return JSONResponse(docs)


@app.post("/ask")
async def ask(payload: dict) -> JSONResponse:
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    k = int(payload.get("k", 5))
    retrieved = retrieve(query, k=k)
    retrieved_dicts = [
        {
            "document_id": r.document_id,
            "filename": r.filename,
            "chunk_id": r.chunk_id,
            "page": r.page,
            "text": r.text,
            "score": r.score,
        }
        for r in retrieved
    ]
    gen = generate_answer(query, retrieved_dicts)
    return JSONResponse({
        "answer": gen.answer,
        "citations": [c.__dict__ for c in gen.citations],
        "retrieved": retrieved_dicts,
    })


@app.post("/reindex")
def reindex() -> JSONResponse:
    rows = db.all_chunks_with_embeddings()
    if not rows:
        return JSONResponse({"ok": True, "indexed": 0})
    vectors = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    index = build_faiss_index(vectors)
    meta = {"model": EMBED_MODEL, "dim": str(vectors.shape[1])}
    save_index(index, meta)
    return JSONResponse({"ok": True, "indexed": len(rows)})


@app.get("/chunk")
def get_chunk(document_id: str, chunk_id: int) -> JSONResponse:
    r = db.find_chunk(document_id, int(chunk_id))
    if not r:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return JSONResponse({
        "id": r["id"],
        "document_id": r["document_id"],
        "chunk_id": r["chunk_id"],
        "text": r["text"],
        "page": r["page"],
    })


if __name__ == "__main__":
    uvicorn.run("src.web:app", host=HOST, port=PORT, reload=False)

