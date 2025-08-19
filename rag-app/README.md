## RAG Q&A App (FastAPI + FAISS + SQLite)

Minimal Retrieval-Augmented Generation app with uploads, processing, retrieval, and grounded answers with exact citations.

### Features
- Upload PDFs, DOCX, TXT, MD (drag-and-drop + button)
- SHA-256 deduplication (20MB max per file)
- Text extraction (pdfplumber, python-docx, plain read)
- Chunking with overlap
- OpenAI embeddings (cached on disk and in DB)
- FAISS index with manifest
- Ask questions; answers grounded in retrieved chunks with citations
- Simple in-process worker for processing jobs
- CLI via Typer

### Project Structure
```
rag-app/
  README.md
  requirements.txt
  .env.example
  data/
    uploads/
    index/
    cache/embeddings/
  src/
    __init__.py
    config.py
    db.py
    text_extract.py
    chunk.py
    embed_index.py
    retrieve.py
    generate.py
    web.py
    worker.py
    cli.py
    prompts.py
    utils.py
  templates/
    index.html
  questions.json
```

### Requirements
- Python 3.10+
- `OPENAI_API_KEY` in environment

### Replit secrets
- Add a Secret named `OPENAI_API_KEY` with your API key.

### Setup
```
python3 -m venv .venv  # if python venv not available, install python3-venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional; ensure OPENAI_API_KEY is set in your env
```

### Running the web app
```
python -m src.web
```
Then open: http://127.0.0.1:8000

### Upload → Process → Ask flow
1. Open the web app.
2. Drag-and-drop or choose 4–5 files (PDF/DOCX/TXT/MD) under 20MB.
3. Click "Process files". This starts a background job that extracts text, chunks, embeds, and updates the FAISS index.
4. When done, ask a question in the Ask box.
5. See the answer, citations (filename#chunk_id (+ page)), and top-k retrieved chunks with scores. Click a citation to expand the exact chunk.

Notes:
- Duplicate uploads (same SHA-256) are deduplicated. The app reuses previously embedded chunks and marks the upload as duplicate in responses.
- Embeddings are cached on disk under `data/cache/embeddings/` and stored in the DB.
- If the retrieved evidence is weak (below a confidence threshold) or no relevant chunks, the app will answer: "I don't know." with a short explanation.

### API Endpoints
- `POST /upload` → multipart form `files[]`
- `POST /process` → process all pending or provided `doc_ids`
- `GET /status?job_id=...` → job status
- `GET /documents` → list documents and status (also includes `chunk_count`)
- `POST /ask` → `{ query: string, k?: number }`
- `POST /reindex` → rebuild FAISS from DB
- `GET /chunk?document_id=...&chunk_id=...` → fetch exact chunk

### CLI
```
python -m src.cli ingest-uploads
python -m src.cli ask "What is in the documents?"
python -m src.cli eval
```

### Config
See `src/config.py` for:
- `EMBED_MODEL`, `GENERATE_MODEL`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `K`, `RERANK_TOP_M`, `CONFIDENCE_THRESHOLD`
- `DB_PATH`, `INDEX_PATH`, `CACHE_PATH`

### Deduplication and caching
- File-level dedup via SHA-256. When a duplicate is uploaded, the existing document record is reused and no re-embedding occurs.
- Chunk embeddings are stored in the DB and cached as `.npy` files under `data/cache/embeddings/` keyed by content hash and model name.

### I don't know threshold
- If there are no chunks or the top similarity is below the configured threshold, the app returns "I don't know".

### Troubleshooting
- Ensure `OPENAI_API_KEY` is set.
- If FAISS manifest mismatches (model/dim), use `POST /reindex` or CLI to rebuild the index.
- PDF text extraction quality depends on the PDF structure; scanned PDFs may not extract well.
- FAISS wheels may not be available on Python 3.13 yet. Prefer Python 3.10–3.12, or install `faiss-cpu` manually. Without FAISS, retrieval will return no results until indexing is available.

### License
MIT

