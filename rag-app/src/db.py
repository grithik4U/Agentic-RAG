import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import DB_PATH


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT,
        ext TEXT,
        path TEXT,
        size_bytes INT,
        sha256 TEXT UNIQUE,
        status TEXT,
        created_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT,
        chunk_id INT,
        text TEXT,
        page INT,
        embedding BLOB NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """,
]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        cur = conn.cursor()
        for ddl in SCHEMA:
            cur.executescript(ddl)
        conn.commit()


def upsert_document(doc: Dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, filename, ext, path, size_bytes, sha256, status, created_at)
            VALUES (:id, :filename, :ext, :path, :size_bytes, :sha256, :status, :created_at)
            ON CONFLICT(id) DO UPDATE SET
                filename=excluded.filename,
                ext=excluded.ext,
                path=excluded.path,
                size_bytes=excluded.size_bytes,
                sha256=excluded.sha256,
                status=excluded.status
            ;
            """,
            doc,
        )
        conn.commit()


def insert_chunk(chunk: Dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO chunks (id, document_id, chunk_id, text, page, embedding)
            VALUES (:id, :document_id, :chunk_id, :text, :page, :embedding)
            ON CONFLICT(id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_id=excluded.chunk_id,
                text=excluded.text,
                page=excluded.page,
                embedding=excluded.embedding;
            """,
            chunk,
        )
        conn.commit()


def update_document_status(doc_id: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE documents SET status=? WHERE id=?", (status, doc_id))
        conn.commit()


def get_document_by_sha256(sha256: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM documents WHERE sha256=?", (sha256,))
        row = cur.fetchone()
        return row


def get_document(doc_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,))
        row = cur.fetchone()
        return row


def list_documents() -> List[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, filename, ext, path, size_bytes, sha256, status, created_at FROM documents ORDER BY created_at DESC"
        )
        return cur.fetchall()


def list_documents_with_counts() -> List[Dict[str, Any]]:
	with get_conn() as conn:
		cur = conn.execute(
			"""
			SELECT d.id, d.filename, d.ext, d.path, d.size_bytes, d.sha256, d.status, d.created_at,
			       (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
			FROM documents d
			ORDER BY d.created_at DESC
			"""
		)
		rows = cur.fetchall()
		return [dict(r) for r in rows]


def chunks_for_document(doc_id: str) -> List[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, document_id, chunk_id, text, page, embedding FROM chunks WHERE document_id=? ORDER BY chunk_id",
            (doc_id,),
        )
        return cur.fetchall()


def insert_chunks_bulk(rows: Iterable[Dict[str, Any]]) -> None:
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO chunks (id, document_id, chunk_id, text, page, embedding)
            VALUES (:id, :document_id, :chunk_id, :text, :page, :embedding)
            ON CONFLICT(id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_id=excluded.chunk_id,
                text=excluded.text,
                page=excluded.page,
                embedding=COALESCE(excluded.embedding, chunks.embedding);
            """,
            list(rows),
        )
        conn.commit()


def all_chunks_with_embeddings() -> List[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, document_id, chunk_id, text, page, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY document_id, chunk_id"
        )
        return cur.fetchall()


def set_setting(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()


def get_setting(key: str) -> Optional[str]:
    with get_conn() as conn:
        cur = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None


def find_chunk(document_id: str, chunk_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, document_id, chunk_id, text, page, embedding FROM chunks WHERE document_id=? AND chunk_id=?",
            (document_id, chunk_id),
        )
        return cur.fetchone()

