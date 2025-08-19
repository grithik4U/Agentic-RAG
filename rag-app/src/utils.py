import hashlib
import os
import re
import time
import uuid
from pathlib import Path
from typing import Iterable, Tuple


SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = SAFE_FILENAME_RE.sub("_", name)
    return name[:200]


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    sha256 = hashlib.sha256()
    sha256.update(data)
    return sha256.hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int, overlap: int) -> Iterable[Tuple[int, str]]:
    if size <= 0:
        yield (0, text)
        return
    start = 0
    chunk_id = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        yield (chunk_id, text[start:end])
        chunk_id += 1
        if end == n:
            break
        start = max(0, end - overlap)

