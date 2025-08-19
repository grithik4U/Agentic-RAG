from typing import Iterable, List, Tuple

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .utils import chunk_text, normalize_whitespace


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[int, str]]:
    text = normalize_whitespace(text)
    return list(chunk_text(text, chunk_size, overlap))

