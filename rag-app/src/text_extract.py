from pathlib import Path
from typing import List, Tuple, Iterable

import pdfplumber
from docx import Document as DocxDocument


def extract_pdf(path: Path) -> Tuple[str, List[int]]:
    texts: List[str] = []
    pages: List[int] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            texts.append(text)
            pages.append(i)
    full_text = "\n\n".join(texts)
    return full_text, pages


def extract_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    results: List[Tuple[int, str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            results.append((i, text))
    return results


def extract_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)


def extract_txt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def extract_md(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def extract_text_and_pages(path: Path, ext: str) -> Tuple[str, List[int]]:
    ext = ext.lower().lstrip(".")
    if ext == "pdf":
        text, pages = extract_pdf(path)
        return text, pages
    elif ext == "docx":
        return extract_docx(path), []
    elif ext in {"txt", "md"}:
        return extract_txt(path) if ext == "txt" else extract_md(path), []
    else:
        raise ValueError(f"Unsupported extension: {ext}")

