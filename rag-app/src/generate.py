from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from openai import OpenAI

from .config import CONFIDENCE_THRESHOLD, GENERATE_MODEL, OPENAI_API_KEY
from .prompts import SYSTEM_PROMPT_STRICT


@dataclass
class Citation:
    filename: str
    chunk_id: int
    page: Optional[int]


@dataclass
class GenerateResult:
    answer: str
    citations: List[Citation]


def _format_context(chunks: List[Dict]) -> str:
    lines: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        tag = f"{ch['filename']}#{ch['chunk_id']}"
        if ch.get("page"):
            tag += f" (p. {ch['page']})"
        lines.append(f"[{i}] {tag}:\n{ch['text']}")
    return "\n\n".join(lines)


def _should_say_idk(retrieved: List[Dict]) -> bool:
    if not retrieved:
        return True
    top_score = max(r.get("score", 0.0) for r in retrieved)
    return top_score < CONFIDENCE_THRESHOLD


def generate_answer(query: str, retrieved: List[Dict]) -> GenerateResult:
    if _should_say_idk(retrieved):
        return GenerateResult(
            answer="I don't know. The retrieved context is insufficient or too low-confidence to answer.",
            citations=[],
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    context = _format_context(retrieved)
    user_prompt = (
        "Answer the user's question using ONLY the context. "
        "Cite the evidence numerically like [1], [2] where appropriate.\n\n"
        f"Question: {query}\n\nContext:\n{context}"
    )

    resp = client.chat.completions.create(
        model=GENERATE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_STRICT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    answer = resp.choices[0].message.content.strip()

    citations: List[Citation] = []
    seen = set()
    for ch in retrieved:
        key = (ch["filename"], ch["chunk_id"], ch.get("page"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(filename=ch["filename"], chunk_id=ch["chunk_id"], page=ch.get("page")))

    return GenerateResult(answer=answer, citations=citations)

