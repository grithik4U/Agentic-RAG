import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich import print

from . import db
from .config import K
from .generate import generate_answer
from .retrieve import retrieve
from .worker import get_status, start_processing


app = typer.Typer(add_completion=False)


@app.command("ingest-uploads")
def ingest_uploads() -> None:
    db.init_db()
    job_id = start_processing()
    print(f"[bold green]Started job[/bold green]: {job_id}")
    while True:
        st = get_status(job_id)
        if not st:
            print("Job disappeared")
            break
        print(f"state={st.state} processed={len(st.processed_docs)}/{len(st.queued_docs)} embedded={st.embedded_chunks}")
        if st.state in ("done", "error"):
            if st.error:
                print(f"[red]{st.error}[/red]")
            break
        time.sleep(1)


@app.command("ask")
def ask(question: str, k: int = K) -> None:
    db.init_db()
    retrieved = retrieve(question, k=k)
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
    gen = generate_answer(question, retrieved_dicts)
    print("\n[bold]Answer:[/bold]", gen.answer)
    print("\n[bold]Citations:[/bold]")
    for c in gen.citations[:k]:
        tag = f"{c.filename}#{c.chunk_id}"
        if c.page:
            tag += f" (p. {c.page})"
        print("-", tag)


@app.command("eval")
def eval_cmd(path: Path = Path("questions.json")) -> None:
    db.init_db()
    if not path.exists():
        print(f"No questions at {path}")
        raise typer.Exit(code=1)
    qs = json.loads(path.read_text())
    total = 0
    hits = 0
    for item in qs:
        q = item.get("question", "")
        hints = [h.lower() for h in item.get("doc_hints", [])]
        if not q:
            continue
        total += 1
        print(f"\n[bold cyan]Q:[/bold cyan] {q}")
        ret = retrieve(q)
        ret_dicts = [
            {
                "document_id": r.document_id,
                "filename": r.filename,
                "chunk_id": r.chunk_id,
                "page": r.page,
                "text": r.text,
                "score": r.score,
            }
            for r in ret
        ]
        used_ids = [f"{r['filename']}#{r['chunk_id']}" for r in ret_dicts]
        print("retrieved:", ", ".join(used_ids))
        gen = generate_answer(q, ret_dicts)
        print("answer:", gen.answer)
        print("citations:", ", ".join([f"{c.filename}#{c.chunk_id}" for c in gen.citations]))
        if hints:
            if any(any(h in r["filename"].lower() for h in hints) for r in ret_dicts):
                hits += 1
    if total:
        print(f"\n[bold]Hit-rate:[/bold] {hits}/{total} = {hits/total:.2%}")


if __name__ == "__main__":
    app()

