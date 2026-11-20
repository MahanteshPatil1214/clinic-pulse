"""TF-IDF policy retrieval with section-level citations and a refusal threshold."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "docs" / "policies"
SCORE_THRESHOLD = 0.12


def _metadata(text: str) -> dict[str, str]:
    """Extract version and effective date from a policy document."""
    version = re.search(r"^Version:\s*(.+)$", text, flags=re.MULTILINE)
    effective = re.search(r"^Effective date:\s*(.+)$", text, flags=re.MULTILINE)
    return {
        "version": version.group(1).strip() if version else "unknown",
        "effective_date": effective.group(1).strip() if effective else "unknown",
    }


def _chunk_document(path: Path) -> list[dict[str, str]]:
    """Split one markdown document into heading-based chunks."""
    text = path.read_text(encoding="utf-8")
    meta = _metadata(text)
    chunks: list[dict[str, str]] = []
    section = "Overview"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if body:
                chunks.append(_make_chunk(path.name, section, meta, body))
            section = line[3:].strip()
            body = []
        elif not line.startswith("# ") and line.strip():
            body.append(line.strip())
    if body:
        chunks.append(_make_chunk(path.name, section, meta, body))
    return chunks


def _make_chunk(
    source_file: str, section: str, meta: dict[str, str], body: list[str]
) -> dict[str, str]:
    """Build one citation-ready chunk."""
    return {
        "source_file": source_file,
        "section": section,
        "version": meta["version"],
        "effective_date": meta["effective_date"],
        "text": " ".join(body),
    }


@lru_cache(maxsize=1)
def _index() -> tuple[TfidfVectorizer, Any, list[dict[str, str]]]:
    """Build the TF-IDF index once for all policy chunks."""
    chunks: list[dict[str, str]] = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        chunks.extend(_chunk_document(path))
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return vectorizer, matrix, chunks


def search(query: str, top_k: int = 3) -> dict[str, Any]:
    """Return the best policy chunks, or flag insufficient evidence."""
    if not isinstance(query, str) or not query.strip():
        return {"results": [], "insufficient_evidence": True}
    vectorizer, matrix, chunks = _index()
    if not chunks:
        return {"results": [], "insufficient_evidence": True}
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    results = []
    for index in ranked:
        score = float(scores[index])
        if score < SCORE_THRESHOLD:
            continue
        chunk = chunks[index]
        results.append(
            {
                **chunk,
                "score": round(score, 3),
            }
        )
    return {
        "results": results,
        "insufficient_evidence": len(results) == 0,
    }


if __name__ == "__main__":
    for question in [
        "What discounts can I offer by tier?",
        "What can I offer a clinic with falling orders?",
        "What are the talking points for diagnostics?",
        "What is the weather forecast tomorrow?",
    ]:
        print(question)
        print(search(question, top_k=2))
