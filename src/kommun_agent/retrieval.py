"""Keyword retrieval with BM25.

BM25 needs no embedding API or vector database, so the demo runs with one API key.
Known limitation: Swedish compound words ("förskoleplats" vs "förskola plats") can be
missed. The evals make this visible; see README "Next steps" for the embedding upgrade.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .config import INDEX_PATH

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        corpus = [tokenize(f"{c['title']} {c['text']}") for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "Retriever":
        if not Path(path).exists():
            raise FileNotFoundError(f"No index at {path}. Run `kommun-agent ingest` first.")
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)}
                for i in ranked[:k] if scores[i] > 0]
