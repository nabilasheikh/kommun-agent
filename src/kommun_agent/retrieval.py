"""Keyword retrieval with BM25.

BM25 needs no embedding API or vector database, so the demo runs with one API key.
Swedish compound words ("förskoleplats" vs "förskola") are handled with a small query
expansion (see `Retriever.expand`). Embeddings would handle this better; see README "Next steps".
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from rank_bm25 import BM25Okapi

from .config import INDEX_PATH

_TOKEN = re.compile(r"\w+", re.UNICODE)


_FOLD = str.maketrans("åäöéü", "aaoeu")


def tokenize(text: str) -> list[str]:
    """Lowercase and fold å/ä/ö, so 'forskola' typed without Swedish letters still matches."""
    return [t.translate(_FOLD) for t in _TOKEN.findall(text.lower())]


def _url_words(url: str) -> str:
    """'/skola-och-forskola/forskola/ansok/' -> 'skola och forskola forskola ansok'."""
    return " ".join(re.split(r"[/\-_.]+", urlparse(url).path))


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # Title and URL path count twice: they say what a page is about better than body text.
        corpus = [tokenize(f"{c['title']} {c['title']} {_url_words(c['url'])} {_url_words(c['url'])} "
                           f"{c['text']}") for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._vocab = {t for doc in corpus for t in doc}

    def expand(self, tokens: list[str], max_extra: int = 8) -> list[str]:
        """Cheap Swedish compound-word help for words the index doesn't contain.

        'forskoleplats' -> + words sharing its first 7 letters (forskola, forskolan, ...)
                           + a known word it ends with (plats)
        """
        out = list(tokens)
        for tok in tokens:
            if tok in self._vocab or len(tok) < 8:
                continue
            same_stem = sorted(w for w in self._vocab if w[:7] == tok[:7])[:max_extra]
            tail = next((tok[i:] for i in range(3, len(tok) - 3) if tok[i:] in self._vocab), None)
            out += same_stem + ([tail] if tail else [])
        return out

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "Retriever":
        if not Path(path).exists():
            raise FileNotFoundError(f"No index at {path}. Run `kommun-agent ingest` first.")
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def search(self, query: str, k: int = 5, per_page: int = 2) -> list[dict]:
        """Top-k chunks, at most `per_page` from the same URL so one page can't fill the results."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self.expand(tokenize(query)))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits, per_url = [], {}
        for i in ranked:
            if scores[i] <= 0 or len(hits) == k:
                break
            url = self.chunks[i]["url"]
            if per_url.get(url, 0) >= per_page:
                continue
            per_url[url] = per_url.get(url, 0) + 1
            hits.append({**self.chunks[i], "score": round(float(scores[i]), 3)})
        return hits
