"""Shared helpers for the RAG lessons. Run lessons from the repo root: python learn/01_chunks.py"""
from __future__ import annotations

import hashlib
import json
import re
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # so data/, config/ and .env resolve the same way as the app

from kommun_agent.config import INDEX_PATH, load_config  # noqa: E402
from kommun_agent.retrieval import Retriever, tokenize  # noqa: E402

EMB_PATH = Path("data/embeddings.npz")


# ----------------------------------------------------------------------------- output

def h1(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def h2(text: str) -> None:
    print(f"\n--- {text} " + "-" * max(0, 72 - len(text)))


def short(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def show_hits(hits: list[dict], n: int = 5, text: bool = True) -> None:
    for rank, h in enumerate(hits[:n], 1):
        path = h["url"].split("/", 3)[-1]
        print(f"  {rank}. [{h.get('score', 0):>7.3f}] /{path}")
        if text:
            print(f"     {short(h['text'], 140)}")


def load_chunks() -> list[dict]:
    if not INDEX_PATH.exists():
        sys.exit(f"No index at {INDEX_PATH}. Run `kommun-agent ingest` first.")
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- embeddings

class GeminiEmbedder:
    """Real semantic embeddings from Gemini (free tier works; needs GEMINI_API_KEY)."""
    name = "gemini"

    def __init__(self, model: str | None = None, dim: int = 768):
        from google import genai
        self.client = genai.Client()
        self.model = model or os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
        self.dim = dim

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        """task: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
        from google.genai import types
        out = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            for attempt in range(6):
                try:
                    resp = self.client.models.embed_content(
                        model=self.model, contents=batch,
                        config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim))
                    out += [e.values for e in resp.embeddings]
                    break
                except Exception as exc:  # free tier: back off on rate limits
                    if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc) or attempt == 5:
                        raise
                    wait = 10 * (attempt + 1)
                    print(f"    rate limited, waiting {wait}s ...")
                    time.sleep(wait)
            if len(texts) > 50:
                print(f"    embedded {min(i + 50, len(texts))}/{len(texts)}")
        return normalize(np.array(out, dtype=np.float32))


class HashEmbedder:
    """NOT semantic: a bag-of-words vector, for running the lessons without an API key.
    Use it to see the mechanics; the results will look like keyword search."""
    name = "hash (offline, not semantic)"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                m[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return normalize(np.log1p(m))


def get_embedder():
    if os.getenv("LEARN_OFFLINE") == "1" or not os.getenv("GEMINI_API_KEY"):
        print("  (using the offline hash embedder: set GEMINI_API_KEY for real embeddings)")
        return HashEmbedder()
    return GeminiEmbedder()


def normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class EmbeddingIndex:
    """Dense retrieval: cosine similarity between the question vector and every chunk vector."""

    def __init__(self, chunks: list[dict], embedder=None):
        self.chunks = chunks
        self.embedder = embedder or get_embedder()
        self.vectors = self._load_or_build()

    def _doc_text(self, c: dict) -> str:
        return f"{c['title']}\n{c['text']}"

    def _load_or_build(self) -> np.ndarray:
        key = f"{self.embedder.name}|{len(self.chunks)}|{self.chunks[0]['url'] if self.chunks else ''}"
        if EMB_PATH.exists():
            cached = np.load(EMB_PATH, allow_pickle=False)
            if str(cached["key"]) == key:
                return cached["vectors"]
        print(f"  Embedding {len(self.chunks)} chunks with {self.embedder.name} (cached afterwards) ...")
        vectors = self.embedder.embed([self._doc_text(c) for c in self.chunks], "RETRIEVAL_DOCUMENT")
        EMB_PATH.parent.mkdir(exist_ok=True)
        np.savez(EMB_PATH, key=np.array(key), vectors=vectors)
        return vectors

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.embedder.embed([query], "RETRIEVAL_QUERY")[0]
        sims = self.vectors @ q  # cosine similarity, because all vectors are unit length
        top = np.argsort(-sims)[:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in top]


# ----------------------------------------------------------------------------- hybrid

def rrf(result_lists: list[list[dict]], k: int = 60, top: int = 5) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank).
    Uses only ranks, so BM25 scores (0-30) and cosine scores (0-1) never need to be compared."""
    fused: dict[str, dict] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits, 1):
            key = f"{h['url']}#{h.get('chunk', h.get('id'))}"
            entry = fused.setdefault(key, {**h, "score": 0.0})
            entry["score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:top]


class HybridIndex:
    def __init__(self, chunks: list[dict], embedder=None):
        self.bm25 = Retriever(chunks)
        self.dense = EmbeddingIndex(chunks, embedder)

    def search(self, query: str, k: int = 5, pool: int = 20) -> list[dict]:
        return rrf([self.bm25.search(query, k=pool, per_page=pool), self.dense.search(query, k=pool)], top=k)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


# ----------------------------------------------------------------------------- reranking

RERANK_PROMPT = """Rate how well each passage answers the question, 0-10.
10 = directly answers it, 5 = related but doesn't answer, 0 = unrelated.

Question: {q}

{passages}

Reply with JSON only: {{"scores": [<one integer per passage, in order>]}}"""


def rerank(llm, question: str, hits: list[dict], top: int = 5) -> list[dict]:
    passages = "\n\n".join(f"[{i}] {h['title']}\n{short(h['text'], 700)}" for i, h in enumerate(hits))
    text = llm.complete(RERANK_PROMPT.format(q=question, passages=passages), max_tokens=300)
    try:
        scores = json.loads(re.search(r"\{.*\}", text, re.S).group(0))["scores"]
    except (AttributeError, ValueError, KeyError):
        print(f"  (couldn't parse reranker output: {text[:120]!r})")
        return hits[:top]
    ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)
    return [{**h, "score": float(s)} for h, s in ranked[:top]]


