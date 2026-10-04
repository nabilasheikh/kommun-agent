"""Lesson 7: Measure retrieval, separately from the answer.

    python learn/07_measure_retrieval.py            # BM25 vs embeddings vs hybrid
    python learn/07_measure_retrieval.py --rerank   # + LLM reranking (one LLM call per question)

If the right page never reaches the model, no prompt can fix the answer. So measure the
retrieval step on its own, with labelled questions (learn/retrieval_set.yaml).
"""
import argparse
import time
from pathlib import Path

import yaml

from _common import EmbeddingIndex, HybridIndex, h1, load_chunks, mean, rerank
from kommun_agent.retrieval import Retriever

ap = argparse.ArgumentParser()
ap.add_argument("--rerank", action="store_true")
ap.add_argument("-k", type=int, default=5)
args = ap.parse_args()

chunks = load_chunks()
cases = yaml.safe_load(Path("learn/retrieval_set.yaml").read_text(encoding="utf-8"))
urls = {c["url"] for c in chunks}
missing = [c for c in cases if not any(c["expect"] in u for u in urls)]
for c in missing:
    print(f"  WARNING: no indexed page matches {c['expect']!r} ({c['q']!r}): skipped")
cases = [c for c in cases if c not in missing]

bm25 = Retriever(chunks)
dense = EmbeddingIndex(chunks)
hybrid = HybridIndex(chunks, dense.embedder)
hybrid.dense = dense  # reuse vectors
methods = {
    "BM25": lambda q: bm25.search(q, k=20, per_page=20),
    "Embeddings": lambda q: dense.search(q, k=20),
    "Hybrid (RRF)": lambda q: hybrid.search(q, k=20),
}
if args.rerank:
    from kommun_agent.providers import get_llm
    llm = get_llm()
    methods["Hybrid + rerank"] = lambda q: rerank(llm, q, hybrid.search(q, k=20), top=20)


def first_hit_rank(hits, expect):
    for rank, h in enumerate(hits, 1):
        if expect in h["url"]:
            return rank
    return None


h1(f"Retrieval quality on {len(cases)} labelled questions")
rows = []
for name, search in methods.items():
    ranks, start = [], time.perf_counter()
    for c in cases:
        ranks.append(first_hit_rank(search(c["q"]), c["expect"]))
    ms = (time.perf_counter() - start) / max(len(cases), 1) * 1000
    rows.append((name, mean([r == 1 for r in ranks]), mean([r is not None and r <= args.k for r in ranks]),
                 mean([1 / r if r else 0 for r in ranks]), ms, ranks))

print(f"\n| Method | Hit@1 | Hit@{args.k} | MRR | ms/query |")
print("| --- | --- | --- | --- | --- |")
for name, h1_, hk, mrr, ms, _ in rows:
    print(f"| {name} | {h1_:.0%} | {hk:.0%} | {mrr:.2f} | {ms:.0f} |")

print("\nPer question (rank of the first correct page, - = not in top 20):")
print(f"  {'question':<42}" + "".join(f"{n[:12]:>14}" for n, *_ in rows))
for i, c in enumerate(cases):
    print(f"  {c['q'][:42]:<42}" + "".join(f"{(r[5][i] or '-'):>14}" for r in rows))

print(f"""
What the numbers mean
- Hit@1: the right page came first.  Hit@{args.k}: it was somewhere in what the model sees.
- MRR (mean reciprocal rank): 1.0 = always first, 0.5 = typically second, 0 = never found.
- Read the per-question rows: they tell you WHICH kind of question each method fails on.

Exercise (this is your interview story)
1. Add 10 more questions from real resident language, and 3 that should find nothing.
2. Record the table in the README, change ONE thing (chunk size, headings, hybrid weight,
   reranking), re-run, and record again. "Hit@5 went from 58% to 83% when I ..." is exactly
   the kind of evidence a forward deployed engineer brings to a customer.
""")
