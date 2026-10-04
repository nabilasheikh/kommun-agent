"""Lesson 4: Hybrid search with Reciprocal Rank Fusion (RRF).

    python learn/04_hybrid.py "Stålhamraskolan" "plats på dagis"
"""
import sys

from _common import EmbeddingIndex, h1, h2, load_chunks, rrf, show_hits
from kommun_agent.retrieval import Retriever

chunks = load_chunks()
bm25 = Retriever(chunks)
dense = EmbeddingIndex(chunks)

h1("How RRF combines two ranked lists")
print("""  score(chunk) = 1/(60 + rank in BM25 list) + 1/(60 + rank in embedding list)

  rank 1 in one list only       -> 1/61            = 0.0164
  rank 3 in both lists          -> 1/63 + 1/63     = 0.0317   (agreement wins)
  rank 1 + rank 20              -> 1/61 + 1/80     = 0.0289

  Only ranks are used, so BM25 scores (0-30) and cosine scores (0-1) never need rescaling.""")

queries = sys.argv[1:] or ["Stålhamraskolan", "hur söker man plats på dagis", "bygglov altan"]
for q in queries:
    h1(q)
    b = bm25.search(q, k=20, per_page=20)
    d = dense.search(q, k=20)
    fused = rrf([b, d], top=5)
    for label, hits in [("BM25", b), ("Embeddings", d), ("Hybrid (RRF)", fused)]:
        h2(label)
        show_hits(hits, n=3, text=False)

print("""
What to take away
- Hybrid = run both searches, fuse the rankings. It keeps BM25's precision on names and
  numbers and embeddings' recall on meaning. It's the default in most production RAG.
- k=60 is the standard constant from the original RRF paper; a bigger k flattens the
  difference between rank 1 and rank 10.

Exercise
1. Give BM25 twice the weight (multiply its 1/(k+rank) by 2 in rrf()). When does it help?
2. Promote it: add a `hybrid` option to src/kommun_agent/retrieval.py and a RETRIEVER=bm25|hybrid
   setting, so the agent's search_kommun_info tool can use it. Then run the evals for both.
""")
