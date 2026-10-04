"""Lesson 3: Embeddings. Search by meaning instead of by words.

    python learn/03_embeddings.py                 # uses GEMINI_API_KEY from .env
    LEARN_OFFLINE=1 python learn/03_embeddings.py # no key: mechanics only, not semantic

The first run embeds every chunk once (a few minutes on the free tier) and caches the
vectors in data/embeddings.npz.
"""
import sys

import numpy as np

from _common import EmbeddingIndex, get_embedder, h1, h2, load_chunks, show_hits
from kommun_agent.retrieval import Retriever

chunks = load_chunks()
embedder = get_embedder()

h1("1. A sentence becomes a vector")
pairs = [
    ("Hur ansöker jag om förskoleplats?", "Hur söker man plats på dagis?"),
    ("Hur ansöker jag om förskoleplats?", "How do I apply for preschool?"),
    ("Hur ansöker jag om förskoleplats?", "När töms soptunnan?"),
]
vecs = embedder.embed([s for p in pairs for s in p], "RETRIEVAL_QUERY")
print(f"each text -> {vecs.shape[1]} numbers, e.g. {np.round(vecs[0][:5], 3)} ...")
h2("Cosine similarity: 1 = same meaning, ~0 = unrelated")
for i, (a, b) in enumerate(pairs):
    print(f"  {float(vecs[2 * i] @ vecs[2 * i + 1]):.3f}   {a!r}  vs  {b!r}")
print("\n  Same meaning with different words (dagis/förskola, Swedish/English) should still score high.")

h1("2. Dense retrieval vs BM25 on queries keyword search finds hard")
dense = EmbeddingIndex(chunks, embedder)
bm25 = Retriever(chunks)
queries = sys.argv[1:] or [
    "hur söker man plats på dagis",
    "How do I apply for preschool?",
    "min mamma behöver hjälp hemma",
    "vart lämnar jag gamla möbler",
]
for q in queries:
    h2(q)
    print("  BM25:")
    show_hits(bm25.search(q, k=3), n=3, text=False)
    print("  Embeddings:")
    show_hits(dense.search(q, k=3), n=3, text=False)

print("""
What to take away
- An embedding model turns text into a vector so that similar MEANING = nearby vectors.
  It handles synonyms, other languages and paraphrases that BM25 can't.
- Chunks are embedded once at ingest (task RETRIEVAL_DOCUMENT); each question is embedded at
  query time (RETRIEVAL_QUERY). Retrieval = the k chunks with the highest cosine similarity.
- Weak spots: exact names, numbers, codes ("blankett 4711", "Stålhamraskolan"). BM25 is better
  there. That's why production systems combine both: lesson 4.
- At scale you don't compare against every vector: a vector DB (pgvector, Pinecone, Data 360's
  vector search) uses approximate nearest-neighbour indexes (HNSW) for speed.

Exercise
1. Pass your failed queries from lesson 2:  python learn/03_embeddings.py "dagis" "barnomsorg"
2. Find a query where embeddings do WORSE than BM25 (hint: a specific school name).
""")
