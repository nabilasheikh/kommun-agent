"""Lesson 2: Keyword search (BM25). Why does a page win?

    python learn/02_bm25.py "hur ansöker jag om förskoleplats"
"""
import math
import sys

from _common import h1, h2, load_chunks, show_hits
from kommun_agent.retrieval import Retriever, tokenize

query = sys.argv[1] if len(sys.argv) > 1 else "hur ansöker jag om förskoleplats"
chunks = load_chunks()
r = Retriever(chunks)
bm = r._bm25  # rank_bm25 object; peeking inside for teaching

h1(f"Query: {query!r}")
tokens = tokenize(query)
expanded = r.expand(tokens)
print(f"tokens:   {tokens}")
print(f"expanded: {[t for t in expanded if t not in tokens]}   (compound-word help)")

h2("How rare is each word? (IDF: rare words count more)")
n_docs = len(chunks)
for t in dict.fromkeys(expanded):
    df = sum(1 for doc_freqs in bm.doc_freqs if t in doc_freqs)
    idf = bm.idf.get(t, 0.0)
    print(f"  {t:<18} in {df:>4}/{n_docs} chunks   idf {idf:6.2f}"
          + ("   <- too common to help" if df > n_docs / 2 else "")
          + ("   <- not in the index at all" if df == 0 else ""))

h2("Top results")
hits = r.search(query, k=5)
show_hits(hits)

h2("Why did #1 win? Score contribution per word")
if hits:
    top_tokens = tokenize(f"{hits[0]['title']} {hits[0]['text']}")
    for t in dict.fromkeys(expanded):
        tf = top_tokens.count(t)
        if tf:
            print(f"  {t:<18} appears {tf}x in the chunk")

print(f"""
What to take away
- BM25 = term frequency (word appears often in the chunk) x IDF (word is rare overall),
  with a cap so repeating a word 50 times doesn't win forever, and a penalty for long chunks.
- It's exact-word matching. "förskoleplats" and "plats på förskola" only meet because of the
  compound expansion we added. "barnomsorg" (childcare) won't match "förskola" at all.
- Words like "hur", "jag", "om" are in almost every chunk, so their IDF is near zero.

Exercise
1. Find 3 queries where BM25 fails: synonyms ("dagis" vs "förskola"), English questions on
   Swedish pages, or a question with no shared words. Write them down: lesson 3 fixes them.
2. Try: python learn/02_bm25.py "barnomsorg"   and   python learn/02_bm25.py "dagis"
""")
