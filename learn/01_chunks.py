"""Lesson 1: Chunking. What does the model actually get to read?

    python learn/01_chunks.py
    python learn/01_chunks.py --size 80 --overlap 20     # try other settings
"""
import argparse
from collections import Counter

from _common import h1, h2, load_chunks, short, show_hits
from kommun_agent.ingest import chunk_text
from kommun_agent.retrieval import Retriever

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=180)
ap.add_argument("--overlap", type=int, default=40)
ap.add_argument("--query", default="ansöka förskoleplats")
args = ap.parse_args()

chunks = load_chunks()
h1("1. What is in your index?")
pages = Counter(c["url"] for c in chunks)
lengths = sorted(len(c["text"].split()) for c in chunks)
print(f"{len(chunks)} chunks from {len(pages)} pages")
print(f"words per chunk: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
print("pages with the most chunks (long pages):")
for url, n in pages.most_common(5):
    print(f"  {n:>3}  /{url.split('/', 3)[-1]}")

h2("One chunk, exactly as the model sees it")
c = chunks[len(chunks) // 3]
print(f"title: {c['title']}\nurl:   {c['url']}\ntext:  {short(c['text'], 600)}")
print("\nAsk yourself: could the model answer a question from ONLY this text?"
      "\nNotice what's missing: headings, which section it came from, links.")

h1(f"2. Re-chunk with size={args.size}, overlap={args.overlap} and compare retrieval")
pages_text: dict[str, dict] = {}
for c in chunks:  # rebuild page texts from chunk 0..n (approximate: overlaps are duplicated)
    pages_text.setdefault(c["url"], {"title": c["title"], "parts": []})["parts"].append(c["text"])
rechunked = []
for url, p in pages_text.items():
    for i, t in enumerate(chunk_text(" ".join(p["parts"]), args.size, args.overlap)):
        rechunked.append({"id": str(len(rechunked)), "url": url, "title": p["title"], "chunk": i, "text": t})

for label, cs in [("current index", chunks), (f"size {args.size}", rechunked)]:
    h2(f"{label}: {len(cs)} chunks, query '{args.query}'")
    show_hits(Retriever(cs).search(args.query), n=3)

print("""
What to take away
- Small chunks = precise matches but missing context ("Ansök senast..." without saying for what).
- Big chunks = more context but diluted scores and more tokens per search result.
- Overlap stops a sentence from being cut in half at a chunk boundary.

Look closely at the chunk above
Many chunks on this site start with share-widget text: "Lyssna på sidan Dela Kontakt × Dela
Twitter Öppna i nytt fönster ...". That boilerplate is in hundreds of chunks: it wastes tokens,
dilutes search scores, and the model has to read past it. Real RAG work is often this unglamorous.

Exercise
0. Clean it: in extract_text() (src/kommun_agent/ingest.py), remove the share/listen widgets
   before reading the text (inspect the page HTML to find their CSS class), re-ingest, re-run.
1. Run with --size 60 and --size 400. Which query results get better or worse?
2. Heading-aware chunking: in src/kommun_agent/ingest.py, extract_text() throws away the
   structure. Change it to keep <h2>/<h3> headings, start a new chunk at each heading, and
   prefix every chunk with "Page title > Heading". Re-ingest and re-run your evals.
   This is one of the cheapest, biggest quality wins in real RAG systems.
""")
