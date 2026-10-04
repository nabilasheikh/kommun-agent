"""Lesson 5: Reranking. Retrieve 20 cheaply, let a smarter model pick the best 5.

    python learn/05_rerank.py "Behöver jag bygglov för en altan?"

Uses your LLM_PROVIDER (Gemini or Claude) as the reranker: one call per question.
"""
import sys

from _common import HybridIndex, h1, h2, load_chunks, rerank, show_hits
from kommun_agent.providers import get_llm

question = " ".join(sys.argv[1:]) or "Behöver jag bygglov för att bygga en altan?"
chunks = load_chunks()
hybrid = HybridIndex(chunks)
llm = get_llm()

h1(question)
candidates = hybrid.search(question, k=20)
h2("Hybrid retrieval: top 5 of 20 candidates")
show_hits(candidates, n=5)
h2(f"After reranking with {llm.name}:{llm.model} (score 0-10)")
show_hits(rerank(llm, question, candidates), n=5)

print("""
What to take away
- Retrieval models compare the question and each chunk SEPARATELY (fast, but shallow).
  A reranker reads question + chunk TOGETHER, so it can tell "mentions altan" from
  "answers whether an altan needs bygglov".
- Pattern: retrieve wide (20-50) cheaply -> rerank -> send the best 3-5 to the model.
  Fewer, better passages = cheaper prompts and fewer distracted answers.
- Dedicated rerankers (Cohere Rerank, Voyage, bge-reranker) are faster and cheaper than an LLM;
  an LLM reranker is the easiest way to see the idea.

Exercise
1. Count tokens: how many fewer prompt tokens does the agent use if search returns the
   reranked top 3 instead of BM25's top 5? (Look at input_tokens in logs/traces.jsonl.)
2. What does reranking cost in latency? Time it. When would you NOT rerank?
""")
