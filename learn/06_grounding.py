"""Lesson 6: Generation and grounding. The "G" in RAG, and how it goes wrong.

    python learn/06_grounding.py "Vad kostar förskola per månad?"

Asks the same question three ways and checks the citations.
"""
import re
import sys

from _common import HybridIndex, h1, h2, load_chunks, short
from kommun_agent.providers import get_llm

PROMPT = """Answer the resident's question using ONLY the passages below.
If they don't contain the answer, say "I can't find that on the municipality's pages."
Cite sources as [title](url) using only URLs from the passages.

{passages}

Question: {q}"""

question = " ".join(sys.argv[1:]) or "Vad kostar förskola per månad?"
chunks = load_chunks()
hybrid = HybridIndex(chunks)
llm = get_llm()


def passages(hits):
    return "\n\n".join(f"[{h['title']}]({h['url']})\n{h['text']}" for h in hits)


relevant = hybrid.search(question, k=4)
irrelevant = hybrid.search("biblioteket öppettider", k=4)

h1(question)
runs = [
    ("A. No retrieval (model's own knowledge)", question),
    ("B. With retrieved passages", PROMPT.format(passages=passages(relevant), q=question)),
    ("C. With the WRONG passages", PROMPT.format(passages=passages(irrelevant), q=question)),
]
for label, prompt in runs:
    h2(label)
    answer = llm.complete(prompt, max_tokens=400)
    print(short(answer, 700))
    cited = set(re.findall(r"\]\((https?://[^)\s]+)\)", answer))
    allowed = {h["url"] for h in (relevant if label.startswith("B") else irrelevant)}
    if label.startswith("A"):
        print("  -> sounds confident, but nothing to check it against. For a kommun that's a risk:"
              "\n     fees and deadlines differ per municipality and change every year.")
    else:
        bad = cited - allowed
        print(f"  -> cited {len(cited)} URL(s); {len(bad)} not from the passages"
              + (f": {bad}" if bad else " (good)"))

print("""
What to take away
- A: plausible, unverifiable, possibly from another kommun or an old year. This is hallucination.
- B: grounded. Every claim can be traced to a page the resident can open.
- C: the test that matters. A good RAG prompt makes the model say "I can't find that"
  instead of answering anyway. If it answers, your prompt (or model) isn't grounded enough.
- Your agent's system prompt (src/kommun_agent/agent.py) encodes these rules; the eval
  `--judge` flag scores groundedness automatically.

Exercise
1. Remove the "If they don't contain the answer..." line from PROMPT. What happens in C?
2. Write an automatic citation check for the agent: fail the eval if the answer cites a URL that
   never appeared in a search result (hint: AgentResult.contexts holds the search output).
""")
