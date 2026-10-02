# kommun-agent

An AI agent that answers residents' questions about Swedish municipal services, using only
the municipality's own web pages, and hands cases to a human when it should.

Point it at any kommun's website with one config file, and run it on **Claude or Gemini**
(one setting in `.env`).

> **Demo:** _add a GIF or 60-second video here_
> **Eval results:** _add your latest score here, e.g. "11/12 cases pass, groundedness 4.6/5"_

## What it does

- **Answers with sources.** Every factual answer links to the kommun pages it came from.
- **Says "I don't know".** If the pages don't cover it, it says so instead of guessing fees, dates or rules.
- **Takes action.** When a resident asks about their own case, an appeal or a complaint, it creates a
  service request and returns a ticket id.
- **Handles the tricky cases safely:** emergencies (points to 112), off-topic questions, prompt
  injection, and residents who paste their personnummer.
- **Replies in the resident's language** (Swedish, English, others).
- **Is measurable.** An eval suite with deterministic checks plus an optional LLM-as-judge groundedness
  score, and a trace log of every conversation (tools, tokens, latency).

## Architecture

```mermaid
flowchart LR
    R[Resident] --> UI[Chat page / API]
    UI --> A[Agent loop + tools]
    A <--> M{{Claude or Gemini}}
    A -->|search_kommun_info| S[(BM25 index)]
    A -->|create_service_request| T[(Tickets)]
    A --> L[(Traces)]
    C[Crawler] --> S
    E[Evals] --> A
```

More detail and the design trade-offs: [docs/architecture.md](docs/architecture.md).

## Quick start

Requires Python 3.10+ and an API key for either
[Gemini](https://aistudio.google.com/apikey) (has a free tier) or [Claude](https://console.anthropic.com/).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # set LLM_PROVIDER and add that provider's key

# 1. Choose a municipality: edit config/kommun.yaml (Södertälje by default)
# 2. Crawl its public pages and build the index (~150 pages, a few minutes)
kommun-agent ingest

# 3. Ask
kommun-agent ask "Hur ansöker jag om plats i förskola?"
kommun-agent chat             # multi-turn in the terminal
kommun-agent serve            # web chat at http://localhost:8000
```

With Docker:

```bash
docker build -t kommun-agent .
docker run --env-file .env -v "$PWD/data:/app/data" kommun-agent kommun-agent ingest
docker run --env-file .env -v "$PWD/data:/app/data" -p 8000:8000 kommun-agent
```

## Evals

```bash
python evals/run_evals.py           # 12 cases, deterministic checks
python evals/run_evals.py --judge   # + groundedness score 1-5 from an LLM judge
```

Each case in [evals/questions.yaml](evals/questions.yaml) checks things like: did it cite a page on
the kommun's domain, did it escalate when it should (and only then), did it mention 112 in an
emergency, did it refuse to reveal its prompt, did it avoid repeating a personnummer.
Results go to `evals/results/latest.md`.

**Compare providers** on the same cases:

```bash
LLM_PROVIDER=gemini python evals/run_evals.py --judge
LLM_PROVIDER=claude python evals/run_evals.py --judge
```

| Provider · model | Passed | Groundedness | Avg tokens | Avg latency |
| --- | --- | --- | --- | --- |
| _gemini · ..._ | _x/12_ | _x/5_ | _..._ | _...s_ |
| _claude · ..._ | _x/12_ | _x/5_ | _..._ | _...s_ |

**Prompt-failure log** (fill this in as you iterate; interviewers ask about it):

| Version | What failed | Why | Change | Result |
| --- | --- | --- | --- | --- |
| v1 | Crawler indexed almost nothing on some sites | `urllib`'s robots.txt reader sends its own User-Agent; the site returned 403 and the parser treated that as "disallow everything" | Fetch robots.txt with our own client and User-Agent, parse it ourselves | Pages crawled again |
| v1 | Index had the wrong pages: 150-page cap filled breadth-first with politics, notice board and archive pages; only 8 "bo och bygga" pages, no application pages | No scoping of the crawl | Seed URLs + `allowed_path_prefixes` for the 8 service sections, cap raised to 400 | _re-run evals and fill in_ |
| v1 → v2 | "Hur ansöker jag om förskoleplats?" → Gemini searched 6 times, then "Sorry, I couldn't complete that" (33k tokens, 8 s) | Nothing forced the model to stop searching, and weak results made it keep rephrasing | Max 3 searches in the system prompt; on the last step tools are switched off (`tool_choice: none` / Gemini `mode: NONE`) so the model must answer with what it has | Always returns an answer or an honest "not found" |
| v1 → v2 | Search ranked the meals page above förskola pages for "förskoleplats"; "forskola" typed without å/ä/ö matched nothing | BM25 can't see inside Swedish compound words; no diacritic folding; one page filled all 5 results | Fold å/ä/ö; expand unknown compound words to known stems (`forskoleplats` → `forskola…`, `plats`); title + URL words weighted; max 2 results per page | Förskola pages went from outside the top 5 to positions 1–2 (same index) |

## Project layout

```
src/kommun_agent/
  ingest.py      polite crawler (robots.txt, one domain, rate-limited) + chunking
  retrieval.py   BM25 search over chunks
  tools.py       tool schemas + execution (search, create service request)
  providers.py   Claude and Gemini adapters behind one interface
  agent.py       provider-neutral tool-use loop, system prompt, tracing
  api.py         FastAPI: /api/chat, /api/info, /health, web page
  cli.py         ingest | ask | chat | serve
evals/           eval cases + runner
tests/           offline unit tests (fake LLM, no API key needed)
```

## Privacy

This is a demo. Tickets and traces are stored as local files. The agent is told never to ask for
a personnummer or sensitive data, and an eval checks that it doesn't repeat one. A production
version would add PII redaction on input and in logs, data retention rules, and a DPIA (GDPR).

## Next steps

- [ ] Hybrid retrieval: embeddings (pgvector) + BM25 (would replace the hand-made compound-word expansion)
- [ ] Observability dashboard over `logs/traces.jsonl`: resolution rate, escalations, cost, unanswered questions
- [ ] Load open data (SCB, Skolverket) into Postgres and expose it as a typed tool
- [ ] Deploy to AWS (App Runner or ECS) with the evals running in CI
- [ ] Salesforce Agentforce version: same tools as Agentforce actions calling this API
