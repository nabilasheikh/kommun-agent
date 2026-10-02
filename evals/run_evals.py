"""Run the eval set against the live agent and write a report.

    python evals/run_evals.py            # deterministic checks only
    python evals/run_evals.py --judge    # + LLM-as-judge groundedness score (1-5)
    python evals/run_evals.py --only emergency,off_topic
    LLM_PROVIDER=gemini python evals/run_evals.py   # compare providers

Writes evals/results/latest.md (readable table) and latest.json (full detail).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from kommun_agent.agent import KommunAgent  # noqa: E402
from kommun_agent.config import load_config  # noqa: E402
from kommun_agent.retrieval import Retriever  # noqa: E402

JUDGE_PROMPT = """You are grading a municipal assistant's answer for groundedness.

Question: {question}

Retrieved passages the assistant saw:
<passages>
{contexts}
</passages>

Assistant's answer:
<answer>
{answer}
</answer>

Score 1-5: 5 = every factual claim is supported by the passages (or the assistant correctly \
says it doesn't know); 3 = mostly supported, minor unsupported details; 1 = key claims are \
invented. Reply with JSON only: {{"score": <int>, "reason": "<one sentence>"}}"""


def check(case: dict, result, domain: str) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    fails = []
    answer = result.answer.lower()
    if case.get("must_cite"):
        on_domain = [u for u in result.sources if urlparse(u).netloc == domain]
        if not on_domain:
            fails.append("no source on kommun domain")
    if kws := case.get("keywords_any"):
        if not any(k.lower() in answer for k in kws):
            fails.append(f"missing any of {kws}")
    for k in case.get("keywords_none", []):
        if k.lower() in answer:
            fails.append(f"contains forbidden '{k}'")
    if "escalate" in case:
        escalated = bool(result.tickets)
        if escalated != case["escalate"]:
            fails.append(f"escalate expected {case['escalate']}, got {escalated}")
    if (mx := case.get("max_words")) and len(result.answer.split()) > mx:
        fails.append(f"{len(result.answer.split())} words > {mx}")
    return fails


def judge(agent: KommunAgent, result) -> dict:
    if not result.contexts:
        return {"score": None, "reason": "no retrieval"}
    text = agent.llm.complete(JUDGE_PROMPT.format(
        question=result.question, contexts="\n\n".join(result.contexts)[:20000],
        answer=result.answer))
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(match.group(0)) if match else {"score": None, "reason": text[:200]}
    except json.JSONDecodeError:
        return {"score": None, "reason": text[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--only", help="comma-separated case ids")
    args = ap.parse_args()

    cfg = load_config()
    agent = KommunAgent(cfg, Retriever.load(), trace_path=None)
    domain = urlparse(cfg.base_url).netloc
    cases = yaml.safe_load((ROOT / "questions.yaml").read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    rows = []
    for case in cases:
        print(f"- {case['id']} ...", end=" ", flush=True)
        r = agent.ask(case["question"])
        fails = check(case, r, domain)
        row = {"id": case["id"], "question": case["question"], "passed": not fails, "fails": fails,
               "answer": r.answer, "sources": r.sources, "tickets": r.tickets,
               "tools": [c["name"] for c in r.tool_calls], "steps": r.steps,
               "tokens": r.input_tokens + r.output_tokens, "latency_s": r.latency_s}
        if args.judge:
            row["judge"] = judge(agent, r)
        rows.append(row)
        print("PASS" if not fails else f"FAIL ({'; '.join(fails)})")

    passed = sum(r["passed"] for r in rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"latest-{agent.llm.name}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "latest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    scores = [r["judge"]["score"] for r in rows if r.get("judge", {}).get("score")]
    lines = [f"# Eval results: {cfg.name}", "",
             f"{datetime.now():%Y-%m-%d %H:%M} · `{agent.model}` · "
             f"**{passed}/{len(rows)} passed**"
             + (f" · groundedness {sum(scores) / len(scores):.1f}/5" if scores else ""), "",
             "| Case | Result | Tools | Tokens | Latency | Notes |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        note = "; ".join(r["fails"]) or ""
        if r.get("judge"):
            note += f" judge {r['judge'].get('score')}: {r['judge'].get('reason', '')}"
        lines.append(f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {', '.join(r['tools']) or '-'} "
                     f"| {r['tokens']} | {r['latency_s']}s | {note.strip()} |")
    (out / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{passed}/{len(rows)} passed -> {out / 'latest.md'}")


if __name__ == "__main__":
    main()
