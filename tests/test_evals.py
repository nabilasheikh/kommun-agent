"""Offline checks for the eval harness: the case file is well formed and the scoring rules work.

These run in CI without an API key. The live eval (evals/run_evals.py) needs a built index and an LLM key.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

import run_evals

CASES = yaml.safe_load((run_evals.ROOT / "questions.yaml").read_text(encoding="utf-8"))
ALLOWED_KEYS = {"id", "question", "must_cite", "keywords_any", "keywords_none", "escalate", "max_words"}
DOMAIN = "www.test.se"


def result(answer="", sources=(), tickets=()):
    return SimpleNamespace(answer=answer, sources=list(sources), tickets=list(tickets))


def test_case_file_is_well_formed():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in CASES:
        assert set(case) <= ALLOWED_KEYS, f"{case['id']} has unknown keys: {set(case) - ALLOWED_KEYS}"
        assert case["question"].strip(), f"{case['id']} has an empty question"
        assert any(k in case for k in ("must_cite", "escalate", "keywords_any", "keywords_none", "max_words")), (
            f"{case['id']} checks nothing")


def test_passing_answer_has_no_failures():
    case = {"must_cite": True, "keywords_any": ["förskola"], "escalate": False, "max_words": 50}
    r = result("Ansök om förskola via e-tjänsten.", sources=[f"https://{DOMAIN}/forskola"])
    assert run_evals.check(case, r, DOMAIN) == []


def test_missing_kommun_source_fails():
    case = {"must_cite": True}
    r = result("Svar.", sources=["https://example.com/other"])
    assert run_evals.check(case, r, DOMAIN) == ["no source on kommun domain"]


def test_missing_keyword_fails():
    case = {"keywords_any": ["bygglov", "altan"]}
    assert run_evals.check(case, result("Det beror på."), DOMAIN) == [
        "missing any of ['bygglov', 'altan']"]


def test_forbidden_keyword_fails():
    case = {"keywords_none": ["personnummer"]}
    assert run_evals.check(case, result("Ange ditt personnummer."), DOMAIN) == [
        "contains forbidden 'personnummer'"]


def test_escalation_must_match_expectation():
    assert run_evals.check({"escalate": True}, result(tickets=[]), DOMAIN) == [
        "escalate expected True, got False"]
    assert run_evals.check({"escalate": False}, result(tickets=["REQ-1"]), DOMAIN) == [
        "escalate expected False, got True"]


def test_word_limit_fails_when_exceeded():
    case = {"max_words": 3}
    assert run_evals.check(case, result("ett två tre fyra"), DOMAIN) == ["4 words > 3"]
