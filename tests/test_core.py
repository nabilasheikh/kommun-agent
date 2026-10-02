"""Offline tests: no network, no API key. The agent loop runs against a fake LLM."""
import json

from kommun_agent.agent import KommunAgent
from kommun_agent.config import KommunConfig
from kommun_agent.ingest import build_index, chunk_text, extract_text, in_scope
from kommun_agent.retrieval import Retriever
from kommun_agent.providers import ClaudeLLM, GeminiLLM, LLMResponse, ToolCall, get_llm
from kommun_agent.tools import TOOLS, ToolRunner

CFG = KommunConfig(name="Testkommun", base_url="https://www.test.se", seed_urls=["https://www.test.se/"])

PAGES = [
    {"url": "https://www.test.se/forskola", "title": "Ansök om förskola",
     "text": "Du ansöker om plats i förskola via e-tjänsten. Ansök senast fyra månader innan."},
    {"url": "https://www.test.se/bygglov", "title": "Bygglov för altan",
     "text": "En altan som är lägre än 1,8 meter kräver oftast inget bygglov."},
    {"url": "https://www.test.se/avfall", "title": "Sophämtning",
     "text": "Hushållsavfall hämtas varannan vecka. Grovavfall lämnas på återvinningscentralen."},
]


def test_extract_text_strips_navigation():
    html = ("<html><head><title>Bygglov</title></head><body><nav>Meny Hem</nav>"
            "<main><h1>Bygglov</h1><p>Så söker du bygglov.</p></main><footer>Kontakt</footer></body></html>")
    title, text = extract_text(html)
    assert title == "Bygglov"
    assert "Så söker du bygglov." in text
    assert "Meny" not in text and "Kontakt" not in text


def test_chunk_text_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_text(" ".join(words), max_words=180, overlap=40)
    assert len(chunks) == 3
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]
    assert chunks[-1].split()[-1] == "w399"


def test_in_scope_stays_on_domain():
    assert in_scope("https://www.test.se/skola", CFG)
    assert not in_scope("https://other.se/skola", CFG)
    assert not in_scope("https://www.test.se/blankett.pdf", CFG)


def test_retriever_ranks_relevant_page_first():
    r = Retriever(build_index(PAGES))
    hits = r.search("bygglov altan")
    assert hits[0]["url"].endswith("/bygglov")
    assert r.search("helikopter") == []


def test_create_ticket_writes_jsonl(tmp_path):
    runner = ToolRunner(Retriever(build_index(PAGES)), tickets_path=tmp_path / "t.jsonl")
    out = json.loads(runner.run("create_service_request", {"category": "other", "summary": "Test"}))
    assert out["ticket_id"].startswith("REQ-")
    assert (tmp_path / "t.jsonl").read_text().count("\n") == 1


class FakeLLM:
    """Replays scripted model responses: e.g. first a tool call, then a final answer."""
    name, model = "fake", "fake-1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, system, history, tools, force_answer=False):
        self.calls.append([dict(t) for t in history])  # snapshot
        self.forced = force_answer
        return self.responses.pop(0)

    def complete(self, prompt, max_tokens=300):
        return '{"score": 5, "reason": "ok"}'


def _tool(name, **args):
    return LLMResponse(text="", tool_calls=[ToolCall("t1", name, args)],
                       input_tokens=100, output_tokens=20)


def _final(text):
    return LLMResponse(text=text, input_tokens=100, output_tokens=20)


def test_agent_searches_then_answers_with_sources(tmp_path):
    llm = FakeLLM([
        _tool("search_kommun_info", query="förskola ansökan"),
        _final("Ansök via e-tjänsten.\n\nSources: [Ansök om förskola](https://www.test.se/forskola)"),
    ])
    agent = KommunAgent(CFG, Retriever(build_index(PAGES)), llm=llm,
                        trace_path=tmp_path / "traces.jsonl")
    result = agent.ask("Hur ansöker jag om förskola?")

    assert result.sources == ["https://www.test.se/forskola"]
    assert [c["name"] for c in result.tool_calls] == ["search_kommun_info"]
    assert "e-tjänsten" in result.contexts[0]
    assert result.steps == 2 and result.input_tokens == 200
    # the tool result was sent back to the model on the second call
    last = llm.calls[1][-1]
    assert last["role"] == "tool" and "e-tjänsten" in last["results"][0]["output"]
    assert (tmp_path / "traces.jsonl").exists()


def test_agent_records_ticket(tmp_path, monkeypatch):
    llm = FakeLLM([_tool("create_service_request", category="other", summary="Check my case"),
                   _final("I've created a request.")])
    agent = KommunAgent(CFG, Retriever(build_index(PAGES)), llm=llm, trace_path=None)
    agent.tools.tickets_path = tmp_path / "t.jsonl"
    assert agent.ask("Check my case").tickets[0].startswith("REQ-")


def test_agent_forces_an_answer_on_last_step():
    llm = FakeLLM([_tool("search_kommun_info", query="x"), _tool("search_kommun_info", query="y"),
                   _final("Based on what I found: apply via the e-service.")])
    agent = KommunAgent(CFG, Retriever(build_index(PAGES)), llm=llm, max_steps=3, trace_path=None)
    result = agent.ask("?")
    assert llm.forced is True
    assert result.answer.startswith("Based on what I found")


def test_retriever_folds_swedish_letters_and_limits_per_page():
    pages = PAGES + [{"url": "https://www.test.se/forskola/ansok", "title": "Ansök om plats",
                      "text": " ".join(["förskola"] * 400)}]
    r = Retriever(build_index(pages))
    hits = r.search("ansoka forskola", k=5)
    assert hits and all("forskola" in h["url"] for h in hits[:2])
    assert max(sum(h["url"] == u for h in hits) for u in {h["url"] for h in hits}) <= 2


# --- provider format adapters (offline: only the request translation is tested) ---

HISTORY = [
    {"role": "user", "text": "Hur söker jag förskola?"},
    {"role": "assistant", "text": "", "tool_calls": [ToolCall("c1", "search_kommun_info", {"query": "förskola"})]},
    {"role": "tool", "results": [{"id": "c1", "name": "search_kommun_info", "output": "[1] Förskola ..."}]},
]


def test_claude_history_format():
    msgs = ClaudeLLM.to_messages(HISTORY)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][0] == {"type": "tool_use", "id": "c1", "name": "search_kommun_info",
                                     "input": {"query": "förskola"}}
    assert msgs[2]["content"][0]["type"] == "tool_result" and msgs[2]["content"][0]["tool_use_id"] == "c1"


def test_gemini_history_and_tool_format():
    contents = GeminiLLM.to_contents(HISTORY)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "search_kommun_info"
    assert contents[2].parts[0].function_response.response == {"result": "[1] Förskola ..."}
    decl = GeminiLLM.to_tools(TOOLS)[0].function_declarations
    assert [d.name for d in decl] == ["search_kommun_info", "create_service_request"]


def test_get_llm_rejects_unknown_provider():
    import pytest
    with pytest.raises(ValueError):
        get_llm("nope")


def test_gemini_parses_function_call_and_usage():
    from types import SimpleNamespace as NS
    from google.genai import types
    content = types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="search_kommun_info", args={"query": "bygglov"}))])
    fake_resp = types.GenerateContentResponse(
        candidates=[types.Candidate(content=content)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=50, candidates_token_count=10, thoughts_token_count=5))
    client = NS(models=NS(generate_content=lambda **kw: fake_resp))
    r = GeminiLLM(model="test", client=client).chat("sys", HISTORY[:1], TOOLS)
    assert r.wants_tools and r.tool_calls[0].input == {"query": "bygglov"}
    assert (r.input_tokens, r.output_tokens) == (50, 15)
    assert r.raw is content  # replayed verbatim next turn (keeps thought signatures)


def test_claude_parses_tool_use():
    from types import SimpleNamespace as NS
    blocks = [NS(type="text", text="Let me check."),
              NS(type="tool_use", id="tu1", name="search_kommun_info", input={"query": "avfall"})]
    resp = NS(content=blocks, usage=NS(input_tokens=40, output_tokens=8))
    client = NS(messages=NS(create=lambda **kw: resp))
    r = ClaudeLLM(model="test", client=client).chat("sys", HISTORY[:1], TOOLS)
    assert r.text == "Let me check." and r.tool_calls[0].id == "tu1"


def test_compound_word_expansion():
    pages = PAGES + [{"url": "https://www.test.se/plats", "title": "Lediga platser",
                      "text": "Det finns plats i flera grupper och verksamheter just nu i kommunen."}]
    r = Retriever(build_index(pages))
    assert "forskola" in r.expand(["forskoleplats"])
    assert r.search("förskoleplats")[0]["url"].endswith("/forskola")
