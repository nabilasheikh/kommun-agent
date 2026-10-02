"""Offline tests: no network, no API key. The agent loop runs against a fake Claude client."""
import json
from types import SimpleNamespace as NS

from kommun_agent.agent import KommunAgent
from kommun_agent.config import KommunConfig
from kommun_agent.ingest import build_index, chunk_text, extract_text, in_scope
from kommun_agent.retrieval import Retriever
from kommun_agent.tools import ToolRunner

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


class FakeClient:
    """Replays scripted Claude responses: first a tool call, then a final answer."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot
        return self.responses.pop(0)


def _resp(stop, *blocks):
    return NS(stop_reason=stop, content=list(blocks), usage=NS(input_tokens=100, output_tokens=20))


def test_agent_searches_then_answers_with_sources(tmp_path):
    client = FakeClient([
        _resp("tool_use", NS(type="tool_use", id="t1", name="search_kommun_info",
                             input={"query": "förskola ansökan"})),
        _resp("end_turn", NS(type="text", text="Ansök via e-tjänsten.\n\nSources: "
                                                "[Ansök om förskola](https://www.test.se/forskola)")),
    ])
    agent = KommunAgent(CFG, Retriever(build_index(PAGES)), client=client,
                        trace_path=tmp_path / "traces.jsonl")
    result = agent.ask("Hur ansöker jag om förskola?")

    assert result.sources == ["https://www.test.se/forskola"]
    assert [c["name"] for c in result.tool_calls] == ["search_kommun_info"]
    assert "e-tjänsten" in result.contexts[0]
    assert result.steps == 2 and result.input_tokens == 200
    # the tool result was sent back to the model
    last_user_turn = client.calls[1]["messages"][-1]
    assert last_user_turn["content"][0]["type"] == "tool_result"
    assert (tmp_path / "traces.jsonl").exists()


def test_agent_stops_after_max_steps(tmp_path):
    loop = [_resp("tool_use", NS(type="tool_use", id=f"t{i}", name="search_kommun_info",
                                 input={"query": "x"})) for i in range(3)]
    agent = KommunAgent(CFG, Retriever(build_index(PAGES)), client=FakeClient(loop),
                        max_steps=3, trace_path=None)
    assert "couldn't complete" in agent.ask("?").answer
