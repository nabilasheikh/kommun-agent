"""HTTP API + a minimal chat page. Run with `kommun-agent serve`."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import KommunAgent
from .config import load_config
from .retrieval import Retriever

app = FastAPI(title="kommun-agent")
STATIC = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def get_agent() -> KommunAgent:
    return KommunAgent(load_config(), Retriever.load())


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/info")
def info() -> dict:
    agent = get_agent()
    return {"kommun": agent.cfg.name, "model": agent.model, "chunks": len(agent.tools.retriever.chunks)}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    r = get_agent().ask(req.message, [t.model_dump() for t in req.history])
    return {"answer": r.answer, "sources": r.sources, "tickets": r.tickets,
            "tool_calls": [{"name": c["name"], "input": c["input"]} for c in r.tool_calls],
            "latency_s": r.latency_s, "tokens": r.input_tokens + r.output_tokens}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")
