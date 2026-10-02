"""Tools the agent can call: one to read, one to act."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import TICKETS_PATH
from .retrieval import Retriever

CATEGORIES = ["school_preschool", "building_permit", "social_services", "waste_water",
              "parking_traffic", "elderly_care", "other"]

TOOLS = [
    {
        "name": "search_kommun_info",
        "description": (
            "Search the municipality's public web pages. Returns numbered passages with "
            "their page title and URL. Use it for any factual question about services, "
            "rules, fees, opening hours or how to apply. Swedish keywords work best."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Search keywords, e.g. 'bygglov altan' or 'ansöka förskola'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_service_request",
        "description": (
            "Hand the case to a human case worker. Use when the resident asks about their own "
            "case or application, needs a decision, wants to complain or appeal, or when the "
            "web pages do not answer the question and the resident wants help. Returns a ticket id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "summary": {"type": "string",
                            "description": "One or two sentences a case worker can act on. No sensitive personal data."},
                "contact_email": {"type": "string",
                                  "description": "Only if the resident gave it in the conversation."},
            },
            "required": ["category", "summary"],
        },
    },
]


class ToolRunner:
    def __init__(self, retriever: Retriever, tickets_path: Path = TICKETS_PATH):
        self.retriever = retriever
        self.tickets_path = Path(tickets_path)

    def run(self, name: str, args: dict) -> str:
        if name == "search_kommun_info":
            return self._search(args.get("query", ""))
        if name == "create_service_request":
            return self._create_ticket(args)
        return f"Error: unknown tool '{name}'."

    def _search(self, query: str) -> str:
        hits = self.retriever.search(query, k=5)
        if not hits:
            return "No matching pages found. Try different (Swedish) keywords."
        return "\n\n".join(f"[{i}] {h['title']}\nURL: {h['url']}\n{h['text']}"
                           for i, h in enumerate(hits, 1))

    def _create_ticket(self, args: dict) -> str:
        ticket = {
            "id": f"REQ-{uuid4().hex[:8].upper()}",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "category": args.get("category", "other"),
            "summary": args.get("summary", ""),
            "contact_email": args.get("contact_email"),
        }
        self.tickets_path.parent.mkdir(parents=True, exist_ok=True)
        with self.tickets_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        return json.dumps({"status": "created", "ticket_id": ticket["id"]})
