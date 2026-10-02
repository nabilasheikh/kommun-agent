"""Settings: which municipality to serve, which model to use, where data lives."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
INDEX_PATH = DATA_DIR / "index.json"
TICKETS_PATH = DATA_DIR / "tickets.jsonl"
TRACES_PATH = LOG_DIR / "traces.jsonl"


@dataclass
class KommunConfig:
    name: str
    base_url: str
    seed_urls: list[str]
    max_pages: int = 100
    allowed_path_prefixes: list[str] = field(default_factory=list)
    escalation_contact: str = "the municipality's contact centre"


def load_config(path: str | Path | None = None) -> KommunConfig:
    path = Path(path or os.getenv("KOMMUN_CONFIG", "config/kommun.yaml"))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return KommunConfig(**data)
