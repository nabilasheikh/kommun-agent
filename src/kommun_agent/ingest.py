"""Crawl a municipality's public website and build a searchable index of text chunks.

The crawler is deliberately polite: it stays on one domain, respects robots.txt,
identifies itself, and waits between requests.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .config import INDEX_PATH, KommunConfig

USER_AGENT = "kommun-agent/0.1 (open-source portfolio demo)"
SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip",
            ".doc", ".docx", ".xls", ".xlsx", ".mp4", ".mp3", ".ics")
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "aside"]


def extract_text(html: str) -> tuple[str, str]:
    """Return (title, main text) with navigation and boilerplate removed."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    return title, re.sub(r"\n{2,}", "\n", text)


def chunk_text(text: str, max_words: int = 180, overlap: int = 40) -> list[str]:
    """Split text into overlapping word windows so answers that span a boundary survive."""
    words = text.split()
    if not words:
        return []
    step = max_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return chunks


def in_scope(url: str, cfg: KommunConfig) -> bool:
    p, base = urlparse(url), urlparse(cfg.base_url)
    if p.scheme not in ("http", "https") or p.netloc != base.netloc:
        return False
    if p.path.lower().endswith(SKIP_EXT):
        return False
    if cfg.allowed_path_prefixes and not any(p.path.startswith(x) for x in cfg.allowed_path_prefixes):
        return False
    return True


def crawl(cfg: KommunConfig, delay: float = 0.5) -> list[dict]:
    robots = robotparser.RobotFileParser(urljoin(cfg.base_url, "/robots.txt"))
    try:
        robots.read()
    except Exception:  # unreachable robots.txt -> treat as no rules
        robots = None

    seen: set[str] = set()
    queue = deque(cfg.seed_urls)
    pages: list[dict] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True) as client:
        while queue and len(pages) < cfg.max_pages:
            url = urldefrag(queue.popleft())[0]
            if url in seen or not in_scope(url, cfg):
                continue
            seen.add(url)
            if robots and not robots.can_fetch(USER_AGENT, url):
                continue
            try:
                resp = client.get(url)
            except httpx.HTTPError as exc:
                print(f"  skip {url}: {exc}")
                continue
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                continue

            title, text = extract_text(resp.text)
            if len(text.split()) >= 50:
                pages.append({"url": str(resp.url), "title": title, "text": text})
                print(f"  [{len(pages)}/{cfg.max_pages}] {resp.url}")

            for a in BeautifulSoup(resp.text, "html.parser").find_all("a", href=True):
                nxt = urldefrag(urljoin(str(resp.url), a["href"]))[0]
                if nxt not in seen and in_scope(nxt, cfg):
                    queue.append(nxt)
            time.sleep(delay)
    return pages


def build_index(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        for i, chunk in enumerate(chunk_text(page["text"])):
            chunks.append({"id": f"{len(chunks)}", "url": page["url"], "title": page["title"],
                           "chunk": i, "text": chunk})
    return chunks


def save_index(chunks: list[dict], path: Path = INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")


def run(cfg: KommunConfig) -> None:
    print(f"Crawling {cfg.name} ({cfg.base_url}) ...")
    pages = crawl(cfg)
    chunks = build_index(pages)
    save_index(chunks)
    print(f"Indexed {len(pages)} pages into {len(chunks)} chunks -> {INDEX_PATH}")
