"""LLM providers behind one small interface, so the agent doesn't care which model it talks to.

Pick one with LLM_PROVIDER=claude|gemini in .env.

The agent keeps a provider-neutral history:
    {"role": "user", "text": str}
    {"role": "assistant", "text": str, "tool_calls": [ToolCall], "raw": <provider object or None>}
    {"role": "tool", "results": [{"id", "name", "output"}]}
Each provider translates that into its own request format. "raw" keeps the provider's original
assistant message so nothing is lost on the round trip (Gemini needs its thought signatures back).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None  # provider-native assistant message, replayed on the next request

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLM(Protocol):
    name: str
    model: str

    def chat(self, system: str, history: list[dict], tools: list[dict],
             force_answer: bool = False) -> LLMResponse: ...  # force_answer: text reply only, no tools

    def complete(self, prompt: str, max_tokens: int = 300) -> str: ...


# --------------------------------------------------------------------------- Claude

class ClaudeLLM:
    name = "claude"

    def __init__(self, model: str | None = None, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    @staticmethod
    def to_messages(history: list[dict]) -> list[dict]:
        messages = []
        for turn in history:
            if turn["role"] == "user":
                messages.append({"role": "user", "content": turn["text"]})
            elif turn["role"] == "assistant":
                if turn.get("raw") is not None:
                    content = turn["raw"]
                else:
                    content = []
                    if turn.get("text"):
                        content.append({"type": "text", "text": turn["text"]})
                    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
                                for c in turn.get("tool_calls", [])]
                messages.append({"role": "assistant", "content": content})
            elif turn["role"] == "tool":
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["output"]}
                    for r in turn["results"]]})
        return messages

    def chat(self, system: str, history: list[dict], tools: list[dict],
             force_answer: bool = False) -> LLMResponse:
        extra = {"tool_choice": {"type": "none"}} if force_answer else {}
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, system=system,
            tools=tools, messages=self.to_messages(history), **extra)
        return LLMResponse(
            text="".join(b.text for b in resp.content if b.type == "text").strip(),
            tool_calls=[ToolCall(b.id, b.name, dict(b.input))
                        for b in resp.content if b.type == "tool_use"],
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw=resp.content,
        )

    def complete(self, prompt: str, max_tokens: int = 300) -> str:
        resp = self.client.messages.create(model=self.model, max_tokens=max_tokens,
                                           messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content if b.type == "text")


# --------------------------------------------------------------------------- Gemini

class GeminiLLM:
    name = "gemini"

    def __init__(self, model: str | None = None, client=None):
        if client is None:
            from google import genai
            client = genai.Client()  # reads GEMINI_API_KEY
        self.client = client
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-latest")

    @staticmethod
    def to_contents(history: list[dict]) -> list:
        from google.genai import types
        contents = []
        for turn in history:
            if turn["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=turn["text"])]))
            elif turn["role"] == "assistant":
                if turn.get("raw") is not None:
                    contents.append(turn["raw"])  # keeps thought signatures intact
                    continue
                parts = [types.Part.from_text(text=turn["text"])] if turn.get("text") else []
                parts += [types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.input))
                          for c in turn.get("tool_calls", [])]
                contents.append(types.Content(role="model", parts=parts))
            elif turn["role"] == "tool":
                contents.append(types.Content(role="user", parts=[
                    types.Part(function_response=types.FunctionResponse(
                        id=r["id"], name=r["name"], response={"result": r["output"]}))
                    for r in turn["results"]]))
        return contents

    @staticmethod
    def to_tools(tools: list[dict]) -> list:
        from google.genai import types
        return [types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"],
                                      parameters_json_schema=t["input_schema"])
            for t in tools])]

    def _config(self, system: str | None, tools: list[dict] | None, max_tokens: int,
                force_answer: bool = False):
        from google.genai import types
        tool_config = None
        if tools and force_answer:
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE"))
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=self.to_tools(tools) if tools else None,
            tool_config=tool_config,
            max_output_tokens=max_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    @staticmethod
    def _usage(resp) -> tuple[int, int]:
        u = resp.usage_metadata
        if u is None:
            return 0, 0
        return (u.prompt_token_count or 0,
                (u.candidates_token_count or 0) + (u.thoughts_token_count or 0))

    def chat(self, system: str, history: list[dict], tools: list[dict],
             force_answer: bool = False) -> LLMResponse:
        # Thinking models spend output tokens on reasoning, so allow more room than Claude needs.
        resp = self.client.models.generate_content(
            model=self.model, contents=self.to_contents(history),
            config=self._config(system, tools, max_tokens=4096, force_answer=force_answer))
        content = resp.candidates[0].content if resp.candidates else None
        parts = (content.parts or []) if content else []
        calls = [ToolCall(p.function_call.id or f"call_{i}", p.function_call.name,
                          dict(p.function_call.args or {}))
                 for i, p in enumerate(parts) if p.function_call]
        text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False)).strip()
        tokens_in, tokens_out = self._usage(resp)
        return LLMResponse(text=text, tool_calls=calls, input_tokens=tokens_in,
                           output_tokens=tokens_out, raw=content)

    def complete(self, prompt: str, max_tokens: int = 300) -> str:
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt, config=self._config(None, None, max_tokens * 4))
        return resp.text or ""


# --------------------------------------------------------------------------- factory

PROVIDERS = {"claude": ClaudeLLM, "gemini": GeminiLLM}


def get_llm(provider: str | None = None) -> LLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "claude")).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    return PROVIDERS[provider]()
