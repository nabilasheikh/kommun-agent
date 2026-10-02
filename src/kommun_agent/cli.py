"""Command line: `kommun-agent ingest | ask "..." | chat | serve`."""
from __future__ import annotations

import argparse
import sys

from .config import load_config


def _agent():
    from .agent import KommunAgent
    from .retrieval import Retriever
    return KommunAgent(load_config(), Retriever.load())


def _print(result) -> None:
    print(f"\n{result.answer}\n")
    calls = ", ".join(f"{c['name']}({next(iter(c['input'].values()), '')!r})" for c in result.tool_calls)
    print(f"  tools: {calls or 'none'} | {result.steps} steps | "
          f"{result.input_tokens}+{result.output_tokens} tokens | {result.latency_s}s\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kommun-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="crawl the municipality's site and build the index")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    sub.add_parser("chat", help="interactive chat in the terminal")
    serve = sub.add_parser("serve", help="run the web UI and API")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from .ingest import run
        run(load_config())
    elif args.cmd == "ask":
        _print(_agent().ask(args.question))
    elif args.cmd == "chat":
        agent, history = _agent(), []
        print(f"Chatting with the {agent.cfg.name} assistant. Ctrl+C to quit.")
        try:
            while True:
                q = input("you> ").strip()
                if not q:
                    continue
                r = agent.ask(q, history)
                _print(r)
                history += [{"role": "user", "content": q}, {"role": "assistant", "content": r.answer}]
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("kommun_agent.api:app", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
