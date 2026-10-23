"""Command-line entry point; the default installation only needs Python."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memoryos", description="Editable temporal AI memory")
    parser.add_argument("--root", default=os.environ.get("MEMORYOS_HOME", "./memory-data"),
                        help="Memory directory (default: MEMORYOS_HOME or ./memory-data)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create a Markdown store and derived SQLite index")
    add = commands.add_parser("add", help="Record an explicitly approved fact")
    add.add_argument("entity")
    add.add_argument("attribute")
    add.add_argument("value")
    add.add_argument("--from", dest="valid_from")
    add.add_argument("--to", dest="valid_to")
    add.add_argument("--source", default="manual")
    add.add_argument("--evidence", default="")
    add.add_argument("--confidence", type=float, default=1.0)
    add.add_argument("--tag", dest="tags", action="append", default=[])
    add.add_argument("--link", dest="links", action="append", default=[])
    add.add_argument("--topic", default="general")
    search = commands.add_parser("search", help="Retrieve facts valid at a point in time")
    search.add_argument("query")
    search.add_argument("--as-of")
    search.add_argument("--known-at")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--entity")
    search.add_argument("--mode", choices=["keyword", "semantic", "graph", "hybrid"],
                        default="hybrid")
    history = commands.add_parser("history", help="Show the effective timeline for an entity")
    history.add_argument("entity")
    history.add_argument("--attribute")
    history.add_argument("--known-at")
    correct = commands.add_parser("correct", help="Correct a fact with an auditable new revision")
    correct.add_argument("id")
    correct.add_argument("value")
    correct.add_argument("--source", default="manual")
    correct.add_argument("--evidence", default="")
    retract = commands.add_parser("retract", help="Record a tombstone; retain the audit history")
    retract.add_argument("id")
    retract.add_argument("--source", default="manual")
    commands.add_parser("reindex", help="Validate Markdown edits and rebuild the SQLite index")
    extract = commands.add_parser("extract", help="Propose updates without committing memory")
    content = extract.add_mutually_exclusive_group(required=True)
    content.add_argument("--file", type=Path)
    content.add_argument("--text")
    extract.add_argument("--source", required=True)
    extract.add_argument("--observed-at")
    extract.add_argument("--topic", default="general")
    proposals = commands.add_parser("proposals", help="Inspect the approval queue")
    proposals.add_argument("--status", choices=["pending", "approved", "rejected", "all"],
                           default="pending")
    for command in ("approve", "reject"):
        item = commands.add_parser(command, help=f"{command.title()} a proposed memory update")
        item.add_argument("id")
    commands.add_parser("demo", help="Add synthetic examples (safe to repeat)")
    serve = commands.add_parser("serve", help="Run the optional HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    commands.add_parser("mcp", help="Run the optional read-only MCP stdio server")
    commands.add_parser("git-init", help="Opt in to Git history in a separate memory directory")
    checkpoint = commands.add_parser("checkpoint", help="Commit memory files to local Git")
    checkpoint.add_argument("--message", required=True)
    git_log = commands.add_parser("git-log", help="Inspect local memory checkpoints")
    git_log.add_argument("--limit", type=int, default=20)
    return parser


def _execute(args: argparse.Namespace) -> Any:
    root = Path(args.root).expanduser().resolve()
    if args.command == "serve":
        try:
            import uvicorn
            from .api import create_app
        except ImportError as exc:
            raise RuntimeError("Install the API extra: pip install 'temporal-memory-os[api]'") from exc
        uvicorn.run(create_app(root), host=args.host, port=args.port)
        return None
    if args.command == "mcp":
        from .mcp_server import run_server
        run_server(root)
        return None
    if args.command in ("git-init", "checkpoint", "git-log"):
        from .git_history import GitHistory
        history = GitHistory(root)
        if args.command == "git-init":
            return history.initialize()
        if args.command == "checkpoint":
            return history.checkpoint(args.message)
        return history.log(args.limit)
    from .store import MemoryStore
    with MemoryStore(root) as store:
        if args.command == "init":
            return {"root": str(root), "facts": len(store.all_facts())}
        if args.command == "add":
            fields = ("entity", "attribute", "value", "valid_from", "valid_to", "source",
                      "evidence", "confidence", "tags", "links", "topic")
            return store.add(**{name: getattr(args, name) for name in fields}).to_dict()
        if args.command == "search":
            from .retrieval import HybridRetriever
            return [result.to_dict() for result in HybridRetriever(store).search(
                args.query, as_of=args.as_of, known_at=args.known_at,
                limit=args.limit, mode=args.mode, entity=args.entity)]
        if args.command == "history":
            return [fact.to_dict() for fact in store.history(
                args.entity, attribute=args.attribute, known_at=args.known_at)]
        if args.command == "correct":
            return store.correct(args.id, args.value, source=args.source,
                                 evidence=args.evidence).to_dict()
        if args.command == "retract":
            return store.retract(args.id, source=args.source).to_dict()
        if args.command == "reindex":
            return {"indexed": store.reindex()}
        if args.command == "demo":
            from .demo import seed_demo
            return seed_demo(store)
        from .extraction import ProposalQueue
        queue = ProposalQueue(store)
        if args.command == "extract":
            text = args.file.read_text(encoding="utf-8") if args.file else args.text
            return [proposal.to_dict() for proposal in queue.extract(
                text, args.source, observed_at=args.observed_at, topic=args.topic)]
        if args.command == "proposals":
            return [proposal.to_dict() for proposal in queue.list(
                status=None if args.status == "all" else args.status)]
        if args.command == "approve":
            return queue.approve(args.id).to_dict()
        if args.command == "reject":
            return queue.reject(args.id).to_dict()
        raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _execute(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
