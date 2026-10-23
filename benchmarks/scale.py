#!/usr/bin/env python3
"""Measure open, write, rebuild and search latency as the memory grows.

Each corpus is imported once (Markdown files written directly, then one full
rebuild), so construction time does not depend on the write path under test.
Write latency is measured after files leave the recently-modified window,
which is the steady state of a store that is used every day.
"""

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

if "--use-installed" not in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import temporal_memory_os  # noqa: E402
from temporal_memory_os.embeddings import HashingEmbedder  # noqa: E402
from temporal_memory_os.models import Fact  # noqa: E402
from temporal_memory_os.retrieval import HybridRetriever  # noqa: E402
from temporal_memory_os.store import MemoryStore  # noqa: E402

STATES = (("2025-01-01", "pending"), ("2025-02-01", "interview invited"),
          ("2025-03-01", "interview completed"))


def _import_corpus(root: Path, facts: int) -> None:
    """Write ``facts`` records as an importer would, three states per entity."""
    memory = root / "memory" / "benchmark"
    memory.mkdir(parents=True)
    for index in range(facts):
        effective, value = STATES[index % 3]
        fact = Fact(
            id="{:032x}".format(index + 1), entity="candidate{:05d}".format(index // 3),
            attribute="application_status", value=value, valid_from=effective,
            recorded_at=effective, source="synthetic://scale", topic="benchmark",
            evidence="Synthetic status update: {}.".format(value), tags=["synthetic"],
        )
        (memory / (fact.id + ".md")).write_text(MemoryStore._render(fact), encoding="utf-8")


def _median_ms(samples):
    return round(statistics.median(samples) * 1000, 3)


def measure(facts: int, writes: int, searches: int, settle: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="memoryos-scale-") as directory:
        root = Path(directory)
        _import_corpus(root, facts)
        start = time.perf_counter()
        MemoryStore(root).close()  # First open validates and indexes every file.
        first_open = time.perf_counter() - start
        time.sleep(settle)  # Let imported files leave the recently-modified window.
        with MemoryStore(root) as store:
            # Open of an unchanged store (what every CLI call and API request pays).
            opens = []
            for _ in range(5):
                start = time.perf_counter()
                MemoryStore(root).close()
                opens.append(time.perf_counter() - start)
            rebuilds = []
            for _ in range(3):
                start = time.perf_counter()
                store.reindex()
                rebuilds.append(time.perf_counter() - start)
            time.sleep(settle)
            if hasattr(store, "refresh"):
                store.refresh()
            write_times = []
            for index in range(writes):
                start = time.perf_counter()
                store.add("newcomer{:03d}".format(index), "application_status", "pending",
                          valid_from="2025-04-01", source="synthetic://scale")
                write_times.append(time.perf_counter() - start)
            retriever = HybridRetriever(store, embedder=HashingEmbedder())
            search_times = []
            for index in range(searches):
                entity = "candidate{:05d}".format((index * 7919) % max(1, facts // 3))
                start = time.perf_counter()
                hits = retriever.search(entity + " application_status", as_of="2025-02-15",
                                        known_at="2025-04-01", limit=1)
                search_times.append(time.perf_counter() - start)
                if not hits or hits[0].fact.value != "interview invited":
                    raise AssertionError("Scale benchmark retrieved the wrong version")
    return {
        "facts": facts,
        "first_open_ms": round(first_open * 1000, 3),
        "open_unchanged_ms": _median_ms(opens),
        "full_rebuild_ms": _median_ms(rebuilds),
        "write_ms_median": _median_ms(write_times),
        "search_ms_median": _median_ms(search_times),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 1000, 3000, 10000])
    parser.add_argument("--writes", type=int, default=20)
    parser.add_argument("--searches", type=int, default=20)
    parser.add_argument("--settle", type=float, default=3.5,
                        help="Seconds to wait so files leave the recently-modified window.")
    parser.add_argument("--label", default="current")
    parser.add_argument("--use-installed", action="store_true",
                        help="Benchmark the importable package instead of ./src.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not 3 <= size <= 100_000 for size in args.sizes):
        parser.error("--sizes must be between 3 and 100000")
    rows = []
    for size in args.sizes:
        rows.append(measure(size, args.writes, args.searches, args.settle))
        print(json.dumps(rows[-1]), file=sys.stderr)
    report = {
        "benchmark": "scale_latency_v1",
        "label": args.label,
        "version": getattr(temporal_memory_os, "__version__", "unknown"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine()},
        "workload": "Three dated status records per synthetic entity; hybrid search with "
                    "valid-time and knowledge-time filters; offline lexical embeddings.",
        "notes": [
            "Single local run per size; medians over repeated operations.",
            "Writes are measured after imported files leave the recently-modified window.",
            "Timings are machine-specific and not comparable across hardware.",
        ],
        "results": rows,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
