#!/usr/bin/env python3
"""Measure open, write, rebuild and search latency as the memory grows.

Each corpus is imported once (Markdown files written directly, then one full
rebuild), so construction time does not depend on the write path under test.
Write latency is measured after files leave the recently-modified window,
which is the steady state of a store that is used every day.

Search is measured two ways. A warm search reuses one store and one retriever,
so their per-instance caches (parsed records, document vectors) are filled.
A cold search opens the store, creates a new retriever, searches once and
closes the store, like one CLI call, excluding interpreter start-up.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 1000, 3000, 10000])
    parser.add_argument("--writes", type=int, default=20)
    parser.add_argument("--searches", type=int, default=20,
                        help="Number of warm searches and of cold searches per size.")
    parser.add_argument("--settle", type=float, default=3.5,
                        help="Seconds to wait so files leave the recently-modified window.")
    parser.add_argument("--label", default="current")
    parser.add_argument("--use-installed", action="store_true",
                        help="Benchmark the importable package (for example one on PYTHONPATH) "
                             "instead of ./src.")
    parser.add_argument("--output", type=Path)
    return parser


def _wants_installed(argv=None) -> bool:
    """Whether the command line asks for ``--use-installed``.

    The import location must be chosen before the package is imported. Parsing
    the command line with the full parser applies argparse's own rules, so an
    abbreviation such as --use-inst selects the installed package as well.
    """
    return bool(_parser().parse_args(argv).use_installed)


_USE_INSTALLED = _wants_installed() if __name__ == "__main__" else False
if not _USE_INSTALLED:
    sys.path.insert(0, str(REPOSITORY / "src"))

import temporal_memory_os  # noqa: E402
from temporal_memory_os.embeddings import HashingEmbedder  # noqa: E402
from temporal_memory_os.models import Fact  # noqa: E402
from temporal_memory_os.retrieval import HybridRetriever  # noqa: E402
from temporal_memory_os.store import MemoryStore  # noqa: E402

STATES = (("2025-01-01", "pending"), ("2025-02-01", "interview invited"),
          ("2025-03-01", "interview completed"))
SEARCH = {"as_of": "2025-02-15", "known_at": "2025-04-01", "limit": 1}
EXPECTED_VALUE = "interview invited"


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


def _target(facts: int, index: int):
    """Return the entity searched for by search number ``index`` and the record ID it must
    return. Entity k's records are numbers 3k, 3k + 1 and 3k + 2 (IDs one higher), and
    the second of them is the state valid on the search date."""
    number = (index * 7919) % max(1, facts // 3)
    return "candidate{:05d}".format(number), "{:032x}".format(3 * number + 2)


def _check(hits, entity: str, expected_id: str) -> None:
    if not hits:
        raise AssertionError("Scale benchmark found nothing for " + entity)
    fact = hits[0].fact
    if (fact.entity, fact.value, fact.id) != (entity, EXPECTED_VALUE, expected_id):
        raise AssertionError(
            "Scale benchmark searched for {} and retrieved {} = {!r} (record {}); expected {!r} "
            "(record {})".format(entity, fact.entity, fact.value, fact.id, EXPECTED_VALUE,
                                 expected_id))


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
            if hasattr(store, "refresh"):  # Version 0.1 has no refresh().
                store.refresh()
            write_times = []
            for index in range(writes):
                start = time.perf_counter()
                store.add("newcomer{:03d}".format(index), "application_status", "pending",
                          valid_from="2025-04-01", source="synthetic://scale")
                write_times.append(time.perf_counter() - start)
            # Warm: one retriever answers every search, so the first search fills
            # the caches that the others reuse.
            retriever = HybridRetriever(store, embedder=HashingEmbedder())
            warm_times = []
            for index in range(searches):
                entity, expected_id = _target(facts, index)
                start = time.perf_counter()
                hits = retriever.search(entity + " application_status", **SEARCH)
                warm_times.append(time.perf_counter() - start)
                _check(hits, entity, expected_id)
        # Cold: like one CLI call, open the store as its only user, build a new
        # retriever, search once and close. Interpreter start-up and imports are
        # excluded; the operating system's file cache stays warm.
        time.sleep(settle)
        cold_times = []
        for index in range(searches):
            entity, expected_id = _target(facts, index)
            start = time.perf_counter()
            with MemoryStore(root) as cold_store:
                hits = HybridRetriever(cold_store, embedder=HashingEmbedder()).search(
                    entity + " application_status", **SEARCH)
            cold_times.append(time.perf_counter() - start)
            _check(hits, entity, expected_id)
    return {
        "facts": facts,
        "first_open_ms": round(first_open * 1000, 3),
        "open_unchanged_ms": _median_ms(opens),
        "full_rebuild_ms": _median_ms(rebuilds),
        "write_ms_median": _median_ms(write_times),
        "warm_search_ms_median": _median_ms(warm_times),
        "cold_search_ms_median": _median_ms(cold_times),
    }


def _package() -> dict:
    """Describe which code was measured without recording an absolute path."""
    directory = Path(temporal_memory_os.__file__).resolve().parent
    try:
        location = directory.relative_to(REPOSITORY).as_posix()
    except ValueError:
        location = "outside this repository"
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return {"version": getattr(temporal_memory_os, "__version__", "unknown"),
            "location": location, "use_installed": _USE_INSTALLED,
            "source_sha256": digest.hexdigest()}


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.use_installed != _USE_INSTALLED:
        parser.error("--use-installed only works when scale.py is run as a script")
    if any(not 3 <= size <= 100_000 for size in args.sizes):
        parser.error("--sizes must be between 3 and 100000")
    if args.writes < 1 or args.searches < 1:
        parser.error("--writes and --searches must be at least 1")
    if args.settle < 0:
        parser.error("--settle must not be negative")
    package = _package()
    if args.use_installed and package["location"] == "src/temporal_memory_os":
        print("Note: --use-installed imported this checkout's src/ (an editable install?).",
              file=sys.stderr)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    rows = []
    for size in args.sizes:
        rows.append(measure(size, args.writes, args.searches, args.settle))
        print(json.dumps(rows[-1]), file=sys.stderr)
    report = {
        "benchmark": "scale_latency_v2",
        "label": args.label,
        "version": package["version"],
        "package": package,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "cpu_count": os.cpu_count(),
                        "load_average_1m_at_start": None if load is None else round(load, 2)},
        "parameters": {"sizes": args.sizes, "writes": args.writes, "searches": args.searches,
                       "settle_seconds": args.settle},
        "workload": "Three dated status records per synthetic entity; hybrid search with "
                    "valid-time and knowledge-time filters; offline lexical embeddings.",
        "metrics": {
            "first_open_ms": "Opening the imported corpus for the first time: every record is "
                             "validated and indexed.",
            "open_unchanged_ms": "Median of 5 opens of a store whose files have not changed.",
            "full_rebuild_ms": "Median of 3 forced rebuilds that validate every record.",
            "write_ms_median": "Adding one fact to an open store.",
            "warm_search_ms_median": "One hybrid search through a store and retriever reused "
                                     "for every search; the first search fills their caches.",
            "cold_search_ms_median": "Open the store, create a new retriever, run one hybrid "
                                     "search and close the store, like one CLI call, excluding "
                                     "interpreter start-up.",
        },
        "notes": [
            "Single local run per size; medians over repeated operations.",
            "Writes are measured after imported files leave the recently-modified window.",
            "Every search result is checked for the expected entity, value and record ID.",
            "Cold searches exclude interpreter start-up and imports; the operating system's "
            "file cache is warm.",
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
