# Local validation record

Validated on 23 October 2026 for version 0.2.0, using Python 3.11.7 on macOS (arm64).

- 64 tests passed with no skips, across temporal storage, retrieval, extraction, change detection, CLI, HTTP API, MCP tools, Git history, and independent-process write conflicts.
- `ruff check .` passes with the configured rules.
- The wheel and source distribution build without warnings. The wheel installs into an empty virtual environment with no dependencies, and the installed CLI reproduces the synthetic application timeline.
- An index written by version 0.1 is detected by its schema version and rebuilt from Markdown on first open; dated search results are unchanged.
- Python 3.10, 3.12 and 3.13 are covered by the GitHub Actions matrix, not by local runs.
- Learned model weights were not downloaded. Semantic integration is tested through an injected provider and lazy configuration, not through a downloaded Sentence Transformers model.

## Exact-version retrieval

[Machine-readable report](../benchmarks/results/synthetic-50.json): 50 invented entities per workload.

| Method | State changes (150 queries) | Late corrections (100 queries) | Median retrieval |
| --- | ---: | ---: | ---: |
| Temporal hybrid (valid + knowledge time) | 150 | 100 | 0.9 / 0.7 ms |
| Same engine, valid time only | 150 | 50 | 0.9 / 0.7 ms |
| All-record token overlap | 50 | 50 | 0.2 / 0.1 ms |
| Latest-state token overlap | 50 | 50 | 0.1 / 0.1 ms |

Every miss is a stale or future version of the right entity and attribute; coverage is 100% for all methods. The baselines receive no temporal filtering, so these outcomes are expected by design. This is not evidence of general superiority over RAG. See [full methodology](../benchmarks/README.md).

## Scale

[Version 0.2 report](../benchmarks/results/scale-current.json) and [version 0.1 report](../benchmarks/results/scale-v0.1.json), produced by `benchmarks/scale.py` on the same machine on the same day. Medians:

| Records | Operation | 0.1 | 0.2 |
| ---: | --- | ---: | ---: |
| 300 | write | 39 ms | 2.0 ms |
| 1,000 | write | 127 ms | 4.9 ms |
| 3,000 | write | 399 ms | 12 ms |
| 10,000 | write | 1,389 ms | 46 ms |
| 10,000 | open unchanged store | 1,374 ms | 46 ms |
| 10,000 | full rebuild | 1,351 ms | 1,190 ms |
| 10,000 | dated hybrid search | 501 ms | 61 ms |

Full rebuilds cost about the same in both versions because both validate every record; 0.2 performs them only when files changed outside the engine. Each search result in the scale run is checked for the correct version. Timings are from one run per size and are not comparable across machines.
