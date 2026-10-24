# Local validation record

Validated on 25 October 2026 for version 0.2.2, using Python 3.11.7 on macOS (arm64).

- 891 of 892 tests passed; the other one creates a file name that is not valid UTF-8, which macOS file systems refuse, so it runs only on Linux (CI). They include behavioral tests across temporal storage, retrieval, extraction, change detection, the CLI, HTTP API, MCP tools, Git checkpoints and writers in separate processes; regression tests for every fix in the 0.2.2 changelog; output-contract tests; tests of the benchmark scripts; and property-based tests.
- The property-based state machine runs 200 random histories of up to 40 steps, each seeded with records, and compares the full snapshot with an independent reference model after every step, plus random snapshot, history and search queries and the rejected writes.
- Branch coverage is 98.6%, including CLI child processes. CI fails below 90%.
- `mypy` passes with every function annotated and all bodies checked; `ruff check .` passes.
- Mutation testing (mutmut over `store`, `models`, `retrieval`, `extraction` and `embeddings`): 3,522 mutants; 3,212 caught (91.2%), 309 survived and one crashed the test process. 153 survivors only change error-message wording or SQL keyword case (95.3% caught without them). The other 156 were each reviewed and the verdict checked by a second reviewer: 137 are equivalent and 19 change only message wording, a temporary file's name or a cache's size limit. None is unreviewed. Run on macOS with Python 3.11 from a folder outside iCloud Drive: in a synced folder, files that macOS has offloaded block on read, which stalled one run and turned slow test runs into false timeouts in another.
- The test suite also passes on Python 3.10 and 3.13 (run from scratch copies); CI covers Python 3.10 to 3.13 on Linux.
- The wheel and source distribution build without warnings. The wheel installs into an empty virtual environment with no dependencies, and the installed CLI reproduces the synthetic application timeline.
- An index written by version 0.1 has no file signature (its signature table is empty), so it is rebuilt from Markdown on first open; dated search results are unchanged. An index from 0.2.0 or 0.2.1 (schema version 2) is upgraded in place and read again from Markdown once, so records those releases skipped as notes (a byte-order mark before the marker line, classic Mac line endings) are indexed.
- Learned model weights were not downloaded. Semantic integration is tested through injected providers and a stand-in for the Sentence Transformers package.

## Exact-version retrieval

[Machine-readable report](../benchmarks/results/synthetic-50.json): 50 invented entities per workload. `python benchmarks/run.py --check` turns these expectations into a test, which CI runs.

| Method | State changes (150 queries) | Late corrections (100 queries) | Median retrieval |
| --- | ---: | ---: | ---: |
| Temporal hybrid (valid + knowledge time) | 150 | 100 | 1.0 / 0.8 ms |
| Same engine, valid time only | 150 | 50 | 1.0 / 0.8 ms |
| All-record token overlap | 50 | 50 | 0.2 / 0.1 ms |
| Latest-state token overlap | 50 | 50 | 0.1 / 0.1 ms |

Every miss is a stale or future version of the right entity and attribute; coverage is 100% for all methods. The baselines receive no temporal filtering, so these outcomes are expected by design. This is not evidence of general superiority over RAG. See [full methodology](../benchmarks/README.md).

## Scale

Measured on version 0.2.2 and on the unreleased 0.1 source, on the same machine, one after the other: [0.2.2 report](../benchmarks/results/scale-current.json), [0.1 report](../benchmarks/results/scale-v0.1.json). Medians:

| Records | Operation | 0.1 | 0.2.2 |
| ---: | --- | ---: | ---: |
| 300 | write | 38 ms | 2.0 ms |
| 1,000 | write | 124 ms | 4.5 ms |
| 3,000 | write | 377 ms | 11 ms |
| 10,000 | write | 1,286 ms | 45 ms |
| 10,000 | open unchanged store | 1,304 ms | 41 ms |
| 10,000 | full rebuild | 1,273 ms | 1,165 ms |
| 10,000 | warm search (reused store and retriever) | 496 ms | 57 ms |
| 10,000 | cold search (new store and retriever, like one CLI call) | 1,780 ms | 447 ms |

Full rebuilds cost about the same in both versions because both validate every record; 0.2 performs them only when files changed outside the engine. Every search result in the scale run is checked for the expected entity, value and record. Timings are from one run per size and are not comparable across machines; `python benchmarks/plot.py --check` confirms the charts match these reports.
