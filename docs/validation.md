# Local validation record

Validated on 27 October 2026 for version 0.2.3, using Python 3.11.7 on macOS (arm64).

- 1,068 of 1,069 tests passed. The other one creates a file name that is not valid UTF-8, which macOS file systems refuse, so it runs only on Linux. On Linux, the 10 tests that need a case-insensitive file system, as macOS has, are skipped instead; a run on a case-sensitive disk image on macOS skipped exactly those 10 and passed every other test. CI runs the tests on both systems, so every test runs on at least one of them. The tests cover temporal storage, retrieval, extraction, change detection, the CLI, HTTP API, MCP tools, Git checkpoints and writers in separate processes, with regression tests for every fix in the 0.2.2 and 0.2.3 changelogs, output-contract tests, tests of the benchmark scripts, and property-based tests.
- The property-based state machine runs 200 random histories of up to 40 steps, each seeded with records, and compares the full snapshot with an independent reference model after every step, plus random snapshot, history and search queries and the rejected writes.
- Branch coverage is 98.6%, including CLI child processes. CI fails below 90%.
- `mypy` passes with every function annotated and all bodies checked; `ruff check .` passes.
- Mutation testing (mutmut over `store`, `models`, `retrieval`, `extraction` and `embeddings`): 3,983 mutants; 3,642 caught (91.4%), 340 survived and one crashed the test process. 162 survivors only change error-message wording or SQL keyword case (95.3% caught without them). The other 178 were each reviewed against the 0.2.3 code, including those already reviewed for 0.2.2, since a caller that changed can make an old verdict wrong, and the verdict checked by a second reviewer: 127 are equivalent, 41 change only message wording, a temporary file's name or a cache's size limit, and 10 are caught by the tests on a case-sensitive file system (rerun with the test folders on a case-sensitive disk image), where a folder name in another letter case is another folder. None is unreviewed. The review added 17 tests that catch 31 mutants which had survived, and found one bug, in a fix made during this release (corrected before publishing). The run took 27 minutes on 12 cores, from a folder outside iCloud Drive: in a synced folder, files that macOS has offloaded block on read, which stalled one earlier run and turned slow test runs into false timeouts in another. Give the run a temporary folder with room to spare: each mutant leaves pytest's temporary folders behind, and a full disk makes tests fail, which counts their mutants as caught.
- The test suite also passes on Python 3.10 and 3.13 (run from scratch copies); CI covers Python 3.10 to 3.14 on Linux and Python 3.11 on macOS.
- The wheel and source distribution build without warnings, and the wheel includes the `py.typed` marker. The wheel installs into an empty virtual environment with no dependencies, and the installed CLI reproduces the synthetic application timeline.
- A memory folder written by 0.2.2 opens without a rebuild and gives the same search results, and its pending proposals can be approved. An index written by version 0.1 has no file signature (its signature table is empty), so it is rebuilt from Markdown on first open; dated search results are unchanged. An index from 0.2.0 or 0.2.1 (schema version 2) is upgraded in place and read again from Markdown once, so records those releases skipped as notes (a byte-order mark before the marker line, classic Mac line endings) are indexed.
- Learned model weights were not downloaded. Semantic integration is tested through injected providers and a stand-in for the Sentence Transformers package.

## Exact-version retrieval

[Machine-readable report](../benchmarks/results/synthetic-50.json): 50 invented entities per workload. `python benchmarks/run.py --check` turns these expectations into a test, which CI runs.

| Method | State changes (150 queries) | Late corrections (100 queries) | Median retrieval |
| --- | ---: | ---: | ---: |
| Temporal hybrid (valid + knowledge time) | 150 | 100 | 1.2 / 0.9 ms |
| Same engine, valid time only | 150 | 50 | 1.1 / 1.0 ms |
| All-record token overlap | 50 | 50 | 0.2 / 0.1 ms |
| Latest-state token overlap | 50 | 50 | 0.1 / 0.1 ms |

Every miss is a stale or future version of the right entity and attribute; coverage is 100% for all methods. The baselines receive no temporal filtering, so these outcomes are expected by design. This is not evidence of general superiority over RAG. See [full methodology](../benchmarks/README.md).

## Scale

Measured on version 0.2.3 and on the unreleased 0.1 source, on the same machine, one after the other: [0.2.3 report](../benchmarks/results/scale-current.json), [0.1 report](../benchmarks/results/scale-v0.1.json). Medians:

| Records | Operation | 0.1 | 0.2.3 |
| ---: | --- | ---: | ---: |
| 300 | write | 39 ms | 1.8 ms |
| 1,000 | write | 137 ms | 4.0 ms |
| 3,000 | write | 386 ms | 10.5 ms |
| 10,000 | write | 1,367 ms | 46 ms |
| 10,000 | open unchanged store | 1,400 ms | 39 ms |
| 10,000 | full rebuild | 1,416 ms | 1,262 ms |
| 10,000 | warm search (reused store and retriever) | 520 ms | 57 ms |
| 10,000 | cold search (new store and retriever, like one CLI call) | 1,924 ms | 461 ms |

Full rebuilds cost about the same in both versions because both validate every record; 0.2 performs them only when files changed outside the engine. Every search result in the scale run is checked for the expected entity, value and record. Timings are from one run per size and are not comparable across machines. Even on one machine they vary between runs: in the table above, the unchanged 0.1 code ran 2 to 11% slower here than in the previous record, for version 0.2.2, and single measurements in its full report ranged from 6% faster to 13% slower. `python benchmarks/plot.py --check` confirms the charts match these reports.
