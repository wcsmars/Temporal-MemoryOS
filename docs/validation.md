# Local validation record

Validated on 23 October 2026 for version 0.2.1, using Python 3.11.7 on macOS (arm64).

- 157 tests passed with no skips: behavioral tests across temporal storage, retrieval, extraction, change detection, CLI, HTTP API, MCP tools, Git history and independent-process writes, plus property-based tests.
- The property-based state machine ran 200 random histories of up to 40 steps each against an independent reference model, with no disagreement in snapshots, histories, search eligibility, rejected writes or rebuilt indexes.
- Branch coverage is 94% (1,221 statements, 402 branches), including CLI child processes. CI fails below 90%.
- `mypy` passes with every function annotated and all bodies checked; `ruff check .` passes.
- Mutation testing (mutmut 3.8 over `store`, `models`, `retrieval`, `extraction` and `embeddings`): 1,739 of 2,163 mutants caught (80.4%). 272 of the 424 survivors only change string wording or case; excluding them, 92.0% are caught. The other survivors were reviewed by category and change nothing observable in supported setups. The first run caught 71.1%; the gaps it exposed are now tested.
- The wheel and source distribution build without warnings. The wheel installs into an empty virtual environment with no dependencies, and the installed CLI reproduces the synthetic application timeline.
- An index written by version 0.1 is detected by its schema version and rebuilt from Markdown on first open; dated search results are unchanged.
- Python 3.10, 3.12 and 3.13 are covered by the GitHub Actions matrix, not by local runs.
- Learned model weights were not downloaded. Semantic integration is tested through injected providers and a stand-in for the Sentence Transformers package.

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

Measured on version 0.2.0; 0.2.1 only removes work from the write path. [Version 0.2 report](../benchmarks/results/scale-current.json) and [version 0.1 report](../benchmarks/results/scale-v0.1.json), produced by `benchmarks/scale.py` on the same machine on the same day. Medians:

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
