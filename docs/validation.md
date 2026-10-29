# Local validation record

## Version 0.3.0: 29 October 2026

Validated on macOS arm64 with Python 3.11.7. This revision adds the optional Claude Agent SDK integration and the storage, approval, retrieval and checkpoint repairs in the changelog.

- Full local suite: **1,144 passed, 1 skipped**, with **98.44% combined line/branch coverage** including CLI subprocesses. The skipped non-UTF-8 filename test requires a filesystem that supports such names. Property-based histories still compare against an independent reference model.
- Ruff and mypy pass across 13 source modules. The separate private release-maintenance suite passes 388 tests. The release has 64 manifest files; 63 match the main folder byte for byte and the remaining file is the release-specific `.gitignore`.
- New regressions cover replaced folder symlinks, nonregular Markdown files, nested transaction rollback and SQLite auto-abort, concurrent proposal approval, symbol-only entity names, malformed embedding output, and nested Git repositories/gitlinks in checkpoints.
- The 35 SDK tests run offline. Six contract cases comprise four checks using the installed SDK's real subprocess transport with a local protocol peer, plus two simulated cancellation cases. They check command restrictions, suppressed provider diagnostics, failures, cancellation and child-process cleanup. The peer does not call a model. The SDK dependency tested is `claude-agent-sdk==0.2.174`.
- A live SDK test used only two synthetic application facts. The model called search and history, answered the September 10 state as pending, and cited the exact fact ID and source. Both temporal cutoffs were passed correctly, and hashes of all canonical Markdown records were unchanged. Two live smoke calls passed. The final configuration reported four turns and $0.001110625 in model cost; the earlier call reported three turns and $0.0008784. These checks establish integration, not general answer accuracy or guaranteed citation fidelity.
- The 50-entity exact-version benchmark passes all expected outcomes (250 queries across the two workloads). Scale smoke checks at 30 and 90 records pass, and chart consistency passes. Published 0.2.3 timing reports and charts remain unchanged; these checks do not replace their measurements.
- Wheel and source distribution build in a fresh temporary environment using the declared build-backend requirement. Package inventories contain the agent module and typing marker and exclude private guides, credentials and memory data. The wheel installs with no dependencies into an empty environment, and its installed demo and historical search return the expected result.
- The final full suite ran from a byte-verified copy outside the cloud-synced Documents directory. Cloud-offloaded source/dependency files had stalled imports and one benchmark subprocess in the synced checkout; those environment failures are excluded from the passing run. The project environment was rebuilt outside Documents and the focused checks passed through its local link.
- Learned embedding weights were not downloaded. Provider behavior is covered with injected implementations and malformed-output tests.
- Mutation testing of the five core modules completed: **4,043 mutants, 3,702 test kills, 331 survivors, 9 timeouts and 1 killed process**. Isolated follow-ups passed all controls: two timeout cases failed assertions and seven were confirmed nonterminating under explicit wall-clock limits. The killed process is reported separately, not counted as a test kill. Six earlier survivors were caught by strengthened file-replacement regressions. All remaining survivors were reviewed: 165 equivalent, 153 diagnostic-only, 10 platform-dependent, two cache-limit changes and one equivalent result with unnecessary trie work. The platform-dependent cases concern a case-sensitive filesystem. A later broad serial rerun reached its wall limit; it does not supersede the completed run or the isolated checks. `mutmut` 3.8 does not generate mutants for generator/context-manager bodies, so these results do not cover the transaction context manager; failure-injection tests and independent code review cover that code.
- Remote validation is tracked by the [GitHub Actions workflow](https://github.com/wcsmars/Temporal-MemoryOS/actions/workflows/ci.yml) for each commit. It runs Python 3.10–3.14 on Linux and Python 3.11 on macOS. Local results above are distinct from remote job outcomes.

## Historical validation: 0.2.4 and 0.2.3

Version 0.2.4 (27 October 2026) changes only how a proposal file is checked when it is read, so that Python 3.14 on Linux reports a proposal file nested too deeply to write back instead of failing the whole listing (see the changelog). Its 1,070 tests pass on macOS with Python 3.10, 3.11, 3.13 and 3.14 (1,069 run there; the non-UTF-8 file-name test needs Linux), with 98.6% branch coverage, and `ruff` and `mypy` pass. The rest of this record was measured on 0.2.3; the benchmarks do not run the changed code.

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
