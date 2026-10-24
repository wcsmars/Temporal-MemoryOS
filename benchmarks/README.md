# Benchmarks

Three scripts, all offline: no model downloads, API calls, or personal data. Each run uses temporary stores that are removed afterwards.

```sh
python benchmarks/run.py --entities 50 --check --output benchmarks/results/synthetic-50.json
python benchmarks/scale.py --output benchmarks/results/scale-current.json
python benchmarks/plot.py           # Rewrites the SVG charts from the JSON reports.
python benchmarks/plot.py --check   # Writes nothing; fails if a chart no longer matches its report.
```

## Exact-version retrieval (`run.py`)

Two synthetic workloads, each over 50 invented entities:

| Workload | Records | Queries | What it tests |
| --- | ---: | ---: | --- |
| `state_changes` | 150 | 150 | Three dated status changes per entity; each historical state is queried |
| `late_corrections` | 100 | 100 | A status effective on 2025-01-01 is corrected on 2025-02-10, keeping its effective date; the state on 2025-01-15 is queried with the knowledge available before and after the correction |

Methods:

| Method | Candidate memories | Ranking |
| --- | --- | --- |
| `all_records_keyword` | Every recorded version | Token overlap; ties prefer newest recorded fact |
| `latest_state_keyword` | Most recent state only | Token overlap |
| `valid_time_only` | Facts valid at the query date, using everything known now | The package's hybrid retrieval (ablation of `known_at`) |
| `temporal_hybrid` | Facts valid at the query date and known by the query's cutoff | The package's hybrid retrieval |

**Top-1 fact accuracy** requires the expected fact ID, so a wrong version cannot pass by sharing words with the correct version. **Stale-hit rate** counts results with the correct entity and attribute but an incorrect version, including answers that leak a future state or a later correction into a historical query. Coverage and median and p95 retrieval latency are also reported; p95 uses the nearest-rank method (the 95th of 100 sorted values). Index construction is excluded from timings.

This is a regression diagnostic for temporal filtering, and the outcome is expected by construction: a latest-only store cannot recover discarded states, and a system without a knowledge cutoff cannot reproduce what was believed before a correction. Dates are supplied as structured query arguments, not extracted from natural language. The default embedding backend is lexical hashing rather than a learned semantic model. The keyword baselines are intentionally simple and do **not** represent a tuned vector RAG system. These results support no claim about general superiority, LLM factuality, hallucination rate, inference cost, or large-scale performance.

Because the outcome is known in advance, `--check` turns the run into a test. The script exits with status 1 unless every method's top-1 accuracy equals the share it reaches by construction, and `temporal_hybrid` has a stale-hit rate of 0 and a coverage of 1 in both workloads. The expected shares are one in three for both keyword baselines on `state_changes`, one in two for every method except `temporal_hybrid` on `late_corrections`, and all queries otherwise. They do not depend on the number of entities, so CI runs the check on a small corpus.

## Latency as the store grows (`scale.py`)

For each size, the corpus is imported once as Markdown files (three dated records per entity), then the script measures:

- `first_open_ms`: the first open, which validates and indexes every record;
- `open_unchanged_ms`: the median of 5 opens of a store whose files have not changed, which every CLI command and API request pays;
- `full_rebuild_ms`: the median of 3 forced rebuilds that validate every record and rebuild the index;
- `write_ms_median`: adding one fact, after imported files have left the recently-modified window;
- `warm_search_ms_median`: one hybrid search with valid-time and knowledge-time filters;
- `cold_search_ms_median`: the same search from a freshly opened store.

Search is measured in two ways:

- **Warm** search reuses one open store and one retriever for every search. The first search fills their caches of parsed records and document vectors, and later searches reuse them. This is the cost for a program that keeps one store and retriever open.
- **Cold** search opens the store, creates a new retriever, runs one search and closes the store each time, like one CLI call, excluding interpreter start-up. Starting Python and importing the package add a fixed cost that is not measured. The operating system's file cache stays warm.

The API and MCP server open the store for each request and share only the document vectors between requests, so neither number measures them exactly.

Every search result, warm or cold, is checked for the expected entity, value and record ID, and a wrong answer stops the run. By default each size uses 20 writes, 20 warm searches and 20 cold searches (`--writes`, `--searches`).

### The 0.1 baseline

`results/scale-v0.1.json` was measured from the unreleased 0.1 source, which is not part of this repository. Its raw report is included unchanged. It was produced with `--use-installed`, which benchmarks whatever `temporal_memory_os` is importable instead of `./src`:

```sh
PYTHONPATH=<folder holding the 0.1 package> \
  python benchmarks/scale.py --use-installed --label v0.1 --output benchmarks/results/scale-v0.1.json
```

In that report, `package.location` (`build/lib/temporal_memory_os`) is the local build folder of the 0.1 source, and `package.source_sha256` is a hash of its Python files. Both reports were produced on the same machine, one after the other, with the same parameters. Timings come from one run per size; compare shapes, not absolute values across hardware.

## Charts (`plot.py`)

`plot.py` draws the four SVG charts in `results/` from the three JSON reports. Titles and subtitles take their counts (entities, writes, searches) from the reports, and each chart follows the reader's light or dark color scheme.

`plot.py --check` writes nothing and exits with status 1 if any chart differs from what the reports produce, so a chart cannot drift from its data; CI runs it. If a report is missing, is not valid JSON or lacks a field the charts need, the script stops with an error that names the file and the field.

## Tests

`tests/test_benchmarks.py` tests the scripts themselves: `--check` flags a changed accuracy report, the p95 rank, the scale run's answer check (an answer for the wrong entity stops it), the `--use-installed` flag and its abbreviations, and that the committed charts match the committed reports byte for byte.

The unit tests separately cover corrections, knowledge cutoffs, retractions, gaps, invalid records, manual Markdown edits, and change detection. A research-grade evaluation would add independently collected workloads, learned-embedding baselines, equal context budgets, answer-level scoring, multiple runs, and uncertainty estimates before making comparative claims.
