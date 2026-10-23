# Benchmarks

Three scripts, all offline: no model downloads, API calls, or personal data. Each run uses temporary stores that are removed afterwards.

```sh
python benchmarks/run.py --entities 50 --output benchmarks/results/synthetic-50.json
python benchmarks/scale.py --output benchmarks/results/scale-current.json
python benchmarks/plot.py   # Rewrites the SVG charts from the JSON reports.
```

## Exact-version retrieval (`run.py`)

Two synthetic workloads, each over 50 invented entities:

| Workload | Records | Queries | What it tests |
| --- | ---: | ---: | --- |
| `state_changes` | 150 | 150 | Three dated status changes per entity; each historical state is queried |
| `late_corrections` | 100 | 100 | A status is corrected weeks later at the same effective date; that date is queried with the knowledge available before and after the correction |

Methods:

| Method | Candidate memories | Ranking |
| --- | --- | --- |
| `all_records_keyword` | Every recorded version | Token overlap; ties prefer newest recorded fact |
| `latest_state_keyword` | Most recent state only | Token overlap |
| `valid_time_only` | Facts valid at the query date, using everything known now | The package's hybrid retrieval (ablation of `known_at`) |
| `temporal_hybrid` | Facts valid at the query date and known by the query's cutoff | The package's hybrid retrieval |

**Top-1 fact accuracy** requires the expected fact ID, so a wrong version cannot pass by sharing words with the correct version. **Stale-hit rate** counts results with the correct entity and attribute but an incorrect version, including answers that leak a future state or a later correction into a historical query. Coverage and median/p95 retrieval latency are also reported. Index construction is excluded from timings.

This is a regression diagnostic for temporal filtering, and the outcome is expected by construction: a latest-only store cannot recover discarded states, and a system without a knowledge cutoff cannot reproduce what was believed before a correction. Dates are supplied as structured query arguments, not extracted from natural language. The default embedding backend is lexical hashing rather than a learned semantic model. The keyword baselines are intentionally simple and do **not** represent a tuned vector RAG system. These results support no claim about general superiority, LLM factuality, hallucination rate, inference cost, or large-scale performance.

## Latency as the store grows (`scale.py`)

For each size, the corpus is imported once as Markdown files (three dated records per entity), then the script measures the median of repeated operations:

- `open_unchanged_ms`: opening a store whose files have not changed, which every CLI command and API request pays;
- `full_rebuild_ms`: validating every record and rebuilding the index;
- `write_ms_median`: adding one fact, after imported files have left the recently-modified window;
- `search_ms_median`: one warm hybrid search with valid-time and knowledge-time filters; every answer is checked for the correct version.

`--use-installed` benchmarks whatever `temporal_memory_os` is importable, which is how the 0.1 baseline in `results/scale-v0.1.json` was produced from the 0.1 source. Timings come from one machine and one run per size; compare shapes, not absolute values across hardware.

The unit tests separately cover corrections, knowledge cutoffs, retractions, gaps, invalid records, manual Markdown edits, and change detection. A research-grade evaluation would add independently collected workloads, learned-embedding baselines, equal context budgets, answer-level scoring, multiple runs, and uncertainty estimates before making comparative claims.
