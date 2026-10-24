# Temporal MemoryOS

[![tests](https://github.com/wcsmars/Temporal-MemoryOS/actions/workflows/ci.yml/badge.svg)](https://github.com/wcsmars/Temporal-MemoryOS/actions/workflows/ci.yml)

**An editable, auditable memory engine for AI agents that keeps changing facts in time.**

An application can be pending on September 10, invited on September 20, and completed on October 2. Searching the latest state should not rewrite that history. Temporal MemoryOS stores dated assertions in nested Markdown files, reconstructs the relevant state, and retrieves facts with their evidence and provenance.

A local Python library and CLI work without API keys or third-party dependencies. Optional FastAPI, MCP, and learned embeddings provide integrations. All bundled examples are synthetic.

## Quick start

Python 3.10+ with SQLite FTS5; macOS or Linux for the proposal queue's POSIX file lock.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

memoryos --root ./memory-data demo
memoryos --root ./memory-data search "Acme application status" \
  --as-of 2026-09-10 --entity "Acme application" --limit 1
```

The result contains `value: "pending"`, its effective interval, source, evidence, fact ID, retrieval scores, and the Markdown citation. Run the same search with `--as-of 2026-09-20` for `interview invited`, or `--as-of 2026-10-02` for `interview completed`.

`--root` goes **before** the command. Set `MEMORYOS_HOME` to use a different default directory. The demo is safe to run repeatedly: it never overwrites or corrects existing records, and reports any slot that already holds something else under `skipped_existing`.

To run directly from a checkout without installing:

```sh
PYTHONPATH=src python3 -m temporal_memory_os --root ./memory-data demo
```

## What is implemented

| Capability | Behavior |
| --- | --- |
| Hierarchical Markdown | One canonical JSON block per Markdown fact, in topic subfolders; files are directly editable |
| Temporal state | Separate effective time and recorded time; half-open intervals; out-of-order updates |
| Corrections and retractions | Append revisions and tombstones, preserve original evidence, reject conflicting writes |
| Retrieval | SQLite FTS5/BM25, embedding similarity, one-hop entity relationships, reciprocal rank fusion |
| Review queue | Offline extraction, conflict previews, explicit approve/reject, stale-preview detection |
| Git history | Opt-in local commits of memory and proposal files, with unrelated staging preserved |
| Interfaces | Python API, JSON CLI, optional HTTP API and read-only MCP stdio tools |
| Incremental index | File-signature change detection; writes validate only the history they touch |
| Validation | Behavioral and property-based tests, strict typing, coverage and mutation testing, reproducible benchmarks |

The default embedding provider is **lexical hashing**, not a learned semantic model. Learned semantic embeddings require the optional provider below. Extraction is a deliberately narrow rule-based parser; it does not claim to understand arbitrary conversation.

## Results

All numbers come from synthetic data on one laptop; the [benchmark methodology](benchmarks/README.md) states what they do and do not show.

**Exact-version retrieval.** Each query names an entity and asks for its state on a date, optionally restricted to what was known by a second date. A hit requires the exact record version.

![Exact-version retrieval accuracy](benchmarks/results/accuracy.svg)

Keyword search over every version, or over the latest state only, returns the newest status and misses two of every three historical questions. The same engine without a knowledge cutoff answers every state-change question but returns the later correction when asked what was believed before it arrived. With both time axes, all 250 queries return the expected version. These outcomes follow from the design; the workloads are regression checks that the design holds, not evidence of general superiority over RAG systems.

**Latency as the store grows.** Version 0.1 reread and revalidated every Markdown record on each write and on every open. Version 0.2 checks a file signature instead, validates only the history a write touches, and caches parsed records and document vectors.

![Write latency by store size](benchmarks/results/scale-write.svg)

![Warm search latency by store size](benchmarks/results/scale-search.svg)

![Cold and warm search latency by store size](benchmarks/results/scale-search-cold.svg)

| Records | Write, 0.1 | Write, 0.2.2 | Open, 0.1 | Open, 0.2.2 | Warm search, 0.1 | Warm search, 0.2.2 | Cold search, 0.1 | Cold search, 0.2.2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 124 ms | 4.5 ms | 122 ms | 3.7 ms | 47 ms | 5.0 ms | 175 ms | 42 ms |
| 10,000 | 1,286 ms | 45 ms | 1,304 ms | 41 ms | 496 ms | 57 ms | 1,780 ms | 447 ms |

Medians on an Apple-silicon laptop with Python 3.11; [raw reports](benchmarks/results/). A warm search reuses one open store and retriever, as the HTTP and MCP servers and a long-lived Python program do. A cold search opens the store and builds a new retriever, as each CLI command does (Python start-up excluded); it parses the time-filtered records and embeds them again, which is why it stays about eight times slower than a warm one. Opening and writing still check every file's size and timestamp, so both stay linear in the number of records, at roughly 4 microseconds per file. A full rebuild, needed only after outside edits, costs about the same in both versions.

## Record, correct, and retrieve

These commands use a fresh directory so they do not collide with the demo's synthetic records.

```sh
export MEMORYOS_HOME=./my-memory
memoryos add "Acme application" status pending \
  --from 2026-09-01 --source "conversation:001" \
  --evidence "The application was submitted and is awaiting review." \
  --topic career/applications --link Acme --tag application

memoryos add "Acme application" status "interview invited" \
  --from 2026-09-15 --source "email:invitation"

memoryos history "Acme application" --attribute status
memoryos search "Acme application status" --as-of 2026-09-10
memoryos correct FACT_ID "interview scheduled" --source "conversation:correction"
memoryos retract NEW_ID --source "conversation:retraction"
```

Replace `FACT_ID` with an ID returned by a command, and `NEW_ID` with the ID that `correct` returned: only the latest revision of a record can be corrected or retracted. `correct` keeps the original effective date and creates a new revision. Use `add` with a later `--from` when the world changes. A retraction removes that assertion from resolved snapshots; it is **not deletion of the audit record**.

For knowledge history, pass a separate cutoff. The demo's records carry the dates they were learned, so this works on the demo directory:

```sh
memoryos --root ./memory-data search "Alex preferred drink" \
  --as-of 2026-10-10 --known-at 2026-10-01 --limit 1    # tea: the change was not known yet
memoryos --root ./memory-data search "Alex preferred drink" \
  --as-of 2026-10-10 --limit 1                          # decaf coffee
```

The demo learned on 5 October that Alex's drink had changed, so a query limited to what was known on 1 October still answers tea. Corrections work the same way: one learned in October changes the retrospective answer for September, but cannot appear in a query restricted to knowledge available in September. Records entered with `add` are learned now, so a `--known-at` earlier than today excludes them; an importer can supply `recorded_at` through the Python API. Both query times default to now. Times are `YYYY-MM-DD`, optionally followed by `THH:MM`, seconds, up to six fractional digits, and `Z` or `+HH:MM`; the same strings are accepted on every supported Python version. A date without a time means midnight UTC; offsets are normalized to UTC. Date arguments are explicit: the engine does not infer dates from the question text.

## Extract with human review

```sh
memoryos extract --file examples/conversation.txt --source "synthetic:conversation" \
  --observed-at 2026-10-01 --topic career/applications
memoryos proposals
memoryos approve PROPOSAL_ID
memoryos reject PROPOSAL_ID
```

Extraction writes proposals only. Review the value, effective date, evidence, change kind, and previous value before approving each ID. Direct `add`, `correct`, and `retract` commands are explicit operator actions and do not use the queue. An approval is refused (HTTP 409) if memory changed in a way the reviewer did not see, including manual edits that keep the same fact ID. Approve a dated conversation's proposals in chronological order: a newer predecessor approved from the same source is expected and does not make later proposals stale, and a later line that repeats the value already in effect is approved without writing a new record. Running an extraction again returns the existing pending proposals instead of duplicates, unless a value for the same attribute has taken effect since.

Damaged proposal files are skipped and reported (`proposals` prints a warning for each on standard error; the HTTP list includes them under `problems`), so one bad file does not block the queue; fix or delete the named file.

Supported inputs are one assertion per line:

```text
Acme application | status | pending | 2026-09-01
2026-09-15: My application at Acme is interview invited.
My preferred language is English.
I prefer decaf coffee.
```

The fourth pipe-separated field is an optional effective date. An optional date prefix also works. Otherwise the observation time is used. Unsupported lines are skipped, including prose or tables that contain `|` without forming three or four fields, and lines with an invalid date; zero proposals means no supported statements were found. Text or a source that is not valid Unicode is rejected. A line with both a date prefix and a different fourth-field date is skipped as ambiguous, and so is a dated line that is a `#` comment. The sentence rules are deliberately narrow and can still misread unusual prose, which is why every proposal is reviewed. Rule confidence values describe parser certainty, not calibrated truth probabilities. Each `(entity, attribute)` has one value at a time, so two lines giving the same attribute for the same time are flagged and only one can be approved; use distinct attributes for multiple concurrent preferences.

## Editable storage

```text
memory-data/
  memory/
    career/applications/<fact-id>.md
    profile/preferences/<fact-id>.md
  proposals/<proposal-id>.json
  .memoryos/index.sqlite3
```

Every fact includes `entity`, `attribute`, `value`, `valid_from`, optional `valid_to`, `recorded_at`, `source`, `evidence`, `confidence`, `tags`, `links`, and revision metadata. The JSON fence following `<!-- memoryos:fact:v1 -->` is canonical. Unmarked Markdown notes, in any encoding, are left alone and are not indexed. Names starting with a dot (editor lock and swap files, hidden folders such as `.obsidian/`) are ignored; a symbolic-link Markdown file or folder under `memory/` is rejected. A file that held a record and loses its JSON block (emptied by accident, truncated, or with its marker line changed) is reported by name instead of silently disappearing, so a value it had superseded cannot come back. On case-insensitive file systems (the macOS default), use one spelling per topic folder: `Career/` and `career/` are the same folder there.

Edits made with an editor are picked up automatically. The index records each Markdown file's size, modification time and SHA-256. Opening the store and every write compare sizes and modification times with the folder, and also hash any file modified within a few seconds of its last check (a timestamp that close cannot be trusted on every filesystem). The index is rebuilt only if something changed outside the engine. A copy that keeps the old size and timestamp (`cp -p`, `rsync -t`, `touch -r`) can go unnoticed; after such a change, rebuild explicitly:

```sh
memoryos reindex
```

A complete validation pass precedes index replacement; a malformed record cannot replace a good index with a partial one, and the error names the files involved, relative to the memory directory. Long-lived Python instances call `refresh()` (cheap when nothing changed) to see external edits made while they are open. SQLite is disposable: if the index file is damaged, commands stop with a message, and `memoryos reindex` sets the damaged file aside as `index.sqlite3.corrupt-<time>` and rebuilds it from the Markdown (stop servers using the directory first). An index written by a newer release asks you to delete `.memoryos/`; to go back to an older release, delete `.memoryos/` too, since older releases cannot read the newer index. Changing an old file directly rewrites that source history; use `correct` when the previous assertion must remain auditable.

## Python API and learned embeddings

```python
from temporal_memory_os import MemoryStore
from temporal_memory_os.retrieval import HybridRetriever

with MemoryStore("./memory-data") as store:
    matches = HybridRetriever(store).search(
        "Acme application status", as_of="2026-09-10", limit=3
    )
    for match in matches:
        print(match.fact.value, match.fact.source, match.citation)
```

To enable a learned embedding model for the CLI, API, and MCP:

```sh
python -m pip install -e '.[semantic]'
export MEMORYOS_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
memoryos --root ./memory-data search "What drink does Alex enjoy?" --mode semantic
```

The first embedding request may download model weights; a model already in the local cache loads without network access. Code published alongside a model on the Hub is never executed, so a model that needs such code may load with untrained weights and return meaningless scores; check results when trying a new model. A local model directory is loaded like a Python package, so only use directories you trust. `HybridRetriever(store)` in Python also honors `MEMORYOS_EMBEDDING_MODEL`; pass `embedder=SentenceTransformerEmbedder(name)` to choose a model explicitly. The HTTP and MCP servers load the model once per process and share one cache of document vectors. In Python, vectors are cached per retriever: keep one `HybridRetriever` for repeated searches, or pass the same `vector_cache=VectorCache()` to several.

The semantic threshold (default 0.25) suits MiniLM; models with different similarity scales need their own value, set with `MEMORYOS_SEMANTIC_THRESHOLD`. Semantic mode is for paraphrases; for exact entity names, `hybrid` or `keyword` mode is more reliable.

Modes are `keyword`, `semantic`, `graph`, and `hybrid`. Without a learned model, the embedding channel is clearly returned as `lexical_embedding`. Common question words such as "what" and "does" are ignored in keyword and lexical matching unless they are part of an entity name (so "The Who" is still found), and text is compared after Unicode normalization and case folding; keyword matching also ignores accents. Graph retrieval follows links in either direction for one hop. Temporal filtering occurs before every retrieval channel, including the relationships used for expansion. Rank fusion gives tied scores the same rank. Results are evidence records rather than generated answers, and scores are ranks, not probabilities.

## HTTP API

```sh
python -m pip install -e '.[api]'
# Optional bearer token, required when exposing the API to other users:
export MEMORYOS_API_TOKEN='choose-a-local-token'
memoryos --root ./memory-data serve
```

The default listener is `127.0.0.1:8000`. Interactive documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). `/health` is public; every `/api` route uses bearer authentication when the token is set (ASCII only). Requests must be addressed to `localhost`, `127.0.0.1` or `::1`, which stops web pages from reaching the API through DNS rebinding, and requests that would change memory are refused when a browser marks them as coming from another site. When serving on another interface, add the host names clients use to `MEMORYOS_ALLOWED_HOSTS` (comma-separated; localhost stays allowed) and set a token: these checks protect against browsers, not against other clients on the network. This is a single-user local service, not a multi-tenant deployment.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET / POST | `/api/facts` | Read a dated snapshot / add an assertion |
| GET / DELETE | `/api/facts/{id}` | Fetch an audit record / retract it |
| POST | `/api/facts/{id}/corrections` | Correct a fact |
| GET | `/api/search?q=...` | Search with `as_of`, `known_at`, `limit`, `mode`, `entity` |
| GET | `/api/history?entity=...` | Inspect an effective timeline |
| POST | `/api/proposals/extract` | Propose updates from text and source |
| GET | `/api/proposals` | Review pending proposals |
| POST | `/api/proposals/{id}/approve` or `/reject` | Resolve a proposal |
| POST | `/api/reindex` | Rebuild the derived index |

The API rejects client-supplied `recorded_at`; imports that need it use the Python interface. Invalid data produces 422, missing records 404, conflicting writes and stale approvals 409, and 503 means the server's files need attention: a damaged record or index (the message names the file relative to the memory directory) or an embedding model that cannot load (details go to the server log only).

## MCP integration

```sh
python -m pip install -e '.[mcp]'
memoryos --root ./memory-data mcp
```

Use this configuration in a compatible client that supports local stdio MCP servers, replacing both absolute paths:

```json
{
  "mcpServers": {
    "temporal-memory-os": {
      "command": "/absolute/path/to/.venv/bin/memoryos",
      "args": ["--root", "/absolute/path/to/private-memory", "mcp"]
    }
  }
}
```

Tools are `search`, `fetch`, and `memory_history`. They are read-only; changes happen through the operator CLI or API approval workflow. `fetch` retrieves an audit record and may return a superseded assertion, so use dated `search` to determine current truth. Treat retrieved memory as data, not instructions. This integration does not replace any client's built-in memory, and local stdio configuration is client-specific. The server uses the [official MCP Python SDK v1](https://py.sdk.modelcontextprotocol.io/v1/).

## Opt-in Git checkpoints

Choose a separate private directory **outside any other repository**; `git-init` refuses a directory that is already a repository it did not create:

```sh
memoryos --root /absolute/path/to/private-memory init
memoryos --root /absolute/path/to/private-memory git-init
# If no Git identity is configured (globally or in this repository), set one here.
git -C /absolute/path/to/private-memory config user.name "Your Name"
git -C /absolute/path/to/private-memory config user.email "you@example.com"
memoryos --root /absolute/path/to/private-memory checkpoint --message "Review initial memory"
memoryos --root /absolute/path/to/private-memory git-log
```

Checkpoints use the identity configured in Git (globally or in that repository) or Git's author and committer environment variables, and refuse to run without one; Git is never allowed to guess one from the account name and hostname. Repository-location variables such as `GIT_DIR` are ignored, so a checkpoint always goes to the memory directory's own repository. A checkpoint waits for in-flight writes and approvals, so it never captures an approved proposal without its fact; while it runs, searches continue and writes wait (up to 30 seconds, so a process writing continuously can make a checkpoint fail; retry it). No remote is created and nothing is pushed. Checkpoints only commit memory and proposal changes, excluding locks, temporary files, editor backups (`notes.md~`, `#notes.md#`) and derived indexes. A repository you created yourself is used only after you mark it with `git -C <memory directory> config memoryos.checkpoints true`. Git retains prior contents: retraction is not secure erasure. Use synthetic data in public repositories and keep your personal store private.

To restore a checkpoint, stop anything using the store, bring both folders back to the same commit, then rebuild the index:

```sh
git -C /absolute/path/to/private-memory restore --source=COMMIT --staged --worktree -- memory proposals
memoryos --root /absolute/path/to/private-memory reindex
```

This also removes records committed after `COMMIT`. Files that were never committed stay; `git -C /absolute/path/to/private-memory status --short -- memory proposals` lists them. Do not use `git revert` on a checkpoint whose records were later corrected: removing a record that a later correction refers to leaves a store that cannot be opened until the correction is removed too.

## Project layout

| Path | Contents |
| --- | --- |
| `src/temporal_memory_os/store.py` | Markdown records, change detection, SQLite index, bitemporal resolution, corrections |
| `src/temporal_memory_os/retrieval.py` | Snapshot-first keyword, embedding and link retrieval with rank fusion |
| `src/temporal_memory_os/extraction.py` | Rule-based extraction and the approval queue with stale-preview checks |
| `src/temporal_memory_os/models.py` | Fact validation, UTC normalization, safe topic paths |
| `src/temporal_memory_os/{cli,api,mcp_server,git_history}.py` | JSON CLI, optional HTTP API, read-only MCP tools, opt-in Git checkpoints |
| `tests/` | Behavioral and property-based tests, including a reference model of the time rules |
| `benchmarks/` | Accuracy and scale benchmarks, their JSON reports and the charts above |
| `tools/mutation_summary.py` | Summarizes a mutation-testing run and lists survivors worth review |
| `docs/` | [Architecture and temporal semantics](docs/architecture.md), [validation record](docs/validation.md) |

## Verification

Each layer catches a different kind of mistake. CI runs every layer except mutation testing on each push, for Python 3.10 to 3.13.

| Layer | What it checks |
| --- | --- |
| Behavioral tests | Temporal semantics, corrections, retractions, outside edits, damaged files and indexes, interrupted writes and approvals, CLI, HTTP API, MCP tools, Git checkpoints, and writers in separate processes |
| Regression and contract tests | One or more tests per fixed bug, each confirmed to fail when its fix is reverted, and the JSON shapes, defaults and file names that users and scripts rely on |
| Property-based tests | Random histories of adds, corrections, retractions and searches, compared after every step with an independent reference model of the time rules, including which writes must be rejected and why. Also: time normalization ordering, Markdown round-trips of arbitrary text, and extraction of arbitrary structured lines |
| Coverage | Branch coverage, including the CLI's child processes; CI fails below 90% (currently 98.6%) |
| Types | `mypy` with every function annotated and all bodies checked |
| Mutation testing | Breaks the core modules one small change at a time (3,522 mutants) and checks that some test fails |
| Benchmarks | Exact-version retrieval against baselines and latency as the store grows; CI fails if accuracy changes or a chart no longer matches its report |

```sh
python -m pip install -e '.[api,mcp,dev]'
pytest -q --cov          # 892 tests, coverage report and 90% floor
ruff check .
mypy
python benchmarks/run.py --entities 50 --check --output benchmark-results.json
python benchmarks/scale.py --sizes 300 1000 --output benchmark-results-scale.json
python benchmarks/plot.py --check

# Mutation testing (about 20 minutes on a 12-core laptop; macOS or Linux;
# run it from a folder that a cloud drive does not sync)
python -m pip install -e '.[mutation]'
HYPOTHESIS_PROFILE=mutation mutmut run
python tools/mutation_summary.py
```

In the latest run, on version 0.2.2, tests caught 3,212 of 3,522 mutants (91.2%); 309 survived and one crashed the test process. Of the survivors, 153 only change the wording of an error message or the letter case of an SQL keyword, which the tests deliberately do not pin; without them the score is 95.3%. Every other survivor was reviewed by hand, and a second reviewer tried to refute each verdict. 137 are equivalent: no input can tell them apart from the original, for example a codec name in capitals, which Python matches case-insensitively, or a check on a branch that earlier validation rules out. The other 19 change only an error message's wording, a temporary file's name or a cache's size limit. The review added the tests in `tests/*_pins.py`, which pin behavior no earlier test checked, and, with an adversarial check of its fixes, found 22 edge-case bugs, all fixed in this release (see the changelog).

See [the validation record](docs/validation.md) and the [changelog](CHANGELOG.md). The benchmarks measure exact-version retrieval and latency, not generated answer accuracy.

## Design and scope

See [architecture and temporal semantics](docs/architecture.md).

This is a working local foundation, sized for a personal or team memory of thousands to tens of thousands of records. Each search resolves the temporal snapshot and builds a temporary FTS table over it. Document vectors are cached by the HTTP and MCP servers for the life of the process and by each Python retriever; each CLI command is a new process and starts cold. Sub-linear opens (filesystem notifications), persistent vector indexes, learned conversational extraction, multi-hop reasoning, multi-user authorization, encrypted storage, hosted transport, and general benchmark comparisons are future work. See the [changelog](CHANGELOG.md).

Proposal status and fact writes are separate atomic files. If a process stops between them, approving the proposal again recognizes the fact that was already written. Writes are flushed with `fsync`, like SQLite's default settings; on macOS that reaches the drive's cache rather than guaranteeing survival of a power cut. Manual file editing must not race with writers.

MIT licensed. No personal data, credentials, or remote service is required for the default demo.
