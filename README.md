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

`--root` goes **before** the command. Set `MEMORYOS_HOME` to use a different default directory. The demo is safe to run repeatedly; it does not overwrite existing records.

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
| Validation | Behavioral tests, GitHub Actions, reproducible synthetic diagnostics and scale benchmark |

The default embedding provider is **lexical hashing**, not a learned semantic model. Learned semantic embeddings require the optional provider below. Extraction is a deliberately narrow rule-based parser; it does not claim to understand arbitrary conversation.

## Results

All numbers come from synthetic data on one laptop; the [benchmark methodology](benchmarks/README.md) states what they do and do not show.

**Exact-version retrieval.** Each query names an entity and asks for its state on a date, optionally restricted to what was known by a second date. A hit requires the exact record version.

![Exact-version retrieval accuracy](benchmarks/results/accuracy.svg)

Keyword search over every version, or over the latest state only, returns the newest status and misses two of every three historical questions. The same engine without a knowledge cutoff answers every state-change question but returns the later correction when asked what was believed before it arrived. With both time axes, all 250 queries return the expected version. These outcomes follow from the design; the workloads are regression checks that the design holds, not evidence of general superiority over RAG systems.

**Latency as the store grows.** Version 0.1 reread and revalidated every Markdown record on each write and on every open. Version 0.2 checks a file signature instead and validates only the history a write touches.

![Write latency by store size](benchmarks/results/scale-write.svg)

![Search latency by store size](benchmarks/results/scale-search.svg)

| Records | Write, 0.1 | Write, 0.2 | Open, 0.1 | Open, 0.2 | Dated search, 0.1 | Dated search, 0.2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 127 ms | 4.9 ms | 130 ms | 4.3 ms | 48 ms | 4.8 ms |
| 10,000 | 1,389 ms | 46 ms | 1,374 ms | 46 ms | 501 ms | 61 ms |

Medians on an Apple-silicon laptop with Python 3.11; [raw reports](benchmarks/results/). Opening and writing still stat every file, so both stay linear in the number of records, at roughly 4.5 microseconds per file.

## Record, correct, and retrieve

```sh
memoryos add "Acme application" status pending \
  --from 2026-09-01 --source "conversation:001" \
  --evidence "The application was submitted and is awaiting review." \
  --topic career/applications --link Acme --tag application

memoryos add "Acme application" status "interview invited" \
  --from 2026-09-15 --source "email:invitation"

memoryos history "Acme application" --attribute status
memoryos search "Acme application status" --as-of 2026-09-10
memoryos correct FACT_ID "interview scheduled" --source "conversation:correction"
memoryos retract FACT_ID --source "conversation:retraction"
```

Replace `FACT_ID` with an ID returned by a command. `correct` keeps the original effective date and creates a new revision. Use `add` with a later `--from` when the world changes. A retraction removes that assertion from resolved snapshots; it is **not deletion of the audit record**.

For knowledge history, pass a separate cutoff:

```sh
memoryos search "Acme application status" \
  --as-of 2026-09-10 --known-at 2026-09-12T23:59:59Z
```

A correction learned in October changes the retrospective answer for September, but cannot appear in a query restricted to knowledge available in September. Newly entered historical facts are learned now unless an importer explicitly supplies `recorded_at` through the Python API. Both query times default to now. A date without a time means midnight UTC; timestamps with offsets are normalized to UTC. Date arguments are explicit: the engine does not infer dates from the question text.

## Extract with human review

```sh
memoryos extract --file examples/conversation.txt --source "synthetic:conversation" \
  --observed-at 2026-10-01 --topic career/applications
memoryos proposals
memoryos approve PROPOSAL_ID
memoryos reject PROPOSAL_ID
```

Extraction writes proposals only. Review the value, effective date, evidence, change kind, and previous value before approving each ID. Direct `add`, `correct`, and `retract` commands are explicit operator actions and do not use the queue. Approvals detect changed correction targets, including manual edits retaining the same fact ID. Approve a dated conversation's proposals in chronological order.

Supported inputs are one assertion per line:

```text
Acme application | status | pending | 2026-09-01
2026-09-15: My application at Acme is interview invited.
My preferred language is English.
I prefer decaf coffee.
```

The fourth pipe-separated field is an optional effective date. An optional date prefix also works. Otherwise the observation time is used. Unsupported prose is ignored; zero proposals means no supported statements were found. Rule confidence values describe parser certainty, not calibrated truth probabilities. Each `(entity, attribute)` has one value at a time; use distinct attributes for multiple concurrent preferences.

## Editable storage

```text
memory-data/
  memory/
    career/applications/<fact-id>.md
    profile/preferences/<fact-id>.md
  proposals/<proposal-id>.json
  .memoryos/index.sqlite3
```

Every fact includes `entity`, `attribute`, `value`, `valid_from`, optional `valid_to`, `recorded_at`, `source`, `evidence`, `confidence`, `tags`, `links`, and revision metadata. The JSON fence following `<!-- memoryos:fact:v1 -->` is canonical. Unmarked Markdown notes are left alone and are not indexed.

Edits are picked up automatically. The index stores each Markdown file's size, modification time and hash; opening the store and every write compare them with the folder and rebuild only if something changed outside the engine. To force a full rebuild:

```sh
memoryos reindex
```

A complete validation pass precedes index replacement; a malformed record cannot replace a good index with a partial one. Long-lived Python instances call `refresh()` (cheap when nothing changed) to see external edits made while they are open. SQLite is disposable and can be rebuilt from the Markdown. Changing an old file directly rewrites that source history; use `correct` when the previous assertion must remain auditable.

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
memoryos search "What drink does Alex enjoy?" --mode semantic
```

The first embedding request may download model weights. A local model directory also works. No remote model code is trusted. For Python, supply `SentenceTransformerEmbedder(model_name)` explicitly to `HybridRetriever(store, embedder=...)`.

Modes are `keyword`, `semantic`, `graph`, and `hybrid`. Without a learned model, the embedding channel is clearly returned as `lexical_embedding`. Graph retrieval follows links in either direction for one hop. Temporal filtering occurs before every retrieval channel, including the relationships used for expansion. Results are evidence records rather than generated answers, and scores are ranks, not probabilities.

## HTTP API

```sh
python -m pip install -e '.[api]'
# Optional bearer token, required when exposing the API to other users:
export MEMORYOS_API_TOKEN='choose-a-local-token'
memoryos --root ./memory-data serve
```

The default listener is `127.0.0.1:8000`. Interactive documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). `/health` is public; every `/api` route uses bearer authentication when the token is set. Use the authorization header for requests. This is a single-user local service, not a multi-tenant deployment.

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

The API rejects client-supplied `recorded_at`; imports that need it use the Python interface. Invalid data produces 422, missing records 404, and conflicting memory writes 409.

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

Choose a separate private directory **outside the source repository**:

```sh
memoryos --root /absolute/path/to/private-memory init
memoryos --root /absolute/path/to/private-memory git-init
# If Git identity is not already configured, configure it locally in that directory.
git -C /absolute/path/to/private-memory config user.name "Your Name"
git -C /absolute/path/to/private-memory config user.email "you@example.com"
memoryos --root /absolute/path/to/private-memory checkpoint --message "Review initial memory"
memoryos --root /absolute/path/to/private-memory git-log
```

No remote is created and nothing is pushed. Checkpoints only commit memory and proposal changes, excluding locks and derived indexes. Git retains prior contents: retraction is not secure erasure. Use synthetic data in public repositories and keep your personal store private.

## Project layout

| Path | Contents |
| --- | --- |
| `src/temporal_memory_os/store.py` | Markdown records, change detection, SQLite index, bitemporal resolution, corrections |
| `src/temporal_memory_os/retrieval.py` | Snapshot-first keyword, embedding and link retrieval with rank fusion |
| `src/temporal_memory_os/extraction.py` | Rule-based extraction and the approval queue with stale-preview checks |
| `src/temporal_memory_os/models.py` | Fact validation, UTC normalization, safe topic paths |
| `src/temporal_memory_os/{cli,api,mcp_server,git_history}.py` | JSON CLI, optional HTTP API, read-only MCP tools, opt-in Git checkpoints |
| `tests/` | Behavioral tests, including independent-process write conflicts and outside-edit detection |
| `benchmarks/` | Accuracy and scale benchmarks, their JSON reports and the charts above |
| `docs/` | [Architecture and temporal semantics](docs/architecture.md), [validation record](docs/validation.md) |

## Tests and benchmarks

```sh
python -m pip install -e '.[api,mcp,dev]'
pytest -q
ruff check .
python benchmarks/run.py --entities 50 --output benchmark-results.json
python benchmarks/scale.py --sizes 300 1000 --output benchmark-results-scale.json
```

See [the local validation record](docs/validation.md). The diagnostics measure exact-version retrieval and latency. They are not an evaluation of generated answer accuracy or hallucinations.

## Design and scope

See [architecture and temporal semantics](docs/architecture.md).

This is a working local foundation, sized for a personal or team memory of thousands to tens of thousands of records. Each search resolves the temporal snapshot and builds a temporary FTS table over it; document vectors are cached per process. Sub-linear opens (filesystem notifications), persistent vector indexes, learned conversational extraction, multi-hop reasoning, multi-user authorization, encrypted storage, hosted transport, and general benchmark comparisons are future work. See the [changelog](CHANGELOG.md). Proposal status and fact writes are separate atomic files; a process crash between them may leave a committed fact with a pending proposal that requires operator reconciliation. Manual file editing must not race with writers.

MIT licensed. No personal data, credentials, or remote service is required for the default demo.
