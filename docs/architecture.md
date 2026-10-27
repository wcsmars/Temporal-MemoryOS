# Architecture and temporal semantics

## Data flow

```mermaid
flowchart TD
  C["Conversation"] --> E["Rule extraction"]
  E --> P["Proposal review"]
  P --> W["Approved assertion"]
  O["Explicit operator edit"] --> W
  W --> M["Canonical Markdown"]
  M --> I["Validated SQLite index"]
  Q["Query and time filters"] --> S["Temporal snapshot"]
  I --> S
  S --> R["Keyword, embeddings, links"]
  R --> F["Rank fusion and citations"]
```

The Markdown files are authoritative. SQLite provides a disposable, validated record index and a signature of the files it was built from. It holds no full-text index of its own: keyword search builds one per query over the time-filtered snapshot (see Retrieval). API and MCP requests open a store per request but share one embedding model and one vector cache per process; direct Python instances can be reused. The index file is created readable by its owner only, like the records.

### Keeping the index honest without rereading everything

The index stores, for every Markdown file, its size, modification time and SHA-256. Opening the store and every write first compare that signature with the folder:

1. Stat every `*.md` file under `memory/` (one `os.scandir` pass; no file is read). Names starting with a dot are skipped: editors create lock and swap files that way (Emacs's lock file is a dangling `.#name.md` link), and hidden folders belong to other tools.
2. If the set of paths, or any size or modification time, differs, validate every record and rebuild the index.
3. A file modified within three seconds of its last verification is also hashed, because a coarse filesystem clock could give a later edit the same timestamp. This is the "racily clean" rule Git uses for its own index.
4. Otherwise the index is trusted. This common case is answered in a read transaction, so opening a store does not wait for a writer (such as a Git checkpoint) that holds the write lock.

A write then validates only the history it touches. Every validation rule (one successor per record, corrections keep entity, attribute and start, no duplicate effective start, no correction recorded before its target) is local to one entity/attribute pair, except ID uniqueness, which is a primary-key lookup. The write holds SQLite `BEGIN IMMEDIATE` across the check, the atomic file replacement and the index insert, so independent processes serialize. An independent-process test exercises duplicate-write conflicts.

Any failure inside the write transaction, including the `COMMIT` itself, removes the Markdown file written in it. A process killed after the file replacement but before commit leaves a valid record that the index does not list; the next check sees the extra path and indexes it, so the write took effect. A rebuild is all-or-nothing: a malformed record makes the operation fail and leaves the previous index untouched until the file is repaired. Error messages name files relative to the memory directory, so the HTTP API never discloses where the store lives. Problems found in the files on disk raise `StoreIntegrityError`, which the API reports as 503 rather than blaming the request. A symbolic-link Markdown file or folder under `memory/` (other than a dot-named one) is rejected, and so is a link in place of the index or its log files.

Notes without a record marker line are never decoded as records, so their encoding does not matter. A file that the index knew as a record but that no longer holds one (emptied, truncated, or with its marker line changed) is reported instead of being treated as a note, so a value it had superseded cannot silently come back. A brand-new index has no such memory, so this check applies from the second open onward. Records are UTF-8 and may start with a byte-order mark and use Windows or classic Mac line endings; a record saved in another encoding, including UTF-16 and UTF-32, is reported rather than read as a note. File names under `memory/` must be valid UTF-8; another name is reported, naming the file.

The index has a schema version (currently 3). An older index is upgraded in place and its file signatures are dropped, so the first open reads every file again under the current parsing rules (0.2.0 and 0.2.1 read a record behind a byte-order mark or with classic Mac line endings as a note); an index written by a newer release is refused with a message to delete `.memoryos/`, because older code cannot know what a newer index assumes. An index first created by 0.2.0 keeps that release's full-text table, unused and no longer updated, so 0.2.0 processes still running do not fail; delete `.memoryos/` and run `memoryos reindex` to remove it. A damaged index file (SQLite reports it is not a database, or a damaged page) stops the operation with a message; only `memoryos reindex` (or `MemoryStore(root, repair_index=True)`), which first checks every page and the column names of the index tables, renames it to `index.sqlite3.corrupt-<time>` and rebuilds, because renaming database files under other running processes is unsafe. A damaged column name, or a stored record that no longer decodes, shows up only when a query reads it; it is reported as index damage with the same repair hint (HTTP 503). `memoryos reindex` sets an index with a damaged column name aside and rebuilds a damaged record in place; a normal open pays for neither check. Locked, read-only or full disks are never mistaken for damage. Two processes opening a new store at the same moment retry briefly instead of failing on the schema lock. Files and their folders, including new topic folders, are flushed with `fsync` after each write, matching SQLite's default durability. The index file is readable by its owner only.

Reads parse each indexed record once per store instance and reuse it while its stored payload is unchanged; records are immutable, including their tags and links. Writes made within the same few milliseconds are given strictly increasing recorded times. Snapshots push the knowledge cutoff and optional entity filter into SQL. External text editors do not take the database lock, so do not edit concurrently with application writes.

## Two time axes

| Field or parameter | Meaning |
| --- | --- |
| `valid_from` | First instant an assertion applies to the world |
| `valid_to` | Optional exclusive end; later state changes can shorten it |
| `recorded_at` | When the assertion or correction became known |
| `as_of` | Effective instant requested by the query |
| `known_at` | Latest recorded knowledge available to the query |

For each entity/attribute pair:

1. Keep only audit records recorded at or before `known_at`.
2. Hide records superseded by visible revisions, retaining them in the audit store.
3. Sort remaining assertions, including tombstones, by effective start.
4. Bound each interval by its explicit end and the next assertion's start.
5. Return non-tombstones satisfying `valid_from <= as_of < valid_to`.

A missing end is open-ended. An explicit expiration leaves a gap until the next assertion; an earlier state does not reappear. An approved restatement inside such an interval keeps the same explicit end. A tombstone keeps its interval boundary and prevents resurrection of an earlier state. A later correction restores a tombstoned assertion when desired. Corrections retain the original entity, attribute, and start; they must not predate the record they revise. Duplicate effective starts without a supersession relation are rejected.

This model represents a single-valued changing attribute, not arbitrary overlapping event intervals. Normalize entity and attribute names consistently: identity is case-sensitive in the store. Retrieval's optional entity filter is case-insensitive for convenience. Snapshot `valid_to` is derived and may differ from a record fetched by audit ID.

## Retrieval

Every channel sees the resolved snapshot. Keyword search builds a temporary SQLite FTS5 table over eligible facts so excluded future facts cannot affect BM25 statistics. The default embedding channel uses deterministic token hashing with a small declared alias map. A Sentence Transformers model can replace it through configuration. Document vectors are cached by embedder name and the exact embedded text, so an edited record or a different model can never reuse a stale vector; mostly-zero vectors are stored sparsely and others as arrays of doubles. A retriever caches up to 50,000 vectors, and a cache can be shared by several retrievers. Text is NFKC-normalized and case-folded before tokenizing, and combining marks stay inside words; the keyword index also folds accents. Common English question words are dropped from keyword and lexical queries unless the query names an entity whose name contains them; keyword search decides this after folding accents, so "à" counts as "a", and at most 64 distinct keyword terms are used. Relationship search seeds entities whose words all appear in the query (when several names share their words, as C, C# and C++ do, the one written out in the query wins; quotes, brackets, sentence punctuation and a possessive "'s" around it are allowed) and expands them through explicit `links` for one hop.

Hybrid retrieval uses reciprocal rank fusion with denominator `60 + rank`, and a 0.7 weight on the graph channel. Ranks are competition ranks: equal raw scores share a rank, so a tie can never be broken by the random order of fact IDs before fusion. Raw channel scores and the fused rank are returned for inspection. It does not generate answers, infer missing facts, or turn confidence values into truth guarantees. The record ID, source, evidence, interval, and Markdown path allow the calling assistant to cite evidence.

## Trust boundaries and limits

Only structured canonical blocks are indexed. Ordinary prose notes do not become facts accidentally. Extraction never executes text and never approves its own updates. The CLI/API also provide direct operator writes for intentionally supplied facts. MCP exposes read tools only. Memory text is untrusted data, including when it contains instructions or protocol-looking strings.

Topic paths are confined to the memory tree; symlinked records, folders and index files are rejected. The HTTP API only answers requests addressed to localhost unless other host names are configured. FTS query tokens are quoted, and SQL uses parameters. Approval files use separate serialization and a POSIX lock. This is a local application, not a hardened hostile multi-user filesystem. Backups, filesystem permissions, and encryption remain the operator's responsibility.

There is no cross-file transaction combining a newly written fact and its proposal approval status. Approving again after an interruption recognizes the fact that was already written (same value, date, source, evidence, confidence and topic, not already claimed by an approved proposal that states the same thing) and marks the proposal approved. Rejecting such a proposal is refused while its fact is unrevised and no other pending proposal stating the same line (its topic in another letter case) can claim it, because the fact would stay; approve it to finish, then retract the fact if it is wrong. Once the fact has been retracted or corrected, the proposal can be rejected. Git checkpoints hold the proposal lock and the store's write lock while staging, so a commit never contains one without the other. Git is explicit opt-in and does not erase retracted evidence.

## Modules

| Module | Responsibility |
| --- | --- |
| `models.py` | Fact validation, UTC timestamps, safe topic paths |
| `store.py` | Markdown records, change detection, SQLite index, temporal resolution, revisions |
| `extraction.py` | Rule extraction, proposals, human approval |
| `embeddings.py` | Offline hashing and optional learned provider |
| `retrieval.py` | Snapshot filtering, ranking, links, citations |
| `git_history.py` | Local checkpoint isolation |
| `cli.py` | JSON command-line interface |
| `api.py` | Optional FastAPI service |
| `mcp_server.py` | Optional SDK-managed read-only stdio tools |
