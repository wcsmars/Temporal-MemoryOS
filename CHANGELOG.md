# Changelog

## 0.2.1

- Fix: `checkpoint` no longer lets Git invent an author from the computer's account name and hostname when no identity is configured. It now requires `user.name`/`user.email` in the memory repository or the standard Git environment variables, as documented.
- Approving a proposal file that lacks its target ID now fails with a clear message instead of an unrelated lookup error.
- Fix: proposals extracted within one clock tick could list out of line order (ties broke on random IDs). Creation times are now strictly increasing within a batch.
- The index no longer maintains a persistent full-text table that no query read; keyword search has always built its own over the time-filtered snapshot. Existing indexes drop it on open.
- Removed the unused `embeddings.cosine_similarity` helper and `MemoryStore.conn` alias.
- Verification: property-based tests against an independent reference model, strict `mypy`, branch coverage with a 90% floor in CI, and a configured mutation-testing run (see the README).

## 0.2.0

- Writes no longer reread and revalidate every Markdown record. The index keeps a signature of the files it was built from (size, modification time, SHA-256); a cheap scan detects outside edits, deletions and new files, and only then triggers a full rebuild. A write validates just the entity/attribute history it touches.
- Files modified within three seconds of their last verification are hashed rather than trusted by timestamp, so a same-size edit on a filesystem with a coarse clock is still detected.
- New `MemoryStore.refresh()`; opening a store rebuilds the index only when the files changed. `reindex()` still forces a full rebuild.
- Snapshots push the knowledge cutoff and an optional `entity` filter into SQL; parsed records are reused while their stored payload is unchanged.
- The retriever caches sparse document vectors and token sets by fact ID.
- Index schema version 2. Version 1 indexes are rebuilt automatically on first open.
- Symbolic-link folders anywhere under `memory/` are rejected.
- New `late_corrections` benchmark workload and `valid_time_only` ablation; new `scale.py` latency benchmark and SVG charts.

## 0.1.0

- Markdown fact records with valid time and knowledge time, corrections and retractions.
- SQLite FTS5 index, hybrid keyword/embedding/link retrieval with reciprocal rank fusion.
- Rule-based extraction into a human review queue; opt-in local Git checkpoints.
- JSON CLI, optional FastAPI service and read-only MCP server.
