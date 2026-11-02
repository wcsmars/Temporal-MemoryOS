# Changelog

## 0.3.1

Fixes for six problems found while connecting a personal memory to desktop AI assistants over MCP, and for further problems that reviews of those fixes found.

Upgrading
- Restart MCP clients and any running `memoryos serve` after installing 0.3.1. A process started before the upgrade keeps 0.3.0's behavior: it recreates a missing memory directory, compares names exactly (each spelling is still its own history with its own current value, and names are filed exactly as given, also at a start that another spelling holds) and lacks the new date descriptions. A `serve` started before the upgrade can also fail extraction and proposal requests with HTTP 500 once the installed files are replaced.
- Scripts and services that relied on a command, the HTTP API or the MCP server creating a new store need `memoryos init` with the same `--root` or `MEMORYOS_HOME` first.
- Pending proposals and approved records that 0.3.0 made from bulleted, numbered or to-do lines keep the marker in their entity (`- user`, `1. user`, `- [ ] user`). Extracting the text again proposes the name without the marker instead of returning them, and skips a to-do line. Reject such proposals (`memoryos proposals` lists them); retract such records and add them again under the right name.
- A proposal still pending from 0.3.0 was previewed against its own spelling of the names only. Approval now checks it against every spelling. If a record in another spelling changes what the preview showed, for example by holding a value at the proposal's date, approval is refused, unless that record is a newer predecessor approved from the same source (as when a conversation is approved in order). When that record was recorded before the proposal was made, the refusal names it and says that a release before 0.3.1, which compared names exactly, may have made the proposal. Reject a refused proposal and extract the text again. Otherwise the proposal is approved, and its record takes the spelling memory already uses.

Memory folder
- A memory directory that was moved, renamed or deleted is reported instead of being silently replaced by a new, empty one at the old path. Only `init`, `demo` and `git-init` create a missing directory, now readable by its owner only. Every other command, `add` and `extract` included, stops (exit status 1) with an error that says to move the folder back or run `memoryos init` with the same `--root` or `MEMORYOS_HOME`. `serve` and `mcp` start anyway: the HTTP API answers 503 and never creates the directory, and each MCP tool call says it is missing until it is back, without a restart. Both open the `--root` they were given on each request, so a symbolic link re-pointed after a move is followed.
- In Python, `MemoryStore(root, create=False)` (a new keyword-only argument) raises the new `temporal_memory_os.store.MissingMemoryDirectoryError` when the directory is missing; it is a `StoreIntegrityError` and so a `ValueError`. The default still creates the directory, now also readable by its owner only (0.3.0 used the default mode, usually 0755), and `store.created` says whether it did.
- These problems with the memory directory are now reported without its path: a file where the directory should be ("The memory directory path exists but is not a folder"), a file in place of its `memory/`, `.memoryos/` or `proposals/` folder (HTTP 503 instead of 500 for all of these), a directory that cannot be opened or created ("Cannot open the memory directory: Permission denied", "Cannot create the memory directory: ..."), and a `--root` that is a loop of symbolic links. `memoryos agent` and `agent.ask` give the same errors as other commands for a missing directory or a file in its place, and a separate one for a directory without a `memory/` folder (never set up with `init` or `demo`); 0.3.0 called all three missing. A `git-init` that fails before Git has made its `.git` folder, for example because Git is not installed or the directory is inside another repository, removes the directory it created and any parent folders it made for it.

Names
- Entity and attribute names no longer depend on letter case, Unicode compatibility forms or extra spaces. Before, `User` and `user`, or `Favourite drink` and `favourite drink`, were separate histories, so both values looked current, while `memory_history` matched only the exact spelling. History, snapshots, audit trails, search and extraction now join every spelling. `add` (CLI, HTTP API and Python) files a new record under the spelling memory already uses for those names and returns it with those names, which can differ from the names given; the CLI also says so on standard error. Extraction proposes names in that spelling too; a name memory does not have yet takes the spelling of an earlier proposal that was not rejected, or of an earlier line of the same text. Records keep the spelling they were written with, and the file and index formats are unchanged, so a 0.3.0 process can still read the same folder. `temporal_memory_os.models.name_key(text)` gives the form in which names are compared.
- Stores written in several spellings load unchanged and show one timeline per name, which can change their current answers. A value that only looked current because it was filed under another spelling now ends where the other spelling's next record starts. A retraction made in one spelling to remove such a duplicate now also ends the other spelling's value from the retracted record's start, which can leave the name with no current value; `memoryos correct <retraction id> <value>` restores one. Names that differ only in letter case or compatibility form are now one name, even when they are different words, such as `Apple` and `apple`, or `Maße` and `Masse` (case folding writes ß as ss). Where two spellings share a start, the one whose original record was recorded later is shown, and correcting or retracting either does not change that; a new record cannot take a start another spelling holds (use `correct`).
- The load error for a correction that changes its target's names now says they must match exactly, including letter case.

Dates
- A date alone in `as_of` or `known_at` still means 00:00 UTC at the start of that day (08:00 at UTC+8). Because the interfaces did not say so, an assistant that asked about today with today's date missed facts added later that day. The MCP server's instructions and its date parameter descriptions, the HTTP API documentation and the CLI help for `search`, `history` and `agent` now say so, how to give the end of a local day instead (a timestamp with a UTC offset, such as `2026-11-01T23:59:59.999999+08:00`), and that no cutoff means now. The HTTP API documentation adds that `+` is written `%2B` in a URL. The agent's prompt says to copy the cutoffs it is given exactly. MCP `search` also returns `as_of_utc` and `known_at_utc`, the UTC instants it used, and `memory_history` returns `known_at` and `known_at_utc`; existing fields are unchanged. The help for `add --from/--to` and `extract --observed-at` says what a date alone means.

Extraction
- A line in a bulleted or numbered list is read without its marker (a bullet such as `-`, `*`, `+`, `•` or `·`, or up to three digits with `.` or `)`, followed by whitespace), and the evidence leaves the marker out. Before, `- user | snack | popcorn` proposed the entity `- user`, and numbered prose such as `1. 2026-11-01: My favourite drink is matcha.` was skipped. Letters and bracketed numbers (`a.`, `(1)`) are not markers. A name that itself starts like a list number loses it (`1. FC Köln | league | Bundesliga` now proposes `FC Köln`); put a bullet before such a line to keep it. To-do items (`- [ ] ...`, `[x] ...`) are skipped like comments.
- Extracting the same line again returns its pending proposal, also in another list style or after memory has come to use another spelling of its names. The evidence keeps the line's own capitals, so a line retyped in other capitals gets a proposal of its own, and only one of the two can be approved.

Messages
- The damaged-index message says to stop other programs using the memory directory, then run reindex with `--root` set to it, for example `memoryos --root <memory directory> reindex`. It suggested a bare `memoryos reindex`, which rebuilt the default directory (`MEMORYOS_HOME`, or `./memory-data` in the current folder), so a store opened with another `--root` or `MEMORYOS_HOME` (as in an MCP client's configuration), or the `./memory-data` of another folder, stayed damaged.
- A JSON syntax error in a hand-edited record gives the line and column in the file, as an editor shows them, with any line endings and with or without a byte-order mark; it counted lines from the start of the JSON block. The message no longer includes a character offset.

Tests
- Mutation runs no longer fill the disk with pytest temporary folders. `mutmut` ends each test worker with `os._exit`, which skips pytest's exit-time cleanup, so every mutant left a folder behind. A test session that mutmut starts now deletes its own base temporary folder when it ends; other test runs are unchanged.
- A timing test that compares searches over 100 and 400 nested names alternates the two sizes, so a busy machine slows both; it failed once under heavy load.
- Every mutant that survived in code changed for 0.3.1 was reviewed. Where one showed a difference no test caught, a test now does: a start held by another spelling of only the entity or only the attribute, the exact text of messages this release added or reworded (including the name of a column missing from a damaged index), a record recorded at the very instant a proposal was made, a retracted target beside a hand-written record whose ID is "None", a line spelled like one of several queued proposals, a repair that finds the reopened index damaged again, and the cache of stored entity names.

## 0.3.0

- Add optional Claude Agent SDK support through `memoryos agent` and the async Python `agent.ask` function. The SDK manages model calls and tool execution; memory access uses the existing three read-only MCP tools. Turn, budget and time limits are configurable, with built-in tools, external MCP configuration and saved conversations disabled.
- Approval validation and fact writes now share a transaction, preventing a concurrent direct write from changing the reviewed state before approval. Nested writes roll back their Markdown files together with SQLite changes.
- An open store rejects a replacement symlink for its memory folder and rejects nonregular Markdown files, including named pipes that could otherwise block reads. Failed refresh transactions are rolled back.
- Graph and hybrid retrieval handle entity names made only of symbols. Invalid numeric output from a learned embedding provider is reported as an embedding failure.
- Git checkpoints refuse nested repositories and gitlinks instead of silently omitting their fact contents.
- Mutation summaries refuse to publish a score when any mutant has no associated tests or was skipped, including a cached resume that lost its test mappings.
- Add regression tests, SDK boundary tests, and SDK dependency installation to CI. Benchmark figures from 0.2.3 remain pinned; current validation is reported separately.

## 0.2.4

- On Linux with Python 3.14, which can load JSON nested more deeply than it can write back, such a proposal file is reported as a damaged file like any other; before, it stopped the whole proposal listing (and the CLI command or HTTP request that listed proposals).
- CI runs every Python version to the end instead of cancelling the others at the first failure.
- A speed test compares how the time grows with the input instead of a fixed time limit, which a slow CI machine exceeded.

## 0.2.3

Fixes from another full review of the code, the documentation and every step of the user guide.

Memory store
- `memoryos reindex` checks every page of the index and the column names of its tables first, so damage that a normal open never reads (a record page, an index page, or the schema, including a damaged column name) is set aside and rebuilt; stored record text that is no longer valid UTF-8, or no longer reads as a record, is rebuilt in place. Elsewhere, such damage now gets the repair hint: a CLI message, HTTP 503 and an MCP tool error, instead of a raw database error (HTTP 500) or, for a damaged record, invalid input (HTTP 422).
- A topic spelled differently from an existing folder on a case-insensitive file system (the macOS default) is refused instead of forcing a full rebuild on every later write; corrections and retractions of a record filed under such a folder, or moved out of its topic folder, follow the existing folder's spelling instead of being refused.
- A symbolic link in place of `memory/`, `.memoryos/`, the index or its log files, the proposal folder or a proposal file is reported as a problem on disk (HTTP 503) rather than as invalid input (422).

Retrieval
- A common word inside an entity name ("The Who") counts only when the question names that entity; before, one such entity made the word match every fact. Keyword search decides this after folding accents, so "à" counts as "a".
- The document-vector cache keeps its most recently used vectors when full; before, it cached nothing for snapshots above 50,000 records.
- One record with an unusable vector is skipped instead of breaking every hybrid and semantic search; unusable model output is an embedding problem (HTTP 503), not invalid input. A broken dependency inside sentence-transformers is reported with its cause.

Review queue
- A line with its own date (in the line, or an explicit observation time) that repeats the current value from an earlier date is recorded when approved, so a change approved later for an earlier date cannot override it; it keeps the end of the record it confirms, so it never extends a value past an explicit end; other sources' pending proposals stay valid when such a restatement changes nothing they showed, and a second restatement for the same date is approved by the record the first one wrote. Proposals have a new `observed` field (true when an undated line took an explicit observation time); earlier releases refuse proposal files written by 0.2.3, so settle pending proposals before going back to an older release.
- An undated restatement does not fold into an older pending proposal when a different value was said in between; re-extracting an unchanged log still returns the existing proposals. On a case-insensitive file system, approving a proposal whose topic differs from an existing folder only in letter case files the record in that folder.
- Approving again after an interruption also works for restatements and returns to an earlier value, and a fact counts as claimed only by a proposal that states the same thing. Rejecting a proposal whose interrupted approval already wrote its fact is refused while that fact is unrevised and no other pending proposal for the same line can claim it.
- Sentence rules skip a line that holds more than one sentence instead of putting the second sentence into the value. A sentence end before a closing quote or bracket counts too. A period after a capital initial in any script or a two-letter abbreviation ("J. P. Morgan", "É. Macron", "St. Louis") does not end a sentence; lines with other abbreviations followed by a space ("Mrs. Jones", "e.g."), extracted in 0.2.2, are now skipped, so use the pipe form for them. A repeated line gives one proposal, also after another value was said in between; a byte-order mark at the start of any line is ignored, so text holding only byte-order marks is empty however many lines it has.
- Proposal-file problems name the file relative to the memory directory and never show an absolute path. Approving no longer reads every proposal file, and extraction looks lines up in an index of the queue.

Interfaces
- Git checkpoints ignore your personal Git ignore file, which could silently leave records out, and commit topic folders named like database files; the exclude rules of older repositories are rewritten in their intended order.
- The HTTP API checks the bearer token before reading a request body, also under a root path (`uvicorn --root-path`) or when mounted in another app. It sends `WWW-Authenticate: Bearer` with 401 and limits tags, links and time strings in length, including `as_of` and `known_at` in query strings.
- `serve --port` outside 0 to 65535 and an empty `--root` give the documented behavior instead of a traceback or using the current folder.

Packaging and verification
- The package ships a `py.typed` marker; the version has one source. CI adds Python 3.14 and a macOS job, with a job timeout.
- `tools/mutation_summary.py` reports a missing, failed or unfinished mutation run (mutants never checked) instead of a score, and exits 1; the benchmark charts handle a flat series.
- The test suite also passes when the temporary folder (TMPDIR) is inside a Git checkout.
- A new full mutation-testing run. Every surviving mutant was reviewed against the 0.2.3 code, including those already reviewed for 0.2.2, and each verdict checked by a second reviewer. 17 new tests catch the 31 mutants that showed untested behavior. The review found no bug in released code; it found one in a fix made for this release (`memoryos reindex` could refuse to repair an index whose page check happened to quote a missing column), corrected before publishing.
- Python API: the unused `embeddings.query_terms` helper is removed; the new `MemoryStore.existing_topic()` returns a topic spelled like its existing folder.

## 0.2.2

Fixes found by a full audit of the code, documentation and release process.

Memory store
- Times are parsed with one explicit grammar (ASCII digits only), so the same strings are accepted on every supported Python version. Forms that 0.2.0 accepted only on Python 3.11 or later, such as compact dates, are now rejected everywhere. Out-of-range times and timestamps at the end of the representable range are validation errors, not crashes.
- A failed `COMMIT` no longer leaves the new Markdown record behind. Records, their folders and new topic folders are flushed with `fsync`.
- A damaged index stops with a message; `memoryos reindex` sets it aside and rebuilds it. Locks, read-only files and full disks are never mistaken for damage. Concurrent first opens retry instead of failing with "database is locked". The index file is owner-readable only, including indexes created by older releases.
- Opening a store whose files have not changed no longer waits for a writer such as a running checkpoint.
- Dot-named files and folders (editor lock and swap files, hidden tool folders) are ignored. Notes are never decoded as records, so any encoding works. A file that held a record and lost its JSON block is reported instead of silently dropped. Records may use a byte-order mark and CRLF or CR line endings; a damaged record behind a byte-order mark, or a record saved as UTF-16 or UTF-32, is reported instead of skipped as a note, and so is a deeply nested record. A file name that is not valid UTF-8 is reported by name instead of stopping the store with an unrelated error. An index from 0.2.0 or 0.2.1 is read again from Markdown once on upgrade, because those releases skipped such records.
- A symbolic link under `memory/` is rejected even if it was swapped for a plain file while the index was being rebuilt. A link in place of the index's log files is rejected, and opening a store never changes the permissions of a file such a link points to.
- A datetime whose time zone gives no offset is read as UTC, like a naive one, instead of as the computer's local time.
- Error messages name the files involved, relative to the memory directory; problems in the files on disk are reported by the HTTP API as 503 and no longer disclose absolute paths.
- A record imported with a timestamp ahead of the local clock can be corrected and retracted, and does not hide new writes from current queries.
- `correct()` accepts `confidence` and `topic`; new `audit_trail()`. Tags and links are immutable tuples, and a single string is rejected instead of being split into letters.

Retrieval
- Rank fusion uses competition ranks: tied scores share a rank. Before, ties became rank differences decided by random fact IDs, which could change the top result of the default search.
- Document vectors are cached by model and exact text, compactly; the HTTP and MCP servers keep one model and cache per process instead of rebuilding them, and reloading a learned model, on every request. A model in the local cache loads without network access.
- Unicode-aware tokenizing (NFKC and case folding; combining marks stay inside words); keyword matching ignores accents. Common question words no longer make every fact match, while entity names made of such words ("The Who") are still found.
- Graph seeding tells apart names that share their words (C, C#, C++) and accepts names written only in symbols. A name counts as written out in quotes or brackets and with trailing punctuation or a possessive ("C++", St. Louis, Yahoo!, C++'s), and matching stays linear in the query's length.
- `MEMORYOS_SEMANTIC_THRESHOLD`; a threshold too large for a float, or embedding vectors whose length overflows, is a `ValueError` like any other out-of-range value; a model that cannot load raises `EmbeddingUnavailableError` (HTTP 503 without details, CLI JSON error).

Review queue
- Approving a dated conversation in order works when the attribute already had an earlier value. Approving again after an interruption recognizes the fact already written. Stale approvals, including an edited record that kept its ID, return HTTP 409, as documented, also when a newer approved value from the same source is present. In-order approval also accepts a later line that repeats the value already in effect (nothing new is written) or returns to an earlier value.
- Damaged proposal files are reported by name and skipped, so one bad file no longer blocks the queue. Lines with `|` that are not structured facts, lines with invalid or conflicting dates, and dated comments are skipped instead of aborting the extraction or becoming facts. Re-extracting undated lines returns the pending proposals instead of duplicates, unless a value has taken effect since (a value planned for later does not count). This also holds for proposals saved by 0.2.1 and for an empty or blank observation time, which means now. Text or a source that is not valid Unicode is rejected instead of yielding no proposals, and a proposal file holding such text, or nested too deeply to read, is reported like other damaged files. A time edited by hand in a proposal file may use any accepted form. Correction approvals keep the record's topic folder.
- Proposals extracted within one clock tick list in line order.

Interfaces
- Git checkpoints: inherited `GIT_DIR`-style and pathspec variables can no longer redirect or break a checkpoint; stores with tens of thousands of files are staged in linear time without hitting the argument limit; checkpoints wait for in-flight writes and approvals; temporary files, dot files and editor backups are never committed, while topic folders with any name are; `git-init` refuses an existing repository it did not create, and a repository you created yourself can be opted in. A Git timeout explains how to clear the lock it may leave.
- The HTTP API only answers requests addressed to localhost or to hosts listed in `MEMORYOS_ALLOWED_HOSTS`, and refuses cross-site requests that would change memory; `serve` warns when exposed without a token.
- CLI errors stay in the JSON error format; an empty `MEMORYOS_HOME` means the default directory; the demo never fails half way.
- Install hints point to the source checkout (the package is not published on PyPI). The `semantic` extra requires PyTorch 2.6 or later where it is available. Development dependencies install on Python 3.10.

Verification
- Regression tests for every fix above, tests for the release tooling, and benchmarks that report cold (one command) and warm (long-lived process) search separately and fail in CI if accuracy or charts drift.
- Every mutant that survived a full mutation-testing run was reviewed by hand and the verdict checked by a second reviewer. The review added tests for the behavior no earlier test checked (`tests/*_pins.py`) and found many of the fixes above; the remaining survivors are equivalent or change only message wording (see the README).

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
