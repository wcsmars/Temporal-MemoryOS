"""Markdown source of truth with a disposable SQLite temporal index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from .models import Fact, TimeInput, normalize_time

_MARKER = "<!-- memoryos:fact:v1 -->"
_MARKER_LINE = re.compile(r"^[ \t]*<!-- memoryos:fact:v1 -->[ \t]*\r?$", re.MULTILINE)
_BLOCK = re.compile(
    r"^[ \t]*<!-- memoryos:fact:v1 -->[ \t]*\r?\n[ \t]*```json[ \t]*\r?\n"
    r"(.*?)\r?\n[ \t]*```[ \t]*\r?$",
    re.DOTALL | re.MULTILINE,
)
# A file modified this close to its last verification may share a timestamp
# with a later edit on filesystems with coarse clocks, so its contents are
# hashed instead of trusting size and mtime (the "racy clean" rule Git uses).
_RACY_WINDOW_NS = 3_000_000_000
_SCHEMA_VERSION = 2
_COLUMNS = (
    "id", "entity", "attribute", "value", "valid_from", "valid_to", "recorded_at",
    "source", "evidence", "confidence", "tags", "links", "topic", "supersedes", "deleted",
)


def _bounded(fact: Fact, end: Optional[str]) -> Fact:
    """Copy a validated record with a derived end; the end is already normalized
    and later than ``valid_from`` because equal starts are rejected on write."""
    if end == fact.valid_to:
        return fact
    bounded = copy.copy(fact)
    object.__setattr__(bounded, "valid_to", end)
    return bounded


class MemoryConflictError(ValueError):
    """A proposed write conflicts with the existing assertion history."""


class MemoryStore:
    """Open a memory workspace; SQLite can always be rebuilt from Markdown.

    The index remembers the size, modification time and (for recently touched
    files) content hash of every Markdown file it was built from. Opening the
    store and every write compare that signature with the folder: if nothing
    changed outside the engine, the index is trusted and a write validates only
    the affected entity/attribute history; otherwise every record is validated
    and the index is rebuilt. Reads on a long-lived instance use the index as
    it is, so call :meth:`refresh` (cheap) or :meth:`reindex` (forced) after
    external edits. Every rebuild is all-or-nothing. Unmarked Markdown notes
    are left alone and not indexed.
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser().resolve()
        self.memory_dir = self.root / "memory"
        self.state_dir = self.root / ".memoryos"
        if self.memory_dir.is_symlink() or self.state_dir.is_symlink():
            raise ValueError("Memory and index folders must not be symbolic links")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "index.sqlite3"
        if self.db_path.is_symlink():
            raise ValueError("The index database must not be a symbolic link")
        self.connection = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None,
                                          check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        # Parsed records keyed by ID. The stored payload is compared on every
        # hit, so a rebuild by another process can never serve a stale record.
        self._parsed: Dict[str, Tuple[str, Fact]] = {}
        try:
            self.connection.execute("PRAGMA busy_timeout = 30000")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self._initialize()
            self.refresh()
        except Exception:
            self.connection.close()
            raise

    def _initialize(self) -> None:
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version > _SCHEMA_VERSION:
            raise ValueError("Index was created by a newer MemoryOS; rebuild it with a compatible version")
        # Version 1 indexes had no source signature. The table below starts
        # empty, which forces one full rebuild from Markdown on first refresh.
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY, entity TEXT NOT NULL, attribute TEXT NOT NULL,
                value TEXT NOT NULL, valid_from TEXT NOT NULL, valid_to TEXT,
                recorded_at TEXT NOT NULL, source TEXT NOT NULL, evidence TEXT NOT NULL,
                confidence REAL NOT NULL, tags TEXT NOT NULL, links TEXT NOT NULL,
                topic TEXT NOT NULL, supersedes TEXT, deleted INTEGER NOT NULL,
                path TEXT NOT NULL UNIQUE, payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS facts_temporal ON facts(entity, attribute, valid_from, recorded_at);
            CREATE INDEX IF NOT EXISTS facts_recorded ON facts(recorded_at);
            CREATE INDEX IF NOT EXISTS facts_supersedes ON facts(supersedes);
            -- Keyword search builds its FTS table per query over the time-filtered
            -- snapshot, so a persistent one would only cost writes.
            DROP TABLE IF EXISTS facts_fts;
            CREATE TABLE IF NOT EXISTS sources (
                path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL, verified_ns INTEGER NOT NULL
            );
        """)
        self.connection.execute("PRAGMA user_version = {}".format(_SCHEMA_VERSION))

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.connection.execute("COMMIT")
            except BaseException:
                if self.connection.in_transaction:
                    self.connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _render(fact: Fact) -> str:
        title = "{} · {}".format(fact.entity, fact.attribute).replace("\n", " ").replace("\r", " ")
        return (
            "# " + title + "\n\n"
            "The JSON block below is canonical. Edits are picked up the next time the store opens.\n\n"
            + _MARKER + "\n```json\n"
            + json.dumps(fact.to_dict(), ensure_ascii=False, indent=2)
            + "\n```\n"
        )

    def _scan(self) -> Dict[str, Tuple[int, int]]:
        """Stat every Markdown file; a symlink makes the scan look changed."""
        found: Dict[str, Tuple[int, int]] = {}
        pending = [(str(self.memory_dir), "memory/")]
        while pending:
            directory, prefix = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = prefix + entry.name
                    if entry.is_dir(follow_symlinks=False):
                        pending.append((entry.path, relative + "/"))
                    elif entry.name.endswith(".md"):
                        if entry.is_symlink():
                            found[relative] = (-1, -1)
                        else:
                            status = entry.stat(follow_symlinks=False)
                            found[relative] = (status.st_size, status.st_mtime_ns)
                    elif entry.is_symlink() and entry.is_dir():
                        found[relative] = (-1, -1)  # Linked folders are rejected on rebuild.
        return found

    def _is_current(self, scan: Dict[str, Tuple[int, int]]) -> bool:
        """True when the index was built from exactly the files on disk."""
        rows = {row["path"]: row for row in self.connection.execute(
            "SELECT path, size, mtime_ns, sha256, verified_ns FROM sources")}
        if rows.keys() != scan.keys():
            return False
        if self.connection.execute(
                "SELECT 1 FROM facts WHERE path NOT IN (SELECT path FROM sources) LIMIT 1").fetchone():
            return False
        racy = []
        for relative, (size, mtime_ns) in scan.items():
            row = rows[relative]
            if (size, mtime_ns) != (row["size"], row["mtime_ns"]):
                return False
            if mtime_ns + _RACY_WINDOW_NS >= row["verified_ns"]:
                racy.append(relative)
        now = time.time_ns()
        for relative in racy:
            digest = hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
            if digest != rows[relative]["sha256"]:
                return False
        # Contents confirmed; these files stop being racy once time moves on.
        self.connection.executemany("UPDATE sources SET verified_ns = ? WHERE path = ?",
                                    [(now, relative) for relative in racy])
        return True

    @staticmethod
    def _parse(path: Path, raw: bytes) -> Optional[Fact]:
        text = raw.decode("utf-8")
        markers = _MARKER_LINE.findall(text)
        if not markers:
            return None
        matches = list(_BLOCK.finditer(text))
        if len(matches) != 1 or len(markers) != 1:
            raise ValueError("{}: expected one canonical memory JSON block".format(path))
        try:
            return Fact.from_dict(json.loads(matches[0].group(1)))
        except (ValueError, TypeError) as exc:
            raise ValueError("{}: {}".format(path, exc)) from exc

    def _load_markdown(self, scan: Optional[Dict[str, Tuple[int, int]]] = None
                       ) -> Tuple[List[Tuple[Fact, Path]], List[tuple]]:
        """Parse and validate every record; return records and source signatures.

        ``scan`` must be taken before reading. If a file changes after its
        stat, the stored signature is stale and the next refresh rebuilds.
        """
        scan = self._scan() if scan is None else scan
        now = time.time_ns()
        records: List[Tuple[Fact, Path]] = []
        sources: List[tuple] = []
        memory_root = self.memory_dir.resolve()
        for relative in sorted(scan):
            path = self.root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(memory_root):
                raise ValueError("Memory records must not be symbolic links or escape the memory folder")
            raw = path.read_bytes()
            size, mtime_ns = scan[relative]
            sources.append((relative, size, mtime_ns, hashlib.sha256(raw).hexdigest(), now))
            fact = self._parse(path, raw)
            if fact is not None:
                records.append((fact, path))
        self._validate_collection([fact for fact, _ in records])
        return records, sources

    @staticmethod
    def _validate_collection(facts: List[Fact]) -> None:
        by_id: Dict[str, Fact] = {}
        for fact in facts:
            if fact.id in by_id:
                raise ValueError("Duplicate memory id: " + fact.id)
            by_id[fact.id] = fact
        successors: Dict[str, str] = {}
        for fact in facts:
            if fact.supersedes is None:
                continue
            if fact.supersedes not in by_id:
                raise ValueError("Missing superseded record: " + fact.supersedes)
            old = by_id[fact.supersedes]
            if (fact.entity, fact.attribute, fact.valid_from) != (old.entity, old.attribute, old.valid_from):
                raise ValueError("Corrections must retain entity, attribute, and valid_from")
            if fact.recorded_at < old.recorded_at:
                raise ValueError("A correction cannot be recorded before the assertion it corrects")
            if old.id in successors:
                raise MemoryConflictError("Two corrections supersede the same record: " + old.id)
            successors[old.id] = fact.id
        for fact in facts:
            visited = set()
            current = fact
            while current.supersedes:
                if current.id in visited:
                    raise ValueError("Cycle in memory correction history")
                visited.add(current.id)
                current = by_id[current.supersedes]
        effective: Dict[Tuple[str, str, str], str] = {}
        for fact in facts:
            if fact.id in successors:
                continue
            key = (fact.entity, fact.attribute, fact.valid_from)
            if key in effective:
                raise MemoryConflictError("Duplicate effective time for {} / {}; use correct()".format(
                    fact.entity, fact.attribute))
            effective[key] = fact.id

    def _index_record(self, fact: Fact, path: Path) -> None:
        data = fact.to_dict()
        values = [json.dumps(data[key], ensure_ascii=False) if key in ("tags", "links")
                  else int(data[key]) if key == "deleted" else data[key] for key in _COLUMNS]
        values.extend([path.relative_to(self.root).as_posix(), json.dumps(data, ensure_ascii=False)])
        self.connection.execute("INSERT INTO facts ({}, path, payload) VALUES ({})".format(
            ", ".join(_COLUMNS), ", ".join("?" for _ in values)), values)

    def _replace_index(self, records: List[Tuple[Fact, Path]], sources: List[tuple]) -> None:
        self.connection.execute("DELETE FROM facts")
        self.connection.execute("DELETE FROM sources")
        for fact, path in records:
            self._index_record(fact, path)
        self.connection.executemany("INSERT INTO sources VALUES (?, ?, ?, ?, ?)", sources)

    def _rebuild(self) -> int:
        records, sources = self._load_markdown()
        self._replace_index(records, sources)
        return len(records)

    def reindex(self) -> int:
        """Validate every Markdown record and atomically replace the index."""
        with self._transaction():
            return self._rebuild()

    def _refresh_locked(self) -> bool:
        scan = self._scan()
        if self._is_current(scan):
            return False
        records, sources = self._load_markdown(scan)
        self._replace_index(records, sources)
        return True

    def refresh(self) -> bool:
        """Rebuild only if Markdown changed outside the engine; return whether it did."""
        with self._transaction():
            return self._refresh_locked()

    def _group(self, entity: str, attribute: str) -> List[Fact]:
        rows = self.connection.execute(
            "SELECT id, payload FROM facts WHERE entity = ? AND attribute = ?", (entity, attribute))
        return [self._fact(row) for row in rows]

    def _fact(self, row: sqlite3.Row) -> Fact:
        payload = row["payload"]
        cached = self._parsed.get(row["id"])
        if cached is not None and cached[0] == payload:
            return cached[1]
        fact = Fact.from_dict(json.loads(payload))
        if len(self._parsed) >= 200_000:
            self._parsed.clear()
        self._parsed[row["id"]] = (payload, fact)
        return fact

    def _indexed(self, fact_id: str) -> Optional[Fact]:
        row = self.connection.execute("SELECT id, payload FROM facts WHERE id = ?",
                                      (fact_id,)).fetchone()
        return None if row is None else self._fact(row)

    def _write(self, fact: Fact) -> Fact:
        # Every rule in _validate_collection is local to one entity/attribute
        # history (corrections keep both), except global ID uniqueness.
        if self._indexed(fact.id) is not None:
            raise ValueError("Duplicate memory id: " + fact.id)
        self._validate_collection(self._group(fact.entity, fact.attribute) + [fact])
        path = self.memory_dir / fact.topic / (fact.id + ".md")
        if not path.parent.resolve().is_relative_to(self.memory_dir.resolve()):
            raise ValueError("Memory topic must not escape the memory folder through a symbolic link")
        for parent in path.parents:
            if parent == self.memory_dir:
                break
            if parent.is_symlink():
                raise ValueError("Memory topic folders must not be symbolic links")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise MemoryConflictError("Refusing to overwrite an existing Markdown record")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent),
                                             prefix=".memoryos-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(self._render(fact))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(path))
            temporary = None
            self._index_record(fact, path)
            status = path.stat()
            self.connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                (path.relative_to(self.root).as_posix(), status.st_size, status.st_mtime_ns,
                 hashlib.sha256(path.read_bytes()).hexdigest(), time.time_ns()))
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            # The file and SQLite are updated under the same write lock. If
            # indexing fails, remove only the new record; older files are intact.
            path.unlink(missing_ok=True)
            raise
        return fact

    def _recorded_time(self, value: TimeInput) -> str:
        if value is not None:
            return normalize_time(value)
        now = normalize_time()
        # Break an exact clock tie without moving new observations into the
        # future merely because a caller explicitly imported a future record.
        latest = self.connection.execute("SELECT max(recorded_at) FROM facts").fetchone()[0]
        if latest == now:
            instant = datetime.fromisoformat(latest.replace("Z", "+00:00")) + timedelta(microseconds=1)
            return instant.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        return now

    def add(self, entity: str, attribute: str, value: str, valid_from: TimeInput = None,
            valid_to: TimeInput = None, source: str = "manual", evidence: str = "",
            confidence: float = 1.0, tags: Optional[List[str]] = None,
            links: Optional[List[str]] = None, topic: str = "general",
            recorded_at: TimeInput = None) -> Fact:
        with self._transaction():
            self._refresh_locked()
            fact = Fact(id=uuid.uuid4().hex, entity=entity, attribute=attribute, value=value,
                        valid_from=normalize_time(valid_from),
                        valid_to=None if valid_to is None else normalize_time(valid_to),
                        recorded_at=self._recorded_time(recorded_at), source=source,
                        evidence=evidence, confidence=confidence,
                        tags=[] if tags is None else tags,
                        links=[] if links is None else links, topic=topic)
            return self._write(fact)

    def _revision(self, fact_id: str, value: Optional[str], source: str, evidence: str,
                  recorded_at: TimeInput, deleted: bool) -> Fact:
        with self._transaction():
            self._refresh_locked()
            old = self._indexed(fact_id)
            if old is None:
                raise KeyError(fact_id)
            if self.connection.execute("SELECT 1 FROM facts WHERE supersedes = ?",
                                       (fact_id,)).fetchone():
                raise MemoryConflictError("This record has already been superseded; correct the latest revision")
            if deleted and old.deleted:
                raise MemoryConflictError("This record has already been retracted")
            fact = replace(old, id=uuid.uuid4().hex, value="" if deleted or value is None else value,
                           source=source, evidence=evidence, deleted=deleted,
                           recorded_at=self._recorded_time(recorded_at), supersedes=old.id)
            return self._write(fact)

    def correct(self, fact_id: str, value: str, source: str = "manual", evidence: str = "",
                recorded_at: TimeInput = None) -> Fact:
        """Revise an assertion at its original effective time; keep the old record."""
        return self._revision(fact_id, value, source, evidence, recorded_at, False)

    def retract(self, fact_id: str, source: str = "manual", recorded_at: TimeInput = None) -> Fact:
        """Retract one effective interval, retaining its audit history in Markdown."""
        return self._revision(fact_id, None, source, "Retracted assertion", recorded_at, True)

    def get(self, fact_id: str) -> Fact:
        with self._lock:
            fact = self._indexed(fact_id)
        if fact is None:
            raise KeyError(fact_id)
        return fact

    def path_for(self, fact_id: str) -> Path:
        with self._lock:
            row = self.connection.execute("SELECT path FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if row is None:
            raise KeyError(fact_id)
        return self.root / str(row["path"])

    def all_facts(self) -> List[Fact]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT id, payload FROM facts ORDER BY recorded_at, id").fetchall()
            return [self._fact(row) for row in rows]

    def _visible(self, known: str, entity: Optional[str] = None,
                 attribute: Optional[str] = None) -> List[Fact]:
        clauses, parameters = ["recorded_at <= ?"], [known]
        if entity is not None:
            clauses.append("entity = ?")
            parameters.append(entity)
        if attribute is not None:
            clauses.append("attribute = ?")
            parameters.append(attribute)
        with self._lock:
            rows = self.connection.execute(
                "SELECT id, payload FROM facts WHERE " + " AND ".join(clauses)
                + " ORDER BY entity, attribute, valid_from, recorded_at, id", parameters).fetchall()
            return [self._fact(row) for row in rows]

    def _timeline(self, known_at: TimeInput = None, entity: Optional[str] = None,
                  attribute: Optional[str] = None) -> List[Fact]:
        visible = self._visible(normalize_time(known_at), entity, attribute)
        # Revisions keep entity, attribute and valid_from, so a filtered query
        # still sees every revision that could supersede a visible record.
        superseded = {fact.supersedes for fact in visible if fact.supersedes is not None}
        groups: Dict[Tuple[str, str], List[Fact]] = {}
        for fact in visible:
            if fact.id not in superseded:
                groups.setdefault((fact.entity, fact.attribute), []).append(fact)
        result: List[Fact] = []
        for key in sorted(groups):
            timeline = groups[key]  # Already ordered by valid_from, recorded_at, id.
            for index, fact in enumerate(timeline):
                end = fact.valid_to
                if index + 1 < len(timeline):
                    following = timeline[index + 1].valid_from
                    end = min(end, following) if end is not None else following
                result.append(_bounded(fact, end))
        return result

    def snapshot(self, as_of: TimeInput = None, known_at: TimeInput = None,
                 entity: Optional[str] = None) -> List[Fact]:
        """Return what held at ``as_of``, using only knowledge recorded by ``known_at``.

        Both default to now. Corrections learned later change retrospective
        truth, while a historical ``known_at`` reproduces earlier knowledge.
        ``entity`` optionally restricts the snapshot to one exact entity name.
        """
        effective = normalize_time(as_of)
        return [fact for fact in self._timeline(known_at, entity)
                if not fact.deleted and fact.valid_from <= effective
                and (fact.valid_to is None or effective < fact.valid_to)]

    def history(self, entity: str, attribute: Optional[str] = None,
                known_at: TimeInput = None) -> List[Fact]:
        """Return the effective timeline, including retracted intervals.

        Superseded revisions remain available through ``get``/``all_facts``;
        history shows the timeline as it was known at the requested cutoff.
        """
        return sorted(self._timeline(known_at, entity, attribute),
                      key=lambda fact: (fact.valid_from, fact.attribute, fact.recorded_at, fact.id))

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
