"""Markdown source of truth with a disposable SQLite temporal index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
import threading
import time
import uuid
import copy
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .models import Fact, TimeInput, normalize_time

_MARKER = "<!-- memoryos:fact:v1 -->"
_MARKER_BYTES = _MARKER.encode("utf-8")
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
# 3: the persistent full-text table is no longer written or read. Releases
# before 0.2.1 refuse a version 3 index; a table left by them is kept so a
# process of theirs that is still running does not fail on its next write.
_SCHEMA_VERSION = 3
# Writes this close together (or a clock this far behind) stay strictly ordered.
_TIE_WINDOW = timedelta(milliseconds=10)
_LOCK_RETRY_SECONDS = 30.0
_COLUMNS = (
    "id", "entity", "attribute", "value", "valid_from", "valid_to", "recorded_at",
    "source", "evidence", "confidence", "tags", "links", "topic", "supersedes", "deleted",
)
# Every column of the engine's tables, in schema order (see _initialize).
_TABLE_COLUMNS = {
    "facts": _COLUMNS + ("path", "payload"),
    "sources": ("path", "size", "mtime_ns", "sha256", "verified_ns"),
}


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


class StoreIntegrityError(ValueError):
    """The Markdown records or the index on disk need repair; the request was fine."""


def _shift(timestamp: str, delta: timedelta) -> str:
    try:
        instant = datetime.fromisoformat(timestamp.replace("Z", "+00:00")) + delta
    except OverflowError as exc:
        raise ValueError("No later time can be recorded after {}".format(timestamp)) from exc
    return instant.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _after(timestamp: str) -> str:
    """The normalized instant one microsecond after ``timestamp``."""
    return _shift(timestamp, timedelta(microseconds=1))


def _flush(descriptor: int) -> None:
    """fsync, matching SQLite's default durability.

    On macOS this reaches the drive's cache, not necessarily the platter or
    flash, exactly like SQLite with ``PRAGMA fullfsync`` off. A full flush
    would cost several milliseconds per write.
    """
    os.fsync(descriptor)


def _flush_directory(directory: Path) -> None:
    """Persist a rename by flushing its directory (POSIX); skipped where unsupported."""
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        _flush(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _locked(error: sqlite3.Error) -> bool:
    text = str(error).lower()
    return isinstance(error, sqlite3.OperationalError) and ("locked" in text or "busy" in text)


def _missing_column(error: sqlite3.Error) -> bool:
    """True when SQLite reports that a statement names a column the table does not have.

    Only SQLite's own wording counts, at the start of the message: a damage report that
    merely quotes such text (a table named that way, say) is still damage.
    """
    text = str(error).lower()
    return (text.startswith("no such column: ")
            or text.startswith("table ") and " has no column named " in text)


def _damaged(error: sqlite3.Error) -> bool:
    """True only for SQLite's corruption errors, never for locks, permissions or full disks.

    A damaged schema page reads as "malformed database schema". Python 3.11 and later also
    give the result code: SQLITE_CORRUPT (11) or SQLITE_NOTADB (26), with or without an
    extended code. A damaged column name still parses; it shows only when a query names
    the column, as a missing column.
    """
    code = getattr(error, "sqlite_errorcode", None)
    if isinstance(code, int) and code & 0xFF in (11, 26):
        return True
    text = str(error).lower()
    if type(error) is sqlite3.DatabaseError and not text:
        return True  # Python 3.10: SQLite's message quoted damaged bytes it could not decode.
    return ("file is not a database" in text or "disk image is malformed" in text
            or "malformed database schema" in text
            # Text the engine wrote is always UTF-8; anything else is a damaged cell.
            or "could not decode to utf-8" in text
            # Queries name only columns the engine created (a new layout gets a new
            # schema version), so in an open index a missing one is a damaged name.
            or _missing_column(error))


_DAMAGED_INDEX = ("The index .memoryos/index.sqlite3 is damaged ({}). Stop other programs using "
                  "this memory directory and run `memoryos reindex` to rebuild it from the "
                  "Markdown records.")


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
    are left alone and not indexed, and names starting with a dot (editor
    lock and swap files, hidden folders) are ignored.

    A damaged index raises :class:`StoreIntegrityError`. With
    ``repair_index=True`` (what ``memoryos reindex`` uses) a damaged index file
    is set aside and rebuilt instead, since everything in it is derived from
    Markdown. Repair renames database files, so it is never done implicitly
    while other processes may be using them.
    """

    def __init__(self, root: Union[str, Path], repair_index: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.memory_dir = self.root / "memory"
        self.state_dir = self.root / ".memoryos"
        # The folders on disk need attention, not the caller's request.
        if self.memory_dir.is_symlink() or self.state_dir.is_symlink():
            raise StoreIntegrityError("Memory and index folders must not be symbolic links")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "index.sqlite3"
        if any(Path(str(self.db_path) + suffix).is_symlink() for suffix in ("", "-wal", "-shm")):
            raise StoreIntegrityError("The index database must not be a symbolic link")
        self._lock = threading.RLock()
        # Parsed records keyed by ID. The stored payload is compared on every
        # hit, so a rebuild by another process can never serve a stale record.
        self._parsed: Dict[str, Tuple[str, Fact]] = {}
        self._undo: List[Callable[[], None]] = []
        self.connection = self._open(repair_index)
        try:
            self.refresh()
        except sqlite3.DatabaseError as error:
            self.connection.close()
            if not (_damaged(error) and repair_index):
                if _damaged(error):
                    raise StoreIntegrityError(_DAMAGED_INDEX.format(error)) from error
                raise
            # Damage past the header shows up only when pages are read.
            self._set_aside()
            self.connection = self._open(False)
            try:
                self.refresh()
            except Exception:
                self.connection.close()
                raise
        except Exception:
            self.connection.close()
            raise

    def _set_aside(self) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        taken = {path.name for path in self.state_dir.iterdir()}
        attempt = 0
        while any("{}{}.corrupt-{}{}".format(self.db_path.name, suffix, stamp,
                                               "-{}".format(attempt) if attempt else "") in taken
                  for suffix in ("", "-wal", "-shm")):
            attempt += 1  # A second repair in the same second keeps the first copy.
        label = stamp + ("-{}".format(attempt) if attempt else "")
        for suffix in ("", "-wal", "-shm"):
            damaged = Path(str(self.db_path) + suffix)
            if damaged.exists():
                os.replace(str(damaged), "{}.corrupt-{}".format(damaged, label))

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            # Create it private, like the Markdown records it mirrors.
            os.close(os.open(str(self.db_path), os.O_CREAT | os.O_WRONLY, 0o600))
        # Indexes made by older releases were world-readable; tighten them.
        for suffix in ("", "-wal", "-shm"):
            try:
                path = Path(str(self.db_path) + suffix)
                status = path.lstat()  # Never change a file that a link points to.
                if (stat.S_ISREG(status.st_mode) and status.st_uid == os.getuid()
                        and status.st_mode & 0o077):
                    path.chmod(0o600)
            except (OSError, AttributeError):
                pass
        connection = sqlite3.connect(str(self.db_path), timeout=_LOCK_RETRY_SECONDS,
                                     isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _open(self, repair: bool) -> sqlite3.Connection:
        """Connect and prepare the schema, retrying first-open races. A damaged
        index file is set aside only when ``repair`` is true."""
        deadline = time.monotonic() + _LOCK_RETRY_SECONDS
        repaired = False
        while True:
            connection = self._connect()
            try:
                connection.execute("PRAGMA busy_timeout = {}".format(int(_LOCK_RETRY_SECONDS * 1000)))
                connection.execute("PRAGMA journal_mode = WAL")
                self._initialize(connection)
                if repair:
                    # Damage in a page that opening and refresh never read (a record
                    # or a secondary index) would otherwise fail the rebuild itself.
                    problems = [row[0] for row in connection.execute("PRAGMA quick_check")]
                    if problems != ["ok"]:
                        raise sqlite3.DatabaseError(
                            "database disk image is malformed ({})".format(problems[0]))
                    # A damaged column name in the schema still parses, so quick_check
                    # accepts it; every query that names the column then fails.
                    for table, columns in _TABLE_COLUMNS.items():
                        # Fetched at once, so a name that cannot be decoded leaves no
                        # statement open when the connection is closed below.
                        rows = connection.execute("PRAGMA table_info({})".format(table)).fetchall()
                        found = tuple(row[1] for row in rows)
                        if found != columns:
                            raise sqlite3.DatabaseError(
                                "database disk image is malformed (columns of {})".format(table))
                return connection
            except (sqlite3.DatabaseError, UnicodeDecodeError) as raised:
                connection.close()
                # Python 3.11+: SQLite's message quoted damaged schema bytes, so even the
                # error could not be decoded. Only the index file is read here.
                error = (sqlite3.DatabaseError("malformed database schema (unreadable text)")
                         if isinstance(raised, UnicodeDecodeError) else raised)
                if _locked(error) and time.monotonic() < deadline:
                    time.sleep(0.05)  # Another process is creating the schema.
                    continue
                # Locks, permissions, full disks: never a reason to discard data. Nor is a
                # missing column here: only the upgrade of an older index names columns,
                # so that table layout was never written by this release.
                if not _damaged(error) or _missing_column(error):
                    raise
                if not repair or repaired:
                    raise StoreIntegrityError(_DAMAGED_INDEX.format(error)) from error
                repaired = True
                self._set_aside()
            except BaseException:
                connection.close()
                raise

    def _initialize(self, connection: sqlite3.Connection) -> None:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == _SCHEMA_VERSION:
            return  # Nothing to write, so opening never waits for another writer.
        if version > _SCHEMA_VERSION:
            raise StoreIntegrityError(
                "The index was written by a newer MemoryOS. Delete the .memoryos folder inside "
                "the memory directory to rebuild it for this version (Markdown records are not "
                "affected).")
        # Version 1 indexes had no source signature. The table below starts
        # empty, which forces one full rebuild from Markdown on first refresh.
        connection.executescript("""
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
            CREATE TABLE IF NOT EXISTS sources (
                path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL, verified_ns INTEGER NOT NULL
            );
        """)
        # Releases before 0.2.2 skipped some records as notes (a byte-order mark
        # before the marker line, CR-only line endings). Forgetting the source
        # signatures makes the first refresh read every file again.
        connection.execute("DELETE FROM sources")
        connection.execute("PRAGMA user_version = {}".format(_SCHEMA_VERSION))

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            self._undo = []
            try:
                yield
                self.connection.execute("COMMIT")
            except BaseException:
                if self.connection.in_transaction:
                    self.connection.execute("ROLLBACK")
                # Files written in this transaction must not outlive it, even
                # when the failure is the COMMIT itself.
                for undo in reversed(self._undo):
                    undo()
                raise
            finally:
                self._undo = []

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
        """Stat every Markdown file; a symlink makes the scan look changed.

        Names starting with a dot are skipped: editors create lock and swap
        files that way (Emacs's lock file is a dangling ``.#name.md`` link),
        and hidden folders belong to other tools.
        """
        found: Dict[str, Tuple[int, int]] = {}
        pending = [(str(self.memory_dir), "memory/")]
        while pending:
            directory, prefix = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
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

    def _check(self, scan: Dict[str, Tuple[int, int]]) -> Optional[List[str]]:
        """None if the index is stale; otherwise the files whose contents were
        hashed because their timestamps were too recent to trust."""
        rows = {row["path"]: row for row in self.connection.execute(
            "SELECT path, size, mtime_ns, sha256, verified_ns FROM sources")}
        if rows.keys() != scan.keys():
            return None
        if self.connection.execute(
                "SELECT 1 FROM facts WHERE path NOT IN (SELECT path FROM sources) LIMIT 1").fetchone():
            return None
        racy = []
        for relative, (size, mtime_ns) in scan.items():
            row = rows[relative]
            # A link always forces a rebuild, which rejects it: its placeholder
            # signature can be stored if the link was replaced during a read.
            if size < 0 or (size, mtime_ns) != (row["size"], row["mtime_ns"]):
                return None
            if mtime_ns + _RACY_WINDOW_NS >= row["verified_ns"]:
                racy.append(relative)
        for relative in racy:
            digest = hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
            if digest != rows[relative]["sha256"]:
                return None
        return racy

    def _is_current(self, scan: Dict[str, Tuple[int, int]]) -> bool:
        """True when the index was built from exactly the files on disk."""
        racy = self._check(scan)
        if racy is None:
            return False
        # Contents confirmed; these files stop being racy once time moves on.
        now = time.time_ns()
        self.connection.executemany("UPDATE sources SET verified_ns = ? WHERE path = ?",
                                    [(now, relative) for relative in racy])
        return True

    @staticmethod
    def _parse(path: Path, raw: bytes) -> Optional[Fact]:
        """Return the record in a file, or None for an ordinary note.

        ``path`` is only used in messages; pass it relative to the store so
        errors never disclose where the store lives.
        """
        if _MARKER_BYTES not in raw:
            # A record saved as UTF-16 (some editors call it "Unicode") holds no UTF-8
            # marker bytes; report it rather than skip it as a note.
            for codec in ("utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"):
                if _MARKER.encode(codec) in raw and _MARKER_LINE.search(
                        raw.decode(codec, errors="replace").lstrip("\ufeff").replace("\r", "\n")):
                    raise ValueError("{}: record is not valid UTF-8 (saved as {}?)".format(
                        path.as_posix(), codec[:6].upper()))
            return None  # A note; it is never decoded, so any encoding is fine.
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            # utf-8-sig, like the strict decode, so a byte-order mark in front of
            # the marker line does not hide a damaged record.
            lenient = raw.decode("utf-8-sig", errors="replace").replace("\r", "\n")
            if not _MARKER_LINE.search(lenient):
                return None  # A note that mentions the marker text inline.
            raise ValueError("{}: record is not valid UTF-8".format(path.as_posix())) from exc
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        markers = _MARKER_LINE.findall(text)
        if not markers:
            return None  # The marker text appears, but not as a marker line.
        matches = list(_BLOCK.finditer(text))
        if len(matches) != 1 or len(markers) != 1:
            raise ValueError("{}: expected one canonical memory JSON block".format(path.as_posix()))
        try:
            return Fact.from_dict(json.loads(matches[0].group(1)))
        except (ValueError, TypeError, RecursionError) as exc:  # Deep nesting: RecursionError.
            raise ValueError("{}: {}".format(path.as_posix(), exc)) from exc

    def _load_markdown(self, scan: Optional[Dict[str, Tuple[int, int]]] = None
                       ) -> Tuple[List[Tuple[Fact, Path]], List[tuple]]:
        """Parse and validate every record; return records and source signatures.

        ``scan`` must be taken before reading. If a file changes after its
        stat, the stored signature is stale and the next refresh rebuilds.
        """
        try:
            return self._load(self._scan() if scan is None else scan)
        except StoreIntegrityError:
            raise
        except ValueError as exc:
            # A problem with the files on disk, not with the caller's request.
            raise StoreIntegrityError(str(exc)) from exc

    def _load(self, scan: Dict[str, Tuple[int, int]]
              ) -> Tuple[List[Tuple[Fact, Path]], List[tuple]]:
        now = time.time_ns()
        records: List[Tuple[Fact, Path]] = []
        sources: List[tuple] = []
        memory_root = self.memory_dir.resolve()
        # Files the index knew as records. One that no longer holds a record was
        # emptied, truncated or had its marker damaged: dropping it silently could
        # bring back a value that it had superseded.
        known = {row["path"] for row in self.connection.execute("SELECT path FROM facts")}
        seen: Dict[str, str] = {}
        for relative in sorted(scan):
            try:
                relative.encode("utf-8")
            except UnicodeEncodeError:  # Undecodable bytes in a POSIX file name.
                raise ValueError("{}: the file name is not valid UTF-8; rename the file".format(
                    relative.encode("utf-8", "backslashreplace").decode("utf-8"))) from None
            path = self.root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(memory_root):
                raise ValueError("{}: memory files and folders must not be symbolic links".format(
                    relative))
            raw = path.read_bytes()
            size, mtime_ns = scan[relative]
            sources.append((relative, size, mtime_ns, hashlib.sha256(raw).hexdigest(), now))
            fact = self._parse(Path(relative), raw)
            if fact is not None:
                if fact.id in seen:
                    raise ValueError("Duplicate memory id: {} ({}, {})".format(
                        fact.id, seen[fact.id], relative))
                seen[fact.id] = relative
                records.append((fact, path))
            elif relative in known:
                raise ValueError(
                    "{}: this file held a memory record but no longer has a valid record block "
                    "(emptied, truncated, or marker line changed?); restore it, or delete the "
                    "file to remove the record".format(relative))
        self._validate_collection(
            [fact for fact, _ in records],
            {fact.id: path.relative_to(self.root).as_posix() for fact, path in records})
        return records, sources

    @staticmethod
    def _validate_collection(facts: List[Fact], paths: Optional[Dict[str, str]] = None) -> None:
        """Check the rules that span records. ``paths`` (ID to relative file) makes
        messages about files on disk name the files involved."""
        def where(*ids: str) -> str:
            names = [paths[fact_id] for fact_id in ids if paths and fact_id in paths]
            return " ({})".format(", ".join(names)) if names else ""

        by_id: Dict[str, Fact] = {}
        for fact in facts:
            if fact.id in by_id:
                raise ValueError("Duplicate memory id: " + fact.id + where(fact.id))
            by_id[fact.id] = fact
        successors: Dict[str, str] = {}
        for fact in facts:
            if fact.supersedes is None:
                continue
            if fact.supersedes not in by_id:
                raise ValueError("Missing superseded record: " + fact.supersedes + where(fact.id))
            old = by_id[fact.supersedes]
            if (fact.entity, fact.attribute, fact.valid_from) != (old.entity, old.attribute, old.valid_from):
                raise ValueError("Corrections must retain entity, attribute, and valid_from"
                                 + where(fact.id, old.id))
            if fact.recorded_at < old.recorded_at:
                raise ValueError("A correction cannot be recorded before the assertion it corrects"
                                 + where(fact.id, old.id))
            if old.id in successors:
                raise MemoryConflictError("Two corrections supersede the same record: " + old.id
                                          + where(successors[old.id], fact.id))
            successors[old.id] = fact.id
        for fact in facts:
            visited = set()
            current = fact
            while current.supersedes:
                if current.id in visited:
                    raise ValueError("Cycle in memory correction history" + where(fact.id))
                visited.add(current.id)
                current = by_id[current.supersedes]
        effective: Dict[Tuple[str, str, str], str] = {}
        for fact in facts:
            if fact.id in successors:
                continue
            key = (fact.entity, fact.attribute, fact.valid_from)
            if key in effective:
                raise MemoryConflictError("Duplicate effective time for {} / {}; use correct()".format(
                    fact.entity, fact.attribute) + where(effective[key], fact.id))
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
        """Rebuild only if Markdown changed outside the engine; return whether it did.

        The common case (nothing changed) is answered in a read transaction, so
        opening a store does not wait for a writer such as a Git checkpoint.
        """
        with self._lock:
            scan = self._scan()
            self.connection.execute("BEGIN")
            try:
                racy = self._check(scan)
            finally:
                self.connection.execute("COMMIT")
            if racy is not None:
                if racy:
                    self._mark_verified()
                return False
        with self._transaction():
            return self._refresh_locked()

    def _mark_verified(self) -> None:
        """Record that recently written files matched their hashes, if the write
        lock is free right now; otherwise leave it for later rather than wait.
        Without this, every open would hash those files again."""
        timeout = int(_LOCK_RETRY_SECONDS * 1000)
        self.connection.execute("PRAGMA busy_timeout = 0")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            return  # A writer (such as a checkpoint) holds the lock, or the index is read-only.
        finally:
            self.connection.execute("PRAGMA busy_timeout = {}".format(timeout))
        try:
            self._is_current(self._scan())  # Re-checks under the lock, then marks.
            self.connection.execute("COMMIT")
        except sqlite3.OperationalError:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")  # Marking is only an optimization.
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def _group(self, entity: str, attribute: str) -> List[Fact]:
        rows = self.connection.execute(
            "SELECT id, payload FROM facts WHERE entity = ? AND attribute = ?", (entity, attribute))
        return [self._fact(row) for row in rows]

    def _fact(self, row: sqlite3.Row) -> Fact:
        payload = row["payload"]
        cached = self._parsed.get(row["id"])
        if cached is not None and cached[0] == payload:
            return cached[1]
        try:
            fact = Fact.from_dict(json.loads(payload))
        except (ValueError, TypeError, RecursionError) as error:
            # The engine stored a valid record here, so the index cell was damaged
            # (SQLite does not checksum cell contents); the request was fine.
            raise StoreIntegrityError(_DAMAGED_INDEX.format(
                "record {} cannot be read: {}".format(row["id"], error))) from error
        if len(self._parsed) >= 200_000:
            self._parsed.clear()
        self._parsed[row["id"]] = (payload, fact)
        return fact

    def _indexed(self, fact_id: str) -> Optional[Fact]:
        row = self.connection.execute("SELECT id, payload FROM facts WHERE id = ?",
                                      (fact_id,)).fetchone()
        return None if row is None else self._fact(row)

    def _check_topic_spelling(self, topic: str) -> None:
        """Refuse a topic whose folder exists only in another letter case.

        A case-insensitive file system (the macOS default) would put the record in
        the existing folder while the index kept the given spelling, so every later
        refresh would rebuild and an emptied record there would go unreported.
        """
        folder = self.memory_dir
        segments = topic.split("/")
        for depth, segment in enumerate(segments):
            if not (folder / segment).exists():
                return  # Created below with the given spelling.
            names = os.listdir(folder)  # The real letter case, unlike exists().
            if segment not in names:
                same = [name for name in names if name.casefold() == segment.casefold()]
                existing = " as {!r}".format("/".join(segments[:depth] + same[:1])) if same else ""
                raise ValueError("Topic folder {!r} already exists{} in another letter case; use "
                                 "the existing spelling".format(
                                     "/".join(segments[:depth + 1]), existing))
            folder = folder / segment

    def existing_topic(self, topic: str) -> str:
        """``topic`` spelled like the topic folders that already exist on disk.

        On a case-insensitive disk, "work" names an existing "Work" folder, so a
        record for it belongs there under that spelling. From the first segment
        without such a folder on, the given spelling is kept.
        """
        folder, segments = self.memory_dir, topic.split("/")
        spelled: List[str] = []
        for segment in segments:
            if not (folder / segment).exists():
                break
            names = sorted(os.listdir(folder))  # The real letter case, unlike exists().
            same = ([name for name in names if name == segment]
                    or [name for name in names if name.casefold() == segment.casefold()])
            if not same:
                break  # Not a folder name at all, such as "..".
            spelled.append(same[0])
            folder = folder / same[0]
        return "/".join(spelled + segments[len(spelled):])

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
        self._check_topic_spelling(fact.topic)
        missing = [parent for parent in path.parents
                   if parent.is_relative_to(self.memory_dir) and parent != self.memory_dir
                   and not parent.exists()]
        path.parent.mkdir(parents=True, exist_ok=True)
        for created in sorted(missing, key=lambda folder: len(folder.parts)):
            _flush_directory(created.parent)  # Persist each new folder's entry too.
        if path.exists() or path.is_symlink():
            raise MemoryConflictError("Refusing to overwrite an existing Markdown record")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent),
                                             prefix=".memoryos-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(self._render(fact))
                handle.flush()
                _flush(handle.fileno())
            os.replace(str(temporary), str(path))
            temporary = None
            self._undo.append(lambda: path.unlink(missing_ok=True))
            _flush_directory(path.parent)
            self._index_record(fact, path)
            status = path.stat()
            self.connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                (path.relative_to(self.root).as_posix(), status.st_size, status.st_mtime_ns,
                 hashlib.sha256(path.read_bytes()).hexdigest(), time.time_ns()))
        except BaseException:
            # The transaction's undo list removes the new record if it was
            # already renamed into place; older files are never touched.
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return fact

    def _recorded_time(self, value: TimeInput) -> str:
        if value is not None:
            return normalize_time(value)
        now = normalize_time()
        # Break an exact clock tie without moving new observations into the
        # future merely because a caller explicitly imported a future record.
        # Writes within one clock tick stay strictly ordered. Records recorded
        # further ahead (imports with future timestamps) are not considered, so
        # new writes stay visible to queries known "now".
        latest = self.connection.execute(
            "SELECT max(recorded_at) FROM facts WHERE recorded_at <= ?",
            (_shift(now, _TIE_WINDOW),)).fetchone()[0]
        if latest is not None and latest >= now:
            return _after(latest)
        return now

    def add(self, entity: str, attribute: str, value: str, valid_from: TimeInput = None,
            valid_to: TimeInput = None, source: str = "manual", evidence: str = "",
            confidence: float = 1.0, tags: Optional[Sequence[str]] = None,
            links: Optional[Sequence[str]] = None, topic: str = "general",
            recorded_at: TimeInput = None) -> Fact:
        with self._transaction():
            self._refresh_locked()
            fact = Fact(id=uuid.uuid4().hex, entity=entity, attribute=attribute, value=value,
                        valid_from=normalize_time(valid_from),
                        valid_to=None if valid_to is None else normalize_time(valid_to),
                        recorded_at=self._recorded_time(recorded_at), source=source,
                        evidence=evidence, confidence=confidence,
                        tags=() if tags is None else tags,  # type: ignore[arg-type]
                        links=() if links is None else links,  # type: ignore[arg-type]
                        topic=topic)
            return self._write(fact)

    def _revision(self, fact_id: str, value: Optional[str], source: str, evidence: str,
                  recorded_at: TimeInput, deleted: bool, confidence: Optional[float] = None,
                  topic: Optional[str] = None) -> Fact:
        with self._transaction():
            self._refresh_locked()
            old = self._indexed(fact_id)
            if old is None:
                raise KeyError(fact_id)
            if self.connection.execute("SELECT 1 FROM facts WHERE supersedes = ?",
                                       (fact_id,)).fetchone():
                raise MemoryConflictError("This record has already been superseded; {} the latest "
                                          "revision instead".format("retract" if deleted else "correct"))
            if deleted and old.deleted:
                raise MemoryConflictError("This record has already been retracted")
            if recorded_at is None:
                # A revision cannot predate its target, even when the target was
                # imported with a timestamp ahead of this machine's clock.
                recorded = max(self._recorded_time(None), _after(old.recorded_at))
            else:
                recorded = normalize_time(recorded_at)
            fact = replace(old, id=uuid.uuid4().hex, value="" if deleted or value is None else value,
                           source=source, evidence=evidence, deleted=deleted,
                           recorded_at=recorded, supersedes=old.id,
                           confidence=old.confidence if confidence is None else confidence,
                           topic=self._folder_topic(old) if topic is None else topic)
            return self._write(fact)

    def _folder_topic(self, fact: Fact) -> str:
        """The record's topic spelled like the folder that holds it.

        Earlier releases filed a topic such as "work" into an existing "Work"
        folder on case-insensitive disks; its revisions follow the folder instead
        of being refused. A record moved to another folder keeps its own topic,
        spelled like that topic's existing folder, as a new record's would be.
        The index path is current: the caller refreshed first.
        """
        row = self.connection.execute("SELECT path FROM facts WHERE id = ?", (fact.id,)).fetchone()
        folder = str(row["path"]).rpartition("/")[0].partition("/")[2]
        same = folder.isascii() and folder.lower() == fact.topic.lower()  # Topics are ASCII.
        return folder if same else self.existing_topic(fact.topic)

    def correct(self, fact_id: str, value: str, source: str = "manual", evidence: str = "",
                recorded_at: TimeInput = None, confidence: Optional[float] = None,
                topic: Optional[str] = None) -> Fact:
        """Revise an assertion at its original effective time; keep the old record.

        ``confidence`` and ``topic`` default to the corrected record's values; the
        topic is spelled like the folder that holds the record, or, for a record
        moved out of its topic folder, like that topic's existing folder.
        """
        return self._revision(fact_id, value, source, evidence, recorded_at, False,
                              confidence, topic)

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

    def audit_trail(self, entity: str, attribute: str) -> List[Fact]:
        """Every record for one entity/attribute, including superseded revisions,
        in the order they were recorded. A correction always follows the record
        it supersedes, even when both carry the same timestamp."""
        with self._lock:
            rows = self.connection.execute(
                "SELECT id, payload FROM facts WHERE entity = ? AND attribute = ? "
                "ORDER BY recorded_at, id", (entity, attribute)).fetchall()
            facts = [self._fact(row) for row in rows]
        placed: set = set()
        ordered: List[Fact] = []
        pending = facts
        present = {fact.id for fact in facts}
        while pending:
            ready = [fact for fact in pending
                     if fact.supersedes is None or fact.supersedes in placed
                     or fact.supersedes not in present]
            if not ready:  # Not reachable for a validated store; keep the order.
                ready = pending
            ordered.extend(ready)
            placed.update(fact.id for fact in ready)
            pending = [fact for fact in pending if fact.id not in placed]
        position = {fact.id: index for index, fact in enumerate(ordered)}
        return sorted(ordered, key=lambda fact: (fact.recorded_at, position[fact.id]))

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
