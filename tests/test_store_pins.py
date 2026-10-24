"""Store behavior that mutation testing showed no other test checked.

Covers time shifting on Python 3.10, opening and lock retries, repair of a
damaged index, file modes, the folder scan, the racy-timestamp rule, records
that are not UTF-8, cross-record validation, verification marking, the
record writer's refusals and cleanup, lookups, the audit trail, record titles,
folder durability, what the index file holds, and messages that name files.
"""

import errno
import io
import json
import os
import re
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import types
from contextlib import closing, contextmanager
from pathlib import Path

import pytest
from helpers import hand_written_record, MARKER, Python310Datetime

from temporal_memory_os import store as store_module
from temporal_memory_os.models import Fact
from temporal_memory_os.store import MemoryConflictError, MemoryStore, StoreIntegrityError


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "store") as memory:
        yield memory


# ==========================================================================
# Time shifting, opening, lock retries, repair and file modes
# ==========================================================================


STAMP = "20260101T000000"


def _state(root) -> Path:
    return Path(root) / ".memoryos"


def _index(root) -> Path:
    return _state(root) / "index.sqlite3"


def _set_aside_names(root) -> list:
    """Names of index files that a repair moved out of the way."""
    return sorted(path.name for path in _state(root).iterdir() if ".corrupt" in path.name)


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


class _Retried(Exception):
    """Raised by a patched sleep: the code under test waited in order to retry."""


def _forbid_retry_waits(monkeypatch) -> None:
    def refuse(seconds):
        raise _Retried(seconds)

    monkeypatch.setattr(store_module.time, "sleep", refuse)


def _record_retry_waits(monkeypatch) -> list:
    """Replace the pause between first-open attempts; return the requested pauses."""
    waits: list = []
    monkeypatch.setattr(store_module.time, "sleep", waits.append)
    return waits


def _fail_schema_preparation(monkeypatch, message: str, times=None) -> list:
    """Make the first ``times`` schema preparations (all, if None) raise an
    OperationalError with ``message``; return the list of attempts."""
    real, attempts = MemoryStore._initialize, []

    def initialize(self, connection):
        attempts.append(1)
        if times is None or len(attempts) <= times:
            raise sqlite3.OperationalError(message)
        return real(self, connection)

    monkeypatch.setattr(MemoryStore, "_initialize", initialize)
    return attempts


@contextmanager
def _deadline(seconds: float):
    """Interrupt the block with TimeoutError if it runs longer than ``seconds``."""
    def expire(_signum, _frame):
        raise TimeoutError("did not finish within {} s".format(seconds))

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class _DamagedConnection:
    """Stands in for SQLite reporting damage before it touches any file."""

    def execute(self, *_arguments):
        raise sqlite3.DatabaseError("file is not a database")

    def close(self):
        pass


def _damage_pages(root) -> None:
    """Keep the first page (header and schema) and overwrite the rest."""
    index = _index(root)
    raw = index.read_bytes()
    page_size = int.from_bytes(raw[16:18], "big")
    assert 0 < page_size < len(raw)
    index.write_bytes(raw[:page_size] + b"\xab" * (len(raw) - page_size))


# Recorded times ------------------------------------------------------------


def test_write_times_do_not_need_fromisoformat_to_accept_a_z_suffix(tmp_path, monkeypatch):
    """Tie-breaking and revision times work with Python 3.10's stricter fromisoformat."""
    monkeypatch.setattr(store_module, "datetime", Python310Datetime)
    with MemoryStore(tmp_path) as store:
        first = store.add("person", "drink", "tea", valid_from="2025-01-01")
        second = store.add("person", "city", "Riverton", valid_from="2025-01-01")
        future = store.add("person", "employer", "Acme", valid_from="2025-01-01",
                           recorded_at="2100-01-01")
        corrected = store.correct(future.id, "Initech")
    assert first.recorded_at < second.recorded_at
    assert corrected.recorded_at == "2100-01-01T00:00:00.000001Z"


# Opening and lock retries --------------------------------------------------


def test_a_schema_error_that_is_not_a_lock_is_reported_without_retrying(tmp_path, monkeypatch):
    """An index with an incompatible table layout fails at once, without lock retries."""
    _state(tmp_path).mkdir()
    with closing(sqlite3.connect(str(_index(tmp_path)))) as connection:
        connection.execute("CREATE TABLE facts (note TEXT)")
        connection.commit()
    _forbid_retry_waits(monkeypatch)
    for repair in (False, True):
        with pytest.raises(sqlite3.OperationalError):
            MemoryStore(tmp_path, repair_index=repair)
    assert _set_aside_names(tmp_path) == []


def test_first_open_retries_when_the_database_reports_itself_busy(tmp_path, monkeypatch):
    """A 'busy' error while preparing the schema is retried like a lock."""
    attempts = _fail_schema_preparation(monkeypatch, "database is busy", times=1)
    waits = _record_retry_waits(monkeypatch)
    with MemoryStore(tmp_path) as store:
        assert store.add("person", "drink", "tea", valid_from="2025-01-01")
    assert len(attempts) == 2 and len(waits) == 1


def test_first_open_tries_again_within_a_fraction_of_a_second_while_the_schema_is_locked(
        tmp_path, monkeypatch):
    """Concurrent first opens retry briefly: each pause before another attempt is short."""
    attempts = _fail_schema_preparation(monkeypatch, "database is locked", times=2)
    waits = _record_retry_waits(monkeypatch)
    MemoryStore(tmp_path).close()
    assert len(attempts) == 3 and len(waits) == 2
    assert all(0 < wait < 0.5 for wait in waits)


def test_a_lock_is_reported_once_the_deadline_is_reached_even_if_the_clock_stands_still(
        tmp_path, monkeypatch):
    """The retry window is half open: a lock seen exactly at the deadline is not retried."""
    _fail_schema_preparation(monkeypatch, "database is locked")
    monkeypatch.setattr(store_module, "_LOCK_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(store_module.time, "monotonic", lambda: 1000.0)
    _forbid_retry_waits(monkeypatch)
    with pytest.raises(sqlite3.OperationalError):
        MemoryStore(tmp_path)


def test_an_open_store_waits_thirty_seconds_for_another_writer(tmp_path):
    """Writes wait up to 30 s for a lock such as a checkpoint's, on a new or reopened store."""
    with MemoryStore(tmp_path) as store:  # New index: nothing to re-verify on opening.
        assert store.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000
        store.add("person", "drink", "tea", valid_from="2025-01-01")
    with MemoryStore(tmp_path) as store:  # A just-written record is re-verified on opening.
        assert store.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000


# Repair --------------------------------------------------------------------


def test_repair_stops_when_the_rebuilt_index_is_damaged_again(tmp_path, monkeypatch):
    """After one set-aside, a damaged file found on reopening stops the repair."""
    with MemoryStore(tmp_path) as store:
        store.add("person", "drink", "tea", valid_from="2025-01-01")
    _damage_pages(tmp_path)
    planted = b"recreated by another program " * 100
    real, calls = MemoryStore._set_aside, []

    def set_aside_then_a_damaged_file_reappears(self):
        real(self)
        calls.append(1)
        if len(calls) == 1:
            self.db_path.write_bytes(planted)

    monkeypatch.setattr(MemoryStore, "_set_aside", set_aside_then_a_damaged_file_reappears)
    with pytest.raises(StoreIntegrityError):
        MemoryStore(tmp_path, repair_index=True)
    assert len(calls) == 1
    assert _index(tmp_path).read_bytes() == planted
    # One repair: whatever was set aside shares a single label.
    assert len({name.rsplit(".corrupt-", 1)[1] for name in _set_aside_names(tmp_path)}) == 1


def test_repeated_repairs_within_one_second_keep_every_damaged_copy(tmp_path, monkeypatch):
    """Each repair in the same second gets the next free numbered label."""
    MemoryStore(tmp_path).close()
    monkeypatch.setattr(store_module.time, "strftime", lambda _format: STAMP)
    contents = [b"first damage " * 100, b"second damage " * 100, b"third damage " * 100]
    with _deadline(2):
        for content in contents:
            _index(tmp_path).write_bytes(content)
            MemoryStore(tmp_path, repair_index=True).close()
    labels = [STAMP, STAMP + "-1", STAMP + "-2"]
    assert _set_aside_names(tmp_path) == ["index.sqlite3.corrupt-" + label for label in labels]
    for label, content in zip(labels, contents):
        assert (_state(tmp_path) / ("index.sqlite3.corrupt-" + label)).read_bytes() == content


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_repair_skips_a_label_already_used_by_an_earlier_log_copy(tmp_path, monkeypatch, suffix):
    """A label taken by any earlier set-aside file is not reused, so it is never overwritten."""
    MemoryStore(tmp_path).close()
    monkeypatch.setattr(store_module.time, "strftime", lambda _format: STAMP)
    state = _state(tmp_path)
    # An earlier repair's copy whose main file has since been deleted.
    earlier = state / "index.sqlite3{}.corrupt-{}".format(suffix, STAMP)
    earlier.write_bytes(b"earlier copy")
    _index(tmp_path).write_bytes(b"damaged index")
    (state / ("index.sqlite3" + suffix)).write_bytes(b"damaged log")
    real, calls = MemoryStore._connect, []

    def connect(self):
        calls.append(1)
        return _DamagedConnection() if len(calls) == 1 else real(self)

    monkeypatch.setattr(MemoryStore, "_connect", connect)
    MemoryStore(tmp_path, repair_index=True).close()
    assert earlier.read_bytes() == b"earlier copy"
    label = STAMP + "-1"
    assert (state / "index.sqlite3.corrupt-{}".format(label)).read_bytes() == b"damaged index"
    assert (state / "index.sqlite3{}.corrupt-{}".format(suffix, label)).read_bytes() == (
        b"damaged log")


# Index file modes ----------------------------------------------------------


def test_index_file_is_created_private_under_a_strict_umask(tmp_path):
    """With umask 077 the new index is 0600, not owner-executable."""
    previous = os.umask(0o077)
    try:
        MemoryStore(tmp_path).close()
    finally:
        os.umask(previous)
    assert _mode(_index(tmp_path)) == 0o600


def test_index_file_is_private_from_creation_before_any_tightening(tmp_path, monkeypatch):
    """The index is 0600 from creation, not only after the pass that tightens own files."""
    other_user = os.getuid() + 1
    monkeypatch.setattr(store_module.os, "getuid", lambda: other_user)
    previous = os.umask(0o022)
    try:
        MemoryStore(tmp_path).close()
    finally:
        os.umask(previous)
    assert _mode(_index(tmp_path)) == 0o600


def test_live_log_and_shared_memory_files_left_world_readable_are_made_private(tmp_path):
    """Opening tightens a non-empty log and shared-memory file held open by another store."""
    state = _state(tmp_path)
    with MemoryStore(tmp_path) as first:
        first.add("person", "drink", "tea", valid_from="2025-01-01")
        for suffix in ("-wal", "-shm"):  # SQLite itself only fixes the mode of empty ones.
            assert (state / ("index.sqlite3" + suffix)).stat().st_size > 0
        for suffix in ("", "-wal", "-shm"):  # As left by an older release.
            (state / ("index.sqlite3" + suffix)).chmod(0o644)
        with MemoryStore(tmp_path) as second:
            assert second.all_facts() == first.all_facts()
            modes = {path.name: _mode(path) for path in state.iterdir()}
    assert modes == {"index.sqlite3": 0o600, "index.sqlite3-wal": 0o600,
                     "index.sqlite3-shm": 0o600}


# ==========================================================================
# Scanning, checking and loading the Markdown folder
# ==========================================================================


_REAL_SCANDIR = os.scandir


class _NameOrderedScandir:
    """``os.scandir`` listing entries sorted by name, as some file systems do."""

    def __init__(self, path):
        self._iterator = _REAL_SCANDIR(path)
        self._entries = sorted(self._iterator, key=lambda entry: entry.name)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __iter__(self):
        return iter(self._entries)

    def close(self):
        self._iterator.close()


# _scan: which entries the folder walk reports


def test_a_dot_named_entry_does_not_hide_the_entries_listed_after_it(store, monkeypatch):
    """Dot-named entries are skipped one at a time, whatever order the folder lists them in."""
    tea = store.add("person", "drink", "tea", valid_from="2025-01-01")
    city = store.add("person", "city", "Riverton", valid_from="2025-01-01", topic="places")
    (store.memory_dir / ".obsidian").mkdir()
    (store.path_for(tea.id).parent / ".#draft.md").symlink_to("someone@host.12345")
    monkeypatch.setattr(os, "scandir", _NameOrderedScandir)
    assert store.refresh() is False
    assert store.reindex() == 2
    assert {fact.id for fact in store.all_facts()} == {tea.id, city.id}


def test_a_symlink_to_a_non_markdown_file_is_ignored(store, tmp_path):
    """Symbolic links that are neither Markdown files nor folders are ignored."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    picture = tmp_path / "photo.jpg"
    picture.write_bytes(b"\xff\xd8\xff")
    folder = store.path_for(fact.id).parent
    (folder / "photo.jpg").symlink_to(picture)
    (folder / "old-link").symlink_to(tmp_path / "missing-target")
    assert store.refresh() is False
    assert store.reindex() == 1
    with MemoryStore(store.root) as reopened:
        assert reopened.get(fact.id).value == "tea"


def test_a_record_replaced_by_a_symlink_is_reported_as_a_store_problem(store, tmp_path):
    """A known record path that turns into a symbolic link raises StoreIntegrityError."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    path = store.path_for(fact.id)
    elsewhere = tmp_path / "elsewhere.md"
    elsewhere.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(elsewhere)
    with pytest.raises(StoreIntegrityError):
        store.refresh()
    with pytest.raises(StoreIntegrityError):
        MemoryStore(store.root)
    assert store.get(fact.id).value == "tea"


def test_a_symlink_replacing_a_one_byte_note_dated_just_before_the_epoch_is_rejected(
        store, tmp_path):
    """A symbolic link never matches the size and time recorded for a real file."""
    note = store.memory_dir / "dot.md"
    note.write_bytes(b"\n")
    try:
        os.utime(note, ns=(-1, -1))
    except OSError:
        pytest.skip("this file system cannot store a time before the epoch")
    if (note.stat().st_size, note.stat().st_mtime_ns) != (1, -1):
        pytest.skip("this file system cannot store a time just before the epoch")
    assert store.refresh() is True
    target = tmp_path / "target.md"
    target.write_bytes(b"\n")
    note.unlink()
    note.symlink_to(target)
    with pytest.raises(StoreIntegrityError):
        store.refresh()


def test_a_linked_folder_replaced_by_a_file_after_the_scan_is_rebuilt_next_time(
        store, tmp_path, monkeypatch):
    """A folder link that turns into a plain file mid-rebuild is rebuilt again next refresh."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = store.memory_dir / "attachments"
    link.symlink_to(outside, target_is_directory=True)
    real_scan = MemoryStore._scan

    def scan_then_replace(self):
        found = real_scan(self)
        if link.is_symlink():
            link.unlink()
            link.write_text("plain file\n", encoding="utf-8")
        return found

    with monkeypatch.context() as patch:
        patch.setattr(MemoryStore, "_scan", scan_then_replace)
        assert store.reindex() == 1
    assert store.refresh() is True
    assert store.refresh() is False
    assert store.get(fact.id).value == "tea"


# _check: the racy-timestamp rule


def test_an_edit_at_the_edge_of_the_racy_window_is_found_by_hash(store, monkeypatch):
    """A same-size edit with its mtime exactly one racy window old is still found by hash."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    path = store.path_for(fact.id)
    stamp = 1_700_000_000_000_000_000
    os.utime(path, ns=(stamp, stamp))
    monkeypatch.setattr(time, "time_ns", lambda: stamp + store_module._RACY_WINDOW_NS)
    assert store.refresh() is True
    path.write_bytes(path.read_bytes().replace(b'"tea"', b'"tee"'))
    os.utime(path, ns=(stamp, stamp))
    assert store.refresh() is True
    assert store.get(fact.id).value == "tee"


# _parse: records that are not valid UTF-8


@pytest.mark.parametrize("newline", [b"\r", b"\r\n"], ids=["cr-only", "crlf"])
def test_a_new_record_with_cr_line_endings_that_is_not_utf8_is_rejected(store, newline):
    """A CR or CRLF record that is not valid UTF-8 is reported, not skipped as a note."""
    fact = store.add("person", "city", "Riverton", valid_from="2025-01-01")
    record = store.memory_dir / "general" / "hand-written.md"
    raw = hand_written_record(value="t\u00e9a", ensure_ascii=False).encode("latin-1")
    record.write_bytes(raw.replace(b"\n", newline))
    with pytest.raises(StoreIntegrityError) as caught:
        store.refresh()
    assert "memory/general/hand-written.md" in str(caught.value)
    assert [f.id for f in store.all_facts()] == [fact.id]


# _load_markdown: the scan that triggered a rebuild is the one recorded


def test_a_file_edited_after_the_staleness_scan_is_rebuilt_on_the_next_refresh(
        store, monkeypatch):
    """A file edited after the staleness scan keeps a stale signature and is rebuilt again."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    path = store.path_for(fact.id)
    path.write_bytes(path.read_bytes().replace(b'"tea"', b'"tee"'))
    real_is_current = MemoryStore._is_current

    def check_then_edit(self, scan):
        current = real_is_current(self, scan)
        if not current and b'"tee"' in path.read_bytes():
            path.write_bytes(path.read_bytes().replace(b'"tee"', b'"green tea"'))
        return current

    with monkeypatch.context() as patch:
        patch.setattr(MemoryStore, "_is_current", check_then_edit)
        assert store.refresh() is True
    assert store.get(fact.id).value == "green tea"
    assert store.refresh() is True
    assert store.refresh() is False


# ==========================================================================
# Cross-record validation, index rows, refresh results and verification marking
# ==========================================================================


def _record(**changes):
    data = dict(id="a" * 32, entity="person", attribute="drink", value="tea",
                valid_from="2025-01-01", recorded_at="2025-01-01")
    data.update(changes)
    return Fact.from_dict(data)


def _busy_timeout(memory):
    return memory.connection.execute("PRAGMA busy_timeout").fetchone()[0]


def _treat_every_file_as_recent(monkeypatch):
    monkeypatch.setattr(store_module, "_RACY_WINDOW_NS", 10 ** 18)


# _validate_collection: rules that span records
#
# The public paths catch a repeated ID before this validator sees it (a rebuild
# checks IDs file by file, a write looks the ID up in the index), so its own
# duplicate-ID rule is exercised directly.


def test_validator_rejects_a_repeated_id_as_a_value_error():
    """The cross-record validator reports two records sharing an ID as a ValueError."""
    with pytest.raises(ValueError):
        MemoryStore._validate_collection([_record(), _record(value="coffee")])


def test_validator_names_the_file_of_a_repeated_id():
    """Given file paths, the duplicate-ID error names the file holding the repeated ID."""
    record = _record()
    with pytest.raises(ValueError) as caught:
        MemoryStore._validate_collection(
            [record, _record(value="coffee")], {record.id: "memory/general/tea-record.md"})
    assert "memory/general/tea-record.md" in str(caught.value)


def test_duplicate_effective_start_error_names_entity_and_attribute(store):
    """A second record at the same effective start names the entity and attribute in conflict."""
    store.add("Ada Lovelace", "employer", "Analytical Engines Ltd", valid_from="2025-01-01")
    with pytest.raises(MemoryConflictError) as caught:
        store.add("Ada Lovelace", "employer", "Difference Works", valid_from="2025-01-01")
    text = str(caught.value)
    assert "Ada Lovelace" in text
    assert "employer" in text


# _index_record: what a record's index row holds


def test_index_row_holds_tags_and_links_as_json_arrays(store):
    """The index row keeps a record's tags and links as JSON arrays, written and rebuilt alike."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01",
                     tags=["work", "café"], links=["Analytical Engine"])

    def columns():
        row = store.connection.execute("SELECT tags, links FROM facts WHERE id = ?",
                                       (fact.id,)).fetchone()
        return json.loads(row["tags"]), json.loads(row["links"])

    assert columns() == (["work", "café"], ["Analytical Engine"])
    assert store.reindex() == 1
    assert columns() == (["work", "café"], ["Analytical Engine"])


# _refresh_locked: what refresh reports


def test_refresh_reports_no_rebuild_when_another_store_rebuilt_first(store, monkeypatch):
    """refresh() returns False when another store rebuilt the stale index before it got the lock."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    path = store.path_for(fact.id)
    path.write_text(path.read_text(encoding="utf-8").replace('"tea"', '"green tea"'),
                    encoding="utf-8")
    original_check = store._check
    first_results = []

    def check_then_let_another_store_rebuild(scan):
        result = original_check(scan)
        if not first_results:
            first_results.append(result)
            # Another store rebuilds after this read check, before the write lock.
            MemoryStore(store.root).close()
        return result

    monkeypatch.setattr(store, "_check", check_then_let_another_store_rebuild)
    assert store.refresh() is False
    assert first_results == [None]  # The read check did find the index stale.
    assert store.get(fact.id).value == "green tea"


# _mark_verified: recording that recently written files still match


def test_verifying_recent_files_restores_the_connection_lock_wait(store, monkeypatch):
    """Marking recent files verified restores the store's lock wait, whether or not the lock is free."""
    _treat_every_file_as_recent(monkeypatch)
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    before = _busy_timeout(store)
    assert before > 0
    assert store.refresh() is False  # Lock free: the files are marked verified.
    assert _busy_timeout(store) == before
    holder = sqlite3.connect(str(store.db_path), isolation_level=None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        assert store.refresh() is False  # Lock held: marking is skipped.
        assert _busy_timeout(store) == before
    finally:
        if holder.in_transaction:
            holder.execute("ROLLBACK")
        holder.close()


def test_failure_while_marking_files_verified_is_reraised_and_rolled_back(store, monkeypatch):
    """An unexpected error while marking files verified propagates and leaves no open transaction."""
    _treat_every_file_as_recent(monkeypatch)
    tea = store.add("person", "drink", "tea", valid_from="2025-01-01")

    def unreadable(scan):
        raise PermissionError("a recently written record could not be reread")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_is_current", unreadable)
        with pytest.raises(PermissionError):
            store.refresh()
    assert not store.connection.in_transaction
    city = store.add("person", "city", "Riverton", valid_from="2025-01-01")
    assert {fact.id for fact in store.all_facts()} == {tea.id, city.id}


# ==========================================================================
# Record writes, lookups and the audit trail
# ==========================================================================


FIXED_ID = "{:032x}".format(0xC0FFEE)


def _fixed_ids(patch, *hexes):
    """Make the store hand out the given record IDs, in order."""
    ids = iter(hexes)
    patch.setattr(store_module.uuid, "uuid4", lambda: types.SimpleNamespace(hex=next(ids)))


def _temporary_files(folder: Path) -> list:
    """Names of temporary files left in ``folder``, sorted."""
    return sorted(path.name for path in folder.iterdir() if path.name.endswith(".tmp"))


def _raw_index(store: MemoryStore) -> sqlite3.Connection:
    """A plain SQLite connection to the store's index, outside MemoryStore."""
    return sqlite3.connect(str(store.db_path), isolation_level=None, timeout=30)


# _write: symbolic links that the folder scan cannot see


def test_dangling_topic_link_leading_outside_memory_is_refused(store, tmp_path):
    """A broken topic link that would lead outside the memory folder is refused as a link."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.memory_dir / "escape").symlink_to(outside / "missing", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        store.add("person", "drink", "tea", valid_from="2025-01-01", topic="escape")
    assert list(outside.iterdir()) == []
    assert store.all_facts() == []


def test_dangling_topic_link_inside_memory_is_refused_as_a_link(store):
    """A broken topic link pointing inside memory raises ValueError naming the link problem."""
    (store.memory_dir / "alias").symlink_to(store.memory_dir / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        store.add("person", "drink", "tea", valid_from="2025-01-01", topic="alias")
    assert not (store.memory_dir / "real").exists()
    assert list(store.memory_dir.rglob("*.md")) == []
    assert store.all_facts() == []


def test_nested_topic_through_a_dangling_link_is_refused_as_a_link(store):
    """A topic whose middle folder is a broken link inside memory raises ValueError."""
    (store.memory_dir / "profile").mkdir()
    (store.memory_dir / "profile" / "alias").symlink_to(store.memory_dir / "real",
                                                        target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        store.add("person", "drink", "tea", valid_from="2025-01-01", topic="profile/alias")
    assert not (store.memory_dir / "real").exists()
    assert store.all_facts() == []


# _write: existing files are never replaced


_NOTE_TEXT = "# My own note\n\nNot a memory record.\n"


def _note_where_next_record_goes(store: MemoryStore) -> Path:
    """An ordinary, unmarked note named like the record that FIXED_ID would produce."""
    folder = store.memory_dir / "general"
    folder.mkdir()
    note = folder / (FIXED_ID + ".md")
    note.write_text(_NOTE_TEXT, encoding="utf-8")
    return note


def test_new_record_never_overwrites_a_note_with_the_same_name(store, monkeypatch):
    """A record whose file name is taken by an ordinary note is refused; the note is kept."""
    note = _note_where_next_record_goes(store)
    with monkeypatch.context() as patch:
        _fixed_ids(patch, FIXED_ID)
        with pytest.raises(MemoryConflictError):
            store.add("person", "drink", "tea", valid_from="2025-01-01")
    assert note.read_text(encoding="utf-8") == _NOTE_TEXT
    assert store.all_facts() == []
    assert _temporary_files(note.parent) == []


def test_refusal_to_overwrite_explains_the_reason(store, monkeypatch):
    """The conflict raised for a taken file name says that it refuses to overwrite."""
    _note_where_next_record_goes(store)
    with monkeypatch.context() as patch:
        _fixed_ids(patch, FIXED_ID)
        with pytest.raises(MemoryConflictError, match="overwrite"):
            store.add("person", "drink", "tea", valid_from="2025-01-01")


# _write: failures and their cleanup


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="needs file permissions that apply to the current user")
def test_write_into_read_only_topic_folder_raises_permission_error(store):
    """A topic folder without write permission makes add() raise PermissionError, nothing else."""
    kept = store.add("person", "city", "Riverton", valid_from="2025-01-01", topic="locked")
    folder = store.memory_dir / "locked"
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(PermissionError):
            store.add("person", "drink", "tea", valid_from="2025-01-01", topic="locked")
    finally:
        folder.chmod(stat.S_IRWXU)
    assert sorted(path.name for path in folder.iterdir()) == [kept.id + ".md"]
    assert store.all_facts() == [kept]


def test_rollback_tolerates_a_new_record_that_already_vanished(store, monkeypatch):
    """If the new file is gone before rollback, the original failure still propagates."""
    kept = store.add("person", "city", "Riverton", valid_from="2025-01-01")

    def vanish_then_fail(_fact, path):
        path.unlink()
        raise RuntimeError("index unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_index_record", vanish_then_fail)
        with pytest.raises(RuntimeError, match="index unavailable"):
            store.add("person", "drink", "tea", valid_from="2025-01-01")
    assert [path.stem for path in store.memory_dir.rglob("*.md")] == [kept.id]
    assert store.all_facts() == [kept]
    assert store.refresh() is False


def test_cleanup_tolerates_a_temporary_file_that_already_vanished(store, monkeypatch):
    """If the temporary file is gone when a write fails, the original failure still propagates."""
    kept = store.add("person", "city", "Riverton", valid_from="2025-01-01")
    folder = store.memory_dir / "general"
    real_flush = store_module._flush

    def lose_temporary_then_fail(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            for leftover in folder.glob(".memoryos-*.tmp"):
                leftover.unlink()
            raise RuntimeError("device failure")
        real_flush(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "_flush", lose_temporary_then_fail)
        with pytest.raises(RuntimeError, match="device failure"):
            store.add("person", "drink", "tea", valid_from="2025-01-01")
    assert sorted(path.name for path in folder.iterdir()) == [kept.id + ".md"]
    assert store.all_facts() == [kept]


# Kept ASCII so that the child decodes it the same way under any locale.
_ASCII_LOCALE_WRITER = r"""
import locale, sys
from temporal_memory_os.store import MemoryStore
if "utf" in locale.getpreferredencoding(False).lower():
    sys.exit(3)
with MemoryStore(sys.argv[1]) as store:
    fact = store.add("Zo\u00eb", "city", "\u6771\u4eac \u2615", valid_from="2025-01-01")
    print(fact.id)
"""


def test_records_are_written_as_utf8_under_an_ascii_locale(tmp_path):
    """Records are UTF-8 files even when the process locale's encoding is ASCII."""
    root = tmp_path / "store"
    source = Path(store_module.__file__).resolve().parents[1]
    environment = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0",
                       PYTHONCOERCECLOCALE="0", PYTHONPATH=str(source))
    environment.pop("PYTHONIOENCODING", None)
    finished = subprocess.run([sys.executable, "-c", _ASCII_LOCALE_WRITER, str(root)],
                              env=environment, capture_output=True, timeout=60)
    if finished.returncode == 3:
        pytest.skip("this system has no non-UTF-8 C locale")
    assert finished.returncode == 0, finished.stderr.decode("utf-8", "replace")
    fact_id = finished.stdout.decode("ascii").strip()
    with MemoryStore(root) as reopened:
        fact = reopened.get(fact_id)
        raw = reopened.path_for(fact_id).read_bytes()
    assert (fact.entity, fact.value) == ("Zoë", "東京 ☕")
    assert "東京 ☕".encode("utf-8") in raw
    assert json.loads(raw.decode("utf-8").split("```json\n", 1)[1].split("\n```", 1)[0])[
        "value"] == "東京 ☕"


# get and path_for


def test_get_unknown_id_raises_key_error_carrying_that_id(store):
    """get() on an unknown ID raises KeyError whose argument is the missing ID."""
    missing = "a" * 32
    with pytest.raises(KeyError) as caught:
        store.get(missing)
    assert caught.value.args == (missing,)


def test_path_for_unknown_id_raises_key_error_carrying_that_id(store):
    """path_for() on an unknown ID raises KeyError whose argument is the missing ID."""
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    missing = "b" * 32
    with pytest.raises(KeyError) as caught:
        store.path_for(missing)
    assert caught.value.args == (missing,)


# audit_trail


def test_audit_trail_lists_every_record_when_the_index_holds_a_correction_cycle(store):
    """A correction cycle written straight into the index still yields every record, in order."""
    first = store.add("person", "drink", "tea", valid_from="2025-01-01", recorded_at="2025-01-01")
    second = store.correct(first.id, "coffee", recorded_at="2025-01-02")
    looped = dict(first.to_dict(), supersedes=second.id)
    with closing(_raw_index(store)) as raw:
        raw.execute("UPDATE facts SET supersedes = ?, payload = ? WHERE id = ?",
                    (second.id, json.dumps(looped, ensure_ascii=False), first.id))
    trail = store.audit_trail("person", "drink")
    assert [fact.id for fact in trail] == [first.id, second.id]
    assert trail[0].supersedes == second.id


# ==========================================================================
# Record titles, folder durability and index file modes
# ==========================================================================


def _record_path(store, fact):
    return store.memory_dir / fact.topic / (fact.id + ".md")


def _first_line(store, fact):
    raw = _record_path(store, fact).read_bytes()
    return raw.split(b"\n", 1)[0].decode("utf-8")


def test_line_feed_in_a_name_becomes_one_space_in_the_record_title(store):
    """A line feed inside an entity name is written as a single space in the heading."""
    fact = store.add("Alex\nSmith", "home city", "Riverton", valid_from="2025-01-01")
    assert _first_line(store, fact) == "# Alex Smith · home city"


def test_carriage_return_in_a_name_becomes_one_space_in_the_record_title(store):
    """A carriage return inside an attribute is written as a single space in the heading."""
    fact = store.add("Alex", "home\rcity", "Riverton", valid_from="2025-01-01")
    raw = _record_path(store, fact).read_bytes()
    assert raw.startswith("# Alex · home city\n\n".encode("utf-8"))
    assert b"\r" not in raw.split(b"\n", 1)[0]


def test_record_title_keeps_names_on_one_line_and_reads_back(tmp_path):
    """A name with CRLF stays on the heading line and the record reopens unchanged."""
    with MemoryStore(tmp_path) as store:
        fact = store.add("Alex\r\nSmith", "home city", "Riverton", valid_from="2025-01-01")
        assert _first_line(store, fact) == "# Alex  Smith · home city"
    with MemoryStore(tmp_path) as reopened:
        assert reopened.get(fact.id).entity == "Alex\r\nSmith"


def test_writing_a_record_flushes_the_folder_that_holds_it(store, monkeypatch, tmp_path):
    """The rename of a new record is made durable by fsyncing its own folder."""
    flushed = []
    real_flush = store_module._flush

    def recording_flush(descriptor):
        status = os.fstat(descriptor)
        flushed.append((status.st_dev, status.st_ino))
        real_flush(descriptor)

    monkeypatch.setattr(store_module, "_flush", recording_flush)
    monkeypatch.chdir(tmp_path)
    fact = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01", topic="profile")
    folder = os.stat(store.memory_dir / "profile")
    parent = os.stat(store.memory_dir)
    assert (folder.st_dev, folder.st_ino) in flushed
    assert (parent.st_dev, parent.st_ino) in flushed  # The new folder's own entry.
    assert _record_path(store, fact).exists()


def test_log_and_shared_memory_files_are_made_private_on_a_case_sensitive_disk(tmp_path, monkeypatch):
    """Opening tightens a world-readable log and shared-memory file found by their exact names."""
    state = tmp_path / ".memoryos"
    names = ("index.sqlite3", "index.sqlite3-wal", "index.sqlite3-shm")
    real_stat = Path.stat

    def case_sensitive_stat(path, *args, **kwargs):
        # Names in the index folder must match exactly, as on Linux file systems.
        if path.parent.name == ".memoryos" and path.name not in os.listdir(path.parent):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
        return real_stat(path, *args, **kwargs)

    with MemoryStore(tmp_path) as first:
        first.add("person", "drink", "tea", valid_from="2025-01-01")
        for name in names:  # As left by an older release, while another store holds them open.
            (state / name).chmod(0o644)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "stat", case_sensitive_stat)
            with MemoryStore(tmp_path) as second:
                assert second.all_facts() == first.all_facts()
        modes = {name: stat.S_IMODE(os.stat(state / name).st_mode) for name in names}
    assert modes == dict.fromkeys(names, 0o600)


# ==========================================================================
# Staleness checks: empty notes and linked folders
# ==========================================================================


@pytest.mark.parametrize("old", [False, True], ids=["recent", "old"])
def test_an_unchanged_empty_note_does_not_force_a_rebuild(store, old):
    """A zero-byte note that has not changed leaves the index current."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    note = store.path_for(fact.id).parent / "empty.md"
    note.write_bytes(b"")
    if old:  # Well outside the racy window, so it is trusted without hashing.
        os.utime(note, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    assert store.refresh() is True  # The new note is picked up once.
    assert store.refresh() is False
    assert store.refresh() is False
    with MemoryStore(store.root) as reopened:
        assert reopened.refresh() is False
        assert reopened.get(fact.id).value == "tea"
    assert note.read_bytes() == b""


def test_a_linked_folder_whose_placeholder_signature_was_stored_is_still_rejected(
        store, tmp_path, monkeypatch):
    """A folder link is rejected even after a race stored its placeholder signature."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = store.memory_dir / "attachments"
    link.symlink_to(outside, target_is_directory=True)
    real_scan = MemoryStore._scan

    def scan_then_replace(self):
        found = real_scan(self)
        if link.is_symlink():  # The link becomes a plain file between stat and read.
            link.unlink()
            link.write_text("plain file\n", encoding="utf-8")
        return found

    with monkeypatch.context() as patch:
        patch.setattr(MemoryStore, "_scan", scan_then_replace)
        assert store.reindex() == 1  # Stores the folder link's placeholder signature.
    link.unlink()
    link.symlink_to(outside, target_is_directory=True)  # A link again before the next refresh.
    with pytest.raises(StoreIntegrityError) as caught:
        store.refresh()
    assert "memory/attachments" in str(caught.value)
    assert store.get(fact.id).value == "tea"  # The previous index is left as it was.
    with pytest.raises(StoreIntegrityError) as reopened:
        MemoryStore(store.root)
    assert "memory/attachments" in str(reopened.value)


# ==========================================================================
# Records in other encodings and messages that name files
# ==========================================================================


def names_file(text, relative):
    """True when ``relative`` appears in ``text`` as a whole path, not glued to other text."""
    return re.search(r"(?<![\w./-])" + re.escape(relative) + r"(?![\w./-])", text) is not None


# UTF-16 detection


@pytest.mark.parametrize("damage", [
    lambda raw: raw + b"\n",  # A stray single byte: an odd length is never valid UTF-16.
    lambda raw: raw + "\ud800".encode("utf-16-le", errors="surrogatepass"),  # A lone surrogate.
], ids=["odd-length", "lone-surrogate"])
def test_a_damaged_utf16_note_that_mentions_the_marker_inline_is_still_a_note(store, damage):
    """A UTF-16 note naming the marker mid-sentence stays a note even if its bytes are damaged."""
    note = damage(("Records start with " + MARKER + " lines.\n").encode("utf-16-le"))
    (store.memory_dir / "note.md").write_bytes(note)
    assert store.refresh() is True and store.all_facts() == []


@pytest.mark.parametrize("codec", ["utf-16-le", "utf-16-be"])
def test_a_truncated_utf16_record_is_reported_by_file_name(store, codec):
    """A UTF-16 record with an odd trailing byte is reported, naming the file."""
    record = hand_written_record().encode(codec) + b"\n"
    (store.memory_dir / "hand.md").write_bytes(record)
    with pytest.raises(StoreIntegrityError) as caught:
        store.refresh()
    assert names_file(str(caught.value), "memory/hand.md")


@pytest.mark.parametrize("prefix", ["X ", "XX", "X\t", "\ufeffX "],
                         ids=["x-space", "xx", "x-tab", "bom-x"])
@pytest.mark.parametrize("codec", ["utf-8", "utf-16-le"])
def test_marker_text_after_other_text_on_the_first_line_is_a_note(store, prefix, codec):
    """A first line holding text before the marker is not a marker line, in UTF-8 or UTF-16."""
    note = (prefix + MARKER + "\n```json\n{}\n```\n").encode(codec)
    (store.memory_dir / "note.md").write_bytes(note)
    assert store.refresh() is True and store.all_facts() == []


@pytest.mark.parametrize("codec", ["utf-16", "utf-16-le", "utf-16-be"])
def test_a_utf16_record_with_classic_mac_line_endings_is_reported(store, codec):
    """A UTF-16 record whose lines end in a bare carriage return is reported, not skipped."""
    record = hand_written_record().replace("\n", "\r").encode(codec)
    (store.memory_dir / "hand.md").write_bytes(record)
    with pytest.raises(StoreIntegrityError) as caught:
        store.refresh()
    assert names_file(str(caught.value), "memory/hand.md")


# Messages naming records and files


def test_a_duplicate_id_on_disk_names_the_id_and_both_files(store):
    """Two files holding the same record ID are reported with that ID and both file names."""
    fact_id = "c" * 32
    (store.memory_dir / "one.md").write_text(hand_written_record(fact_id), encoding="utf-8")
    (store.memory_dir / "two.md").write_text(hand_written_record(fact_id), encoding="utf-8")
    with pytest.raises(StoreIntegrityError) as caught:
        store.reindex()
    text = str(caught.value)
    assert fact_id in text
    assert names_file(text, "memory/one.md") and names_file(text, "memory/two.md")


def test_a_conflict_between_two_files_names_each_file_exactly(store):
    """A duplicate effective time on disk names each of the two files as a whole path."""
    (store.memory_dir / "one.md").write_text(hand_written_record("d" * 32, "tea"), encoding="utf-8")
    (store.memory_dir / "two.md").write_text(hand_written_record("e" * 32, "coffee"),
                                             encoding="utf-8")
    with pytest.raises(StoreIntegrityError) as caught:
        store.reindex()
    text = str(caught.value)
    assert names_file(text, "memory/one.md") and names_file(text, "memory/two.md")


def test_a_correction_predating_its_target_names_both_files_exactly(store):
    """A correction recorded before its target, found on disk, names both files as whole paths."""
    first = store.add("person", "drink", "tea", valid_from="2025-01-01", recorded_at="2025-01-05")
    second = store.correct(first.id, "green tea", recorded_at="2025-01-06")
    path = store.path_for(second.id)
    path.write_bytes(path.read_bytes().replace(b"2025-01-06", b"2025-01-04"))
    with pytest.raises(StoreIntegrityError) as caught:
        store.reindex()
    text = str(caught.value)
    for fact in (first, second):
        assert names_file(text, store.path_for(fact.id).relative_to(store.root).as_posix())


def test_a_write_conflict_reads_like_the_same_conflict_on_disk_without_file_names(store):
    """A conflict caught on write is described as on open, minus the file list it cannot have."""
    store.add("person", "drink", "tea", valid_from="2025-01-01", recorded_at="2025-01-01")
    with pytest.raises(MemoryConflictError) as on_write:
        store.add("person", "drink", "coffee", valid_from="2025-01-01")
    written = str(on_write.value)
    assert "memory/" not in written and ".md" not in written
    (store.memory_dir / "hand.md").write_text(hand_written_record("f" * 32, "coffee"),
                                              encoding="utf-8")
    with pytest.raises(StoreIntegrityError) as on_open:
        store.reindex()
    opened = str(on_open.value)
    assert opened.startswith(written)
    assert names_file(opened[len(written):], "memory/hand.md")


def test_a_correction_recorded_too_early_reads_like_the_same_conflict_on_disk(store):
    """A correction dated before its target is described on write as on open, minus file names."""
    first = store.add("person", "drink", "tea", valid_from="2025-01-01", recorded_at="2025-01-05")
    with pytest.raises(ValueError) as on_write:
        store.correct(first.id, "green tea", recorded_at="2025-01-04")
    written = str(on_write.value)
    assert "memory/" not in written and ".md" not in written
    second = store.correct(first.id, "green tea", recorded_at="2025-01-06")
    path = store.path_for(second.id)
    path.write_bytes(path.read_bytes().replace(b"2025-01-06", b"2025-01-04"))
    with pytest.raises(StoreIntegrityError) as on_open:
        store.reindex()
    opened = str(on_open.value)
    assert opened.startswith(written)
    assert names_file(opened[len(written):], path.relative_to(store.root).as_posix())


# ==========================================================================
# What the index file holds for each record
# ==========================================================================


TAGS = ["work", "café", "日本語", "😀"]


LINKS = ["Zoë Ångström", "Analytical Engine"]


def _index_rows(root: Path, query: str, parameters=()):
    """Run ``query`` on the index file on disk, outside the store."""
    with closing(sqlite3.connect(str(Path(root) / ".memoryos" / "index.sqlite3"))) as raw:
        return raw.execute(query, parameters).fetchall()


def _add_non_ascii_record(root: Path):
    with MemoryStore(root) as store:
        return store.add("Zoë", "favourite drink", "café au lait", valid_from="2025-01-01",
                         evidence="Said so in München", tags=TAGS, links=LINKS)


def test_index_keeps_non_ascii_tags_and_links_unescaped(tmp_path):
    """The index file stores tags and links as one-line JSON lists with non-ASCII text as written."""
    fact = _add_non_ascii_record(tmp_path)
    expected = [('["work", "café", "日本語", "😀"]', '["Zoë Ångström", "Analytical Engine"]')]
    query = "SELECT tags, links FROM facts WHERE id = ?"
    assert _index_rows(tmp_path, query, (fact.id,)) == expected
    with MemoryStore(tmp_path) as store:
        assert store.reindex() == 1
    assert _index_rows(tmp_path, query, (fact.id,)) == expected


def test_index_payload_keeps_non_ascii_text_unescaped(tmp_path):
    """The index file stores each record's JSON payload with non-ASCII text as written."""
    fact = _add_non_ascii_record(tmp_path)
    expected = json.dumps(fact.to_dict(), ensure_ascii=False)
    query = "SELECT payload FROM facts WHERE id = ?"
    [(payload,)] = _index_rows(tmp_path, query, (fact.id,))
    assert payload == expected
    assert "café au lait" in payload and "München" in payload and "\\u" not in payload
    with MemoryStore(tmp_path) as store:
        assert store.reindex() == 1
        assert store.get(fact.id) == fact
    assert _index_rows(tmp_path, query, (fact.id,)) == [(expected,)]


def test_index_stores_retraction_flag_as_integer_despite_a_host_bool_adapter(
        tmp_path, monkeypatch):
    """The retraction flag stays an INTEGER 0/1 even if the host process adapts booleans for SQLite."""
    monkeypatch.setitem(sqlite3.adapters, (bool, sqlite3.PrepareProtocol),
                        lambda flag: "yes" if flag else "no")
    with MemoryStore(tmp_path) as store:
        live = store.add("person", "drink", "tea", valid_from="2025-01-01",
                         recorded_at="2025-01-01")
        tombstone = store.retract(live.id, recorded_at="2025-02-01")
        assert store.snapshot(as_of="2025-03-01", known_at="2025-03-01") == []
    query = "SELECT id, typeof(deleted), deleted FROM facts ORDER BY recorded_at"
    expected = [(live.id, "integer", 0), (tombstone.id, "integer", 1)]
    assert _index_rows(tmp_path, query) == expected
    with MemoryStore(tmp_path) as store:
        assert store.reindex() == 2
        assert store.get(tombstone.id).deleted is True
    assert _index_rows(tmp_path, query) == expected


# ==========================================================================
# Record files are UTF-8 whatever the default encoding
# ==========================================================================


def _latin1_default(encoding, stacklevel=2):
    """The default text encoding of a POSIX system with a Latin-1 locale."""
    return "latin-1" if encoding is None else encoding


def _json_block(raw):
    """The canonical JSON object of a record file, decoded strictly as UTF-8."""
    return json.loads(raw.decode("utf-8").split("```json\n", 1)[1].split("\n```", 1)[0])


def test_records_with_latin1_text_are_utf8_bytes_under_a_latin1_default(store, monkeypatch):
    """Text that Latin-1 could encode is still written to disk as UTF-8 bytes."""
    monkeypatch.setattr(io, "text_encoding", _latin1_default)
    fact = store.add("Zoë", "city", "Zürich", valid_from="2025-01-01")
    raw = store.path_for(fact.id).read_bytes()
    assert raw == MemoryStore._render(fact).encode("utf-8")
    assert "# Zoë · city".encode("utf-8") in raw
    assert _json_block(raw)["value"] == "Zürich"
    assert store.reindex() == 1  # Rebuilt from the Markdown record alone.
    assert store.get(fact.id) == fact


def test_records_with_non_latin1_text_are_written_under_a_latin1_default(store, monkeypatch):
    """Characters outside Latin-1 are stored, not rejected, and rebuild from Markdown intact."""
    monkeypatch.setattr(io, "text_encoding", _latin1_default)
    fact = store.add("Alex", "city", "東京 ☕", valid_from="2025-01-01")
    corrected = store.correct(fact.id, "大阪")
    for record in (fact, corrected):
        raw = store.path_for(record.id).read_bytes()
        assert raw == MemoryStore._render(record).encode("utf-8")
        assert _json_block(raw)["value"] == record.value
    assert store.reindex() == 2
    assert [f.value for f in store.audit_trail("Alex", "city")] == ["東京 ☕", "大阪"]


def test_a_scanned_file_name_that_is_not_utf8_is_named_in_the_error(store):
    """A non-UTF-8 name (as a POSIX scan reports it) is reported by name before any file is read."""
    with pytest.raises(ValueError, match=r"^memory/caf\\udce9\.md: the file name is not valid UTF-8"):
        store._load({"memory/caf\udce9.md": (1, 1)})


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_a_linked_index_log_file_is_refused_by_its_exact_name(tmp_path, monkeypatch, suffix):
    """The link check finds the log files by their exact names, as case-sensitive disks need."""
    MemoryStore(tmp_path).close()
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of the store\n", encoding="utf-8")
    (tmp_path / ".memoryos" / ("index.sqlite3" + suffix)).symlink_to(outside)
    real_is_symlink = Path.is_symlink

    def case_sensitive_is_symlink(path):
        # Names in the index folder must match exactly, as on Linux file systems.
        if path.parent.name == ".memoryos" and path.name not in os.listdir(path.parent):
            return False
        return real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", case_sensitive_is_symlink)
    with pytest.raises(ValueError, match="symbolic link"):
        MemoryStore(tmp_path)
