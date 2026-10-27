"""Regression tests for the 0.2.1 and 0.2.2 fixes.

Each test pins one behavior that an earlier release got wrong, and would fail
if that fix were undone. HTTP and command-line regressions live in
test_api.py and test_cli.py; retrieval and extraction pins added in 0.2.2
live in test_retrieval.py and test_extraction.py.
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import types
import unicodedata
from array import array
from contextlib import closing
from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
from helpers import MARKER, hand_written_record

import temporal_memory_os.embeddings as embeddings_module
import temporal_memory_os.extraction as extraction_module
import temporal_memory_os.retrieval as retrieval_module
import temporal_memory_os.store as store_module
from temporal_memory_os import __version__, cli
from temporal_memory_os.demo import DEMO_SOURCE, seed_demo
from temporal_memory_os.embeddings import (STOPWORDS, EmbeddingUnavailableError, HashingEmbedder,
                                           SentenceTransformerEmbedder, create_embedder,
                                           tokenize)
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.models import normalize_time
from temporal_memory_os.retrieval import MAX_KEYWORD_TERMS, HybridRetriever, VectorCache
from temporal_memory_os.store import MemoryConflictError, MemoryStore, StoreIntegrityError

# Spelled with escapes: the two forms look identical, and an editor that
# normalizes text on save would otherwise make the tests below compare a
# string with itself.
RESUME_NFC = "R\u00e9sum\u00e9"
RESUME_NFD = unicodedata.normalize("NFD", RESUME_NFC)
assert RESUME_NFD == "Re\u0301sume\u0301" != RESUME_NFC
BOM = "\ufeff"


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "store") as memory:
        yield memory


def index_connection(root):
    return sqlite3.connect(str(Path(root) / ".memoryos" / "index.sqlite3"))


class Controlled:
    """A learned-model stand-in whose vectors are chosen by the test."""

    semantic = True

    def __init__(self, name, document, query=(1.0, 0.0)):
        self.name = name
        self.document = list(document)
        self.query = list(query)
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        # The first text of every call is the query; the rest are documents.
        return [self.query] + [self.document for _ in texts[1:]]


class CountingHashing(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return super().embed(texts)


def fake_sentence_transformers(monkeypatch, model_class):
    module = types.SimpleNamespace(SentenceTransformer=model_class)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


# models.py: one explicit time grammar on every Python version


@pytest.mark.parametrize("text, expected", [
    ("2025-01-01", "2025-01-01T00:00:00.000000Z"),
    ("2025-01-01T08:30", "2025-01-01T08:30:00.000000Z"),
    ("2025-01-01t08:30", "2025-01-01T08:30:00.000000Z"),
    ("2025-01-01 08:30", "2025-01-01T08:30:00.000000Z"),
    ("2025-01-01T08:30:15", "2025-01-01T08:30:15.000000Z"),
    ("2025-01-01T08:30:15.5", "2025-01-01T08:30:15.500000Z"),
    ("2025-01-01T08:30:15.123456", "2025-01-01T08:30:15.123456Z"),
    ("2025-01-01T08:30Z", "2025-01-01T08:30:00.000000Z"),
    ("2025-01-01T08:30:15z", "2025-01-01T08:30:15.000000Z"),
    ("2025-01-01T08:30+08:00", "2025-01-01T00:30:00.000000Z"),
    ("2025-01-01T08:30:15.25-05:30", "2025-01-01T14:00:15.250000Z"),
    ("  2025-01-01T08:30  ", "2025-01-01T08:30:00.000000Z"),
])
def test_time_grammar_accepts_exactly_the_documented_forms(text, expected):
    assert normalize_time(text) == expected


@pytest.mark.parametrize("text", [
    "20250101",                         # compact date
    "2025-01-01T0830",                  # compact time
    "20250101T083000",                  # compact date and time
    "2025-01-01T08",                    # hour only
    "2025-W01-1",                       # week date
    "2025-001",                         # ordinal date
    "2025-01-01T08:30+0800",            # compact offset
    "2025-01-01T08:30+08",              # hour-only offset
    "2025-01-01T08:30+08:00:00",        # offset with seconds
    "2025-01-01T08:30:15,5",            # comma fraction
    "2025-01-01T08:30:15.1234567",      # more than six fraction digits
    "2025-01-01T08:30:15.",             # empty fraction
    "2025-01-01T",                      # separator without a time
    "2025-01-01X08:30",                 # unknown separator
    "2025-01-01T24:00",                 # hour out of range
    "2025-01-01T08:30+24:00",           # offset out of range
    "2025-01-01T08:30 UTC",             # zone name
    "\u0662\u0660\u0662\u0665-\u0660\u0661-\u0660\u0661",  # Arabic-Indic digits
    "\uff12\uff10\uff12\uff15-01-01",       # full-width digits
    "2025-01-01T\u0660\u0668:30",         # Arabic-Indic digits in the time
])
def test_time_grammar_rejects_forms_outside_the_documented_grammar(text):
    with pytest.raises(ValueError):
        normalize_time(text)


@pytest.mark.parametrize("value", [
    "0001-01-01T00:00:00+01:00",
    "9999-12-31T23:59:59-01:00",
    datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=5))),
])
def test_out_of_range_instants_raise_value_error_not_overflow(value, store):
    with pytest.raises(ValueError, match="outside the supported range"):
        normalize_time(value)
    # The same input through the store is an ordinary validation error.
    with pytest.raises(ValueError):
        store.add("person", "drink", "tea", valid_from=value)
    with pytest.raises(ValueError):
        store.snapshot(as_of=value)


def test_tags_and_links_are_immutable_tuples_on_shared_records(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01",
                     tags=["a", "a", "b"], links=["Acme"])
    cached = store.get(fact.id)
    assert cached.tags == ("a", "b") and cached.links == ("Acme",)
    with pytest.raises(AttributeError):
        cached.tags.append("INJECTED")
    exported = cached.to_dict()
    assert exported["tags"] == ["a", "b"] and exported["links"] == ["Acme"]
    exported["tags"].append("INJECTED")
    exported["links"].append("X")
    assert store.get(fact.id).tags == ("a", "b")
    assert store.snapshot(as_of="2025-02-01")[0].links == ("Acme",)
    assert store.history("person")[0].to_dict()["tags"] == ["a", "b"]


# store.py: index schema, opening and recovery


def tables(memory):
    return {row[0] for row in memory.connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_old_full_text_table_is_kept_for_running_older_processes(tmp_path):
    with MemoryStore(tmp_path) as memory:
        fact = memory.add("person", "drink", "tea", valid_from="2025-01-01")
        assert "facts_fts" not in tables(memory)  # New indexes never create it.
    # A process of an older release made the table and is still running.
    older = index_connection(tmp_path)
    older.execute("CREATE VIRTUAL TABLE facts_fts USING fts5(id UNINDEXED, value)")
    older.execute("PRAGMA user_version = 2")
    older.commit()
    try:
        with MemoryStore(tmp_path) as memory:
            assert "facts_fts" in tables(memory)
            assert memory.connection.execute("PRAGMA user_version").fetchone()[0] == 3
            assert memory.get(fact.id).value == "tea"
            assert memory.add("person", "city", "Riverton", valid_from="2025-01-01")
        # Its next write still finds the table it relies on.
        older.execute("INSERT INTO facts_fts VALUES (?, ?)", (fact.id, "tea"))
        older.commit()
    finally:
        older.close()


def test_newer_index_message_names_the_folder_to_delete_without_absolute_path(tmp_path):
    root = tmp_path / "store"
    MemoryStore(root).close()
    connection = index_connection(root)
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match=r"Delete the \.memoryos folder") as caught:
        MemoryStore(root)
    assert str(tmp_path) not in str(caught.value)
    assert str(root.resolve()) not in str(caught.value)


def test_first_open_retries_while_another_process_creates_the_schema(tmp_path, monkeypatch):
    real = MemoryStore._initialize
    attempts = []

    def busy_once(self, connection):
        attempts.append(1)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(self, connection)

    monkeypatch.setattr(MemoryStore, "_initialize", busy_once)
    with MemoryStore(tmp_path, repair_index=True) as memory:
        assert memory.add("person", "drink", "tea", valid_from="2025-01-01")
    assert len(attempts) == 2
    # The lock error is not mistaken for damage, even when repair is allowed.
    assert set_aside(tmp_path) == []


def test_first_open_stops_retrying_a_lock_at_the_deadline(tmp_path, monkeypatch):
    def always_busy(self, connection):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(MemoryStore, "_initialize", always_busy)
    monkeypatch.setattr(store_module, "_LOCK_RETRY_SECONDS", 0.0)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        MemoryStore(tmp_path)


def set_aside(root):
    """Names of index files that a repair moved out of the way."""
    return sorted(path.name for path in (Path(root) / ".memoryos").iterdir()
                  if ".corrupt" in path.name)


def damage_header(root):
    """Replace the whole index with bytes SQLite cannot read ("not a database")."""
    garbage = b"this is not a database " * 200
    (Path(root) / ".memoryos" / "index.sqlite3").write_bytes(garbage)
    return garbage


def damage_pages(root):
    """Keep the first page (header and schema) but overwrite every table page, so
    the damage only shows when the first refresh reads them ("malformed")."""
    index = Path(root) / ".memoryos" / "index.sqlite3"
    raw = index.read_bytes()
    page_size = int.from_bytes(raw[16:18], "big")
    assert 0 < page_size < len(raw)
    damaged = raw[:page_size] + b"\xab" * (len(raw) - page_size)
    index.write_bytes(damaged)
    return damaged


def damage_schema(root):
    """Break the SQL of one index in the schema table, so SQLite cannot load the schema
    ("malformed database schema")."""
    index = Path(root) / ".memoryos" / "index.sqlite3"
    with closing(sqlite3.connect(str(index))) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # Everything in the main file.
    raw = index.read_bytes()
    assert raw.count(b"CREATE INDEX facts_recorded") == 1
    damaged = raw.replace(b"CREATE INDEX facts_recorded", b"CREATX INDEX facts_recorded")
    index.write_bytes(damaged)
    return damaged


def damage_cell_text(root):
    """Turn one byte of a stored file hash into invalid UTF-8; the page itself stays valid."""
    index = Path(root) / ".memoryos" / "index.sqlite3"
    with closing(sqlite3.connect(str(index))) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        [(digest,)] = connection.execute("SELECT sha256 FROM sources").fetchall()
    raw = index.read_bytes()
    assert raw.count(digest.encode()) == 1
    index.write_bytes(raw.replace(digest.encode(), b"\xff" + digest.encode()[1:]))


def damage_schema_text(root):
    """Turn one byte of an index definition in the schema into invalid UTF-8."""
    index = Path(root) / ".memoryos" / "index.sqlite3"
    with closing(sqlite3.connect(str(index))) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = index.read_bytes()
    assert raw.count(b"CREATE INDEX facts_recorded") == 1
    index.write_bytes(raw.replace(b"CREATE INDEX facts_recorded", b"CREAT\xff INDEX facts_recorded"))


@pytest.mark.parametrize("damage", [damage_cell_text, damage_schema_text],
                         ids=["cell", "schema"])
def test_damage_that_leaves_unreadable_text_is_reported_and_repaired(tmp_path, damage):
    with MemoryStore(tmp_path) as memory:
        fact = memory.add("person", "drink", "tea", valid_from="2025-01-01")
    damage(tmp_path)
    with pytest.raises(StoreIntegrityError, match="memoryos reindex"):
        MemoryStore(tmp_path)
    with MemoryStore(tmp_path, repair_index=True) as repaired:
        assert repaired.get(fact.id).value == "tea"
    assert len(set_aside(tmp_path)) >= 1


DAMAGE = pytest.mark.parametrize("damage, cause", [
    (damage_header, "file is not a database"),
    (damage_pages, "database disk image is malformed"),
    (damage_schema, 'malformed database schema (facts_recorded) - near "CREATX": syntax error'),
], ids=["header", "pages", "schema"])
STAMPED = re.compile(r"index\.sqlite3(-wal|-shm)?\.corrupt-\d{8}T\d{6}")


@DAMAGE
def test_damaged_index_raises_store_integrity_error_and_is_left_in_place(tmp_path, damage,
                                                                         cause):
    with MemoryStore(tmp_path) as memory:
        memory.add("person", "drink", "tea", valid_from="2025-01-01")
    damage(tmp_path)
    for _ in range(2):  # Opening again does not change anything either.
        with pytest.raises(StoreIntegrityError) as caught:
            MemoryStore(tmp_path)
        message = str(caught.value)
        assert message == (
            "The index .memoryos/index.sqlite3 is damaged ({}). Stop other programs using this "
            "memory directory and run `memoryos reindex` to rebuild it from the Markdown "
            "records.".format(cause))
        # Only an explicit repair renames files that other processes may have open.
        assert set_aside(tmp_path) == []
        assert (tmp_path / ".memoryos" / "index.sqlite3").exists()


@DAMAGE
def test_repair_index_sets_damaged_files_aside_with_a_timestamp(tmp_path, damage, cause):
    with MemoryStore(tmp_path) as memory:
        fact = memory.add("person", "drink", "tea", valid_from="2025-01-01")
    damaged = damage(tmp_path)
    with MemoryStore(tmp_path, repair_index=True) as memory:
        assert memory.get(fact.id).value == "tea"
        assert memory.refresh() is False
    [aside] = set_aside(tmp_path)
    assert STAMPED.fullmatch(aside) and aside.startswith("index.sqlite3.corrupt-"), aside
    if damage is damage_header:  # Kept for inspection, never parsed.
        assert (tmp_path / ".memoryos" / aside).read_bytes() == damaged
    with MemoryStore(tmp_path) as memory:  # The rebuilt index is an ordinary one.
        assert [f.id for f in memory.all_facts()] == [fact.id]


@pytest.mark.parametrize("code, damaged", [
    (11, True), (26, True), (267, True), (779, True),  # Corruption, extended codes included.
    (5, False), (8, False), (10, False), (13, False), (14, False), (None, False),
])
def test_sqlite_result_codes_decide_what_counts_as_damage(code, damaged):
    """Corruption codes count as damage whatever SQLite's text says; other codes never do."""
    error = sqlite3.DatabaseError("unexpected wording")
    if code is not None:
        error.sqlite_errorcode = code  # Set by Python 3.11 and later.
    assert store_module._damaged(error) is damaged


class DamagedConnection:
    """Stands in for SQLite reporting damage before it touches any file (SQLite
    itself deletes a log it cannot use, which would hide what repair renames)."""

    def execute(self, *_arguments):
        raise sqlite3.DatabaseError("file is not a database")

    def close(self):
        pass


def test_repair_sets_the_log_and_shared_memory_files_aside_with_one_stamp(tmp_path,
                                                                         monkeypatch):
    with MemoryStore(tmp_path) as memory:
        fact = memory.add("person", "drink", "tea", valid_from="2025-01-01")
    state = tmp_path / ".memoryos"
    for suffix in ("", "-wal", "-shm"):
        (state / ("index.sqlite3" + suffix)).write_bytes(b"damaged" + suffix.encode())
    real, calls = MemoryStore._connect, []

    def connect(self):
        calls.append(1)
        return DamagedConnection() if len(calls) == 1 else real(self)

    monkeypatch.setattr(MemoryStore, "_connect", connect)
    with MemoryStore(tmp_path, repair_index=True) as memory:
        assert memory.get(fact.id).value == "tea"
    aside = set_aside(tmp_path)
    [stamp] = {name.rsplit("-", 1)[1] for name in aside}
    assert aside == sorted("index.sqlite3{}.corrupt-{}".format(suffix, stamp)
                           for suffix in ("", "-wal", "-shm"))
    assert all(STAMPED.fullmatch(name) for name in aside)
    for suffix in ("", "-wal", "-shm"):  # Moved, not rewritten.
        assert (state / "index.sqlite3{}.corrupt-{}".format(suffix, stamp)).read_bytes() == (
            b"damaged" + suffix.encode())


def test_a_later_repair_keeps_the_earlier_damaged_copy(tmp_path, monkeypatch):
    MemoryStore(tmp_path).close()
    stamps = iter(["20260101T000000", "20260101T000001"])
    monkeypatch.setattr(store_module.time, "strftime", lambda _format: next(stamps))
    index = tmp_path / ".memoryos" / "index.sqlite3"
    for content in (b"first damage " * 100, b"second damage " * 100):
        index.write_bytes(content)
        MemoryStore(tmp_path, repair_index=True).close()
    assert set_aside(tmp_path) == ["index.sqlite3.corrupt-20260101T000000",
                                   "index.sqlite3.corrupt-20260101T000001"]
    assert (index.parent / "index.sqlite3.corrupt-20260101T000000").read_bytes().startswith(
        b"first damage")


def test_index_repair_is_attempted_only_once(tmp_path, monkeypatch):
    attempts = []

    def damaged(self, connection):
        attempts.append(1)
        if len(attempts) > 4:
            raise RuntimeError("recovery looped")
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(MemoryStore, "_initialize", damaged)
    with pytest.raises(StoreIntegrityError, match="not a database"):
        MemoryStore(tmp_path)
    assert len(attempts) == 1  # Without repair_index nothing is retried or renamed.
    assert set_aside(tmp_path) == []
    with pytest.raises(StoreIntegrityError, match="not a database"):
        MemoryStore(tmp_path, repair_index=True)
    assert len(attempts) == 3  # One repair, then the error.
    assert len(set_aside(tmp_path)) == 1


@pytest.mark.parametrize("stage", ["open", "first-refresh"])
@pytest.mark.parametrize("message", [
    "attempt to write a readonly database", "database or disk is full", "disk I/O error",
    "locking protocol", "database is locked",
])
def test_errors_other_than_damage_never_set_the_index_aside(tmp_path, monkeypatch, stage,
                                                            message):
    with MemoryStore(tmp_path) as memory:
        fact = memory.add("person", "drink", "tea", valid_from="2025-01-01")
    monkeypatch.setattr(store_module, "_LOCK_RETRY_SECONDS", 0.0)

    def failing(*_arguments):
        raise sqlite3.OperationalError(message)

    with monkeypatch.context() as patch:
        patch.setattr(MemoryStore, "_initialize" if stage == "open" else "refresh", failing)
        with pytest.raises(sqlite3.OperationalError, match=message):
            MemoryStore(tmp_path, repair_index=True)
    assert set_aside(tmp_path) == []
    with MemoryStore(tmp_path) as memory:  # The live index was never moved.
        assert memory.get(fact.id).value == "tea"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_read_only_index_is_reported_and_never_renamed(tmp_path):
    with MemoryStore(tmp_path) as memory:
        fact = memory.add("person", "drink", "tea", valid_from="2025-01-01")
    state = tmp_path / ".memoryos"
    (state / "index.sqlite3").chmod(0o400)
    try:
        for repair in (False, True):
            # Opening a current index writes nothing, so reading still works;
            # a write fails plainly and nothing is treated as damage.
            with MemoryStore(tmp_path, repair_index=repair) as reader:
                assert reader.get(fact.id).value == "tea"
                with pytest.raises(sqlite3.OperationalError, match="readonly"):
                    reader.add("person", "city", "Riverton", valid_from="2025-01-01")
            assert set_aside(tmp_path) == []
    finally:
        for path in state.iterdir():  # SQLite gave its log files the same mode.
            path.chmod(0o600)
    with MemoryStore(tmp_path) as memory:
        assert memory.get(fact.id).value == "tea"


def test_index_file_is_created_private(tmp_path):
    previous = os.umask(0o022)
    try:
        MemoryStore(tmp_path).close()
    finally:
        os.umask(previous)
    mode = (tmp_path / ".memoryos" / "index.sqlite3").stat().st_mode & 0o777
    assert mode == 0o600


def test_an_existing_world_readable_index_is_made_private(tmp_path):
    MemoryStore(tmp_path).close()
    state = tmp_path / ".memoryos"
    for suffix in ("", "-wal", "-shm"):  # As left by an older release.
        path = state / ("index.sqlite3" + suffix)
        path.touch()
        path.chmod(0o644)
    with MemoryStore(tmp_path) as memory:
        memory.add("person", "drink", "tea", valid_from="2025-01-01")
        modes = {path.name: path.stat().st_mode & 0o777 for path in state.iterdir()}
    assert modes == {"index.sqlite3": 0o600, "index.sqlite3-wal": 0o600,
                     "index.sqlite3-shm": 0o600}


def test_refresh_with_nothing_changed_does_not_wait_for_a_writer(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    hour_ago = time.time() - 3600
    os.utime(store.path_for(fact.id), (hour_ago, hour_ago))
    assert store.refresh() is True  # Re-verified; no longer recently modified.
    with MemoryStore(store.root) as writer:
        writer.connection.execute("BEGIN IMMEDIATE")  # For example a Git checkpoint.
        try:
            store.connection.execute("PRAGMA busy_timeout = 50")
            assert store.refresh() is False
            assert store.get(fact.id).value == "tea"
        finally:
            writer.connection.execute("ROLLBACK")


# store.py: what the scan reads


def test_names_starting_with_a_dot_are_ignored(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    record = store.path_for(fact.id)
    # Emacs marks an unsaved buffer with a dangling symbolic link.
    os.symlink("someone@laptop.12345:1729700000", str(record.parent / (".#" + record.name)))
    # A copy in a hidden folder or a hidden file would be a duplicate ID if read.
    hidden = store.memory_dir / ".obsidian"
    hidden.mkdir()
    (hidden / "copy.md").write_bytes(record.read_bytes())
    (record.parent / ".backup.md").write_bytes(record.read_bytes())
    assert store.refresh() is False
    assert store.reindex() == 1
    with MemoryStore(store.root) as reopened:
        assert [f.id for f in reopened.all_facts()] == [fact.id]


def test_other_symlinked_markdown_is_rejected_with_a_relative_path(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    (store.memory_dir / "general" / "alias.md").symlink_to(store.path_for(fact.id))
    with pytest.raises(ValueError, match="symbolic links") as caught:
        store.refresh()
    assert str(caught.value).startswith("memory/general/alias.md:")
    assert str(store.root) not in str(caught.value)


def test_non_utf8_note_is_ignored(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    note = store.memory_dir / "notes" / "windows-note.md"
    note.parent.mkdir()
    note.write_bytes("Café notes, saved by an old editor\n".encode("cp1252"))
    assert store.refresh() is True
    assert [f.id for f in store.all_facts()] == [fact.id]
    with MemoryStore(store.root) as reopened:
        assert reopened.get(fact.id).value == "tea"


def test_non_utf8_note_quoting_the_marker_inline_is_a_note(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    note = store.memory_dir / "notes" / "about-records.md"
    note.parent.mkdir()
    note.write_bytes(("Records start with " + MARKER + " lines. Café\n").encode("latin-1"))
    assert store.refresh() is True
    assert [f.id for f in store.all_facts()] == [fact.id]


def test_notes_named_like_records_are_still_notes(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    # Joplin's exports and sync folders name every note with 32 hex characters.
    folder = store.memory_dir / "joplin"
    folder.mkdir()
    (folder / "0f3c8a6e9b2d4c1fa7e5b3d2c1a0f9e8.md").write_text(
        "Shopping list\n\nid: 0f3c8a6e9b2d4c1fa7e5b3d2c1a0f9e8\n", encoding="utf-8")
    (folder / ("a" * 32 + ".md")).write_bytes(b"")
    assert store.refresh() is True
    assert [f.id for f in store.all_facts()] == [fact.id]
    with MemoryStore(store.root) as reopened:
        assert [f.id for f in reopened.all_facts()] == [fact.id]


EMPTIED = "this file held a memory record but no longer has a valid record block"


@pytest.mark.parametrize("damage", [
    lambda raw: b"",
    lambda raw: raw[:raw.index(b"<!--")],
    lambda raw: raw.replace(MARKER.encode(), MARKER.encode() + b" (checked)"),
    lambda raw: raw.replace(b".\n\n" + MARKER.encode(), b". " + MARKER.encode()),
], ids=["emptied", "truncated", "marker-line-edited", "marker-joined-to-a-sentence"])
def test_a_record_that_loses_its_block_raises_with_its_relative_path(store, damage):
    first = store.add("person", "drink", "tea", valid_from="2025-01-01")
    fact = store.correct(first.id, "coffee")
    path = store.path_for(fact.id)
    path.write_bytes(damage(path.read_bytes()))
    for opened in (store.refresh, lambda: MemoryStore(store.root)):
        with pytest.raises(StoreIntegrityError, match=EMPTIED) as caught:
            opened()
        assert str(caught.value).startswith("memory/general/{}.md: ".format(fact.id))
        assert str(store.root) not in str(caught.value)
    # The superseded value does not come back; the old index is kept.
    assert [f.value for f in store.snapshot(as_of="2025-02-01")] == ["coffee"]
    path.unlink()  # The message's other way out: remove the record on purpose.
    assert store.refresh() is True
    assert [f.value for f in store.snapshot(as_of="2025-02-01")] == ["tea"]


def test_a_hand_named_record_that_loses_its_block_is_reported(store):
    record = store.memory_dir / "general" / "my-drink.md"
    record.parent.mkdir(exist_ok=True)
    record.write_text(hand_written_record(), encoding="utf-8")
    assert store.refresh() is True and store.get("b" * 32).value == "tea"
    record.write_text("Shopping list: tea\n", encoding="utf-8")
    with pytest.raises(StoreIntegrityError,
                       match="^memory/general/my-drink.md: " + EMPTIED):
        store.refresh()
    assert store.get("b" * 32).value == "tea"


@pytest.mark.parametrize("encode", [
    lambda text: b"\xef\xbb\xbf" + text.encode("utf-8"),
    lambda text: text.replace("\n", "\r\n").encode("utf-8"),
    lambda text: text.replace("\n", "\r").encode("utf-8"),
    lambda text: b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"),
], ids=["bom", "crlf", "cr-only", "bom-crlf"])
def test_records_with_bom_and_other_line_endings_are_read(store, encode):
    (store.memory_dir / "general").mkdir(exist_ok=True)
    (store.memory_dir / "general" / "hand-written.md").write_bytes(
        encode(hand_written_record()))
    assert store.refresh() is True
    assert store.get("b" * 32).value == "tea"


def test_engine_record_converted_to_cr_only_line_endings_is_read(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    path = store.path_for(fact.id)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r"))
    assert store.reindex() == 1
    assert store.get(fact.id).value == "tea"


@pytest.mark.parametrize("damage, message", [
    (lambda raw: raw + b"\n" + MARKER.encode() + b"\n", "expected one canonical"),
    (lambda raw: raw.replace(b'"tea"', b'"t\xe9a"'), "not valid UTF-8"),
    (lambda raw: raw.replace(b'"confidence": 1.0', b'"confidence": 7'), "confidence"),
    (lambda raw: raw.replace(b'"confidence": 1.0', b'"confidence": 1' + b"0" * 400),
     "confidence"),
    (lambda raw: raw.replace(b'"tea"', b'"t\\ud800a"'), "value contains characters"),
    (lambda raw: raw.replace(b'"tags": []', b'"tags": ["\\udfff"]'), "tags item contains"),
], ids=["two-markers", "not-utf8", "confidence", "huge-int-confidence", "lone-surrogate",
        "lone-surrogate-tag"])
def test_record_errors_use_paths_relative_to_the_store(store, damage, message):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    path = store.path_for(fact.id)
    raw = path.read_bytes()
    assert damage(raw) != raw
    path.write_bytes(damage(raw))
    for load in (store.reindex, store.refresh):
        with pytest.raises(StoreIntegrityError, match=message) as caught:
            load()
        assert str(caught.value).startswith("memory/general/{}.md: ".format(fact.id))
        assert str(store.root) not in str(caught.value)
    assert store.get(fact.id).value == "tea"


@pytest.mark.parametrize("fields, message", [
    (dict(entity="person\ud800"), "entity contains characters that are not valid Unicode"),
    (dict(value="t\udfffa"), "value contains characters"),
    (dict(evidence="\ud800"), "evidence contains characters"),
    (dict(tags=["ok", "\ud800"]), "tags item contains characters"),
    (dict(confidence=10 ** 400), "confidence must be a number between 0 and 1"),
], ids=["entity", "value", "evidence", "tag", "huge-int-confidence"])
def test_lone_surrogates_and_huge_confidences_are_value_errors(store, fields, message):
    arguments = dict(entity="person", attribute="drink", value="tea", valid_from="2025-01-01")
    with pytest.raises(ValueError, match=message):
        store.add(**{**arguments, **fields})
    assert list(store.memory_dir.rglob("*.md")) == []


# store.py: writes


class FailingCommit:
    """Wrap a connection (whose attributes are read-only) and fail one COMMIT."""

    def __init__(self, connection, error):
        self.wrapped = connection
        self.error = error

    def execute(self, sql, *parameters):
        if sql == "COMMIT" and self.error is not None:
            error, self.error = self.error, None
            raise error
        return self.wrapped.execute(sql, *parameters)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


@pytest.mark.parametrize("error", [sqlite3.OperationalError("disk I/O error"),
                                   KeyboardInterrupt()])
def test_commit_failure_removes_the_new_record_file(store, error):
    kept = store.add("person", "city", "Riverton", valid_from="2025-01-01")
    proxy = FailingCommit(store.connection, error)
    store.connection = proxy
    try:
        with pytest.raises(type(error)):
            store.add("person", "drink", "tea", valid_from="2025-01-01")
        assert [path.stem for path in store.memory_dir.rglob("*.md")] == [kept.id]
        proxy.error = type(error)("again")
        with pytest.raises(type(error)):
            store.correct(kept.id, "Lakeside")
        assert [path.stem for path in store.memory_dir.rglob("*.md")] == [kept.id]
    finally:
        store.connection = proxy.wrapped
    assert store.refresh() is False
    # A retry is not blocked by a leftover record.
    assert store.add("person", "drink", "tea", valid_from="2025-01-01").value == "tea"
    assert store.correct(kept.id, "Lakeside").value == "Lakeside"


def test_writes_flush_the_record_its_folder_and_every_new_folder(store, monkeypatch):
    flushed, folders = [], []
    monkeypatch.setattr(store_module, "_flush", lambda descriptor: flushed.append(descriptor))
    monkeypatch.setattr(store_module, "_flush_directory", lambda path: folders.append(path))
    memory = store.memory_dir
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01", topic="profile")
    assert flushed and folders == [memory, memory / "profile"]
    folders.clear()
    store.correct(fact.id, "coffee")  # An existing folder: only the rename is flushed.
    assert folders == [memory / "profile"]
    folders.clear()
    # Each new folder's entry is flushed in its parent, outermost first.
    store.add("person", "city", "Riverton", valid_from="2025-01-01", topic="projects/2024/q1")
    assert folders == [memory, memory / "projects", memory / "projects" / "2024",
                       memory / "projects" / "2024" / "q1"]


def test_revision_of_a_future_recorded_target_needs_no_explicit_time(store):
    imported = store.add("person", "drink", "tea", valid_from="2025-01-01",
                         recorded_at="2999-01-01")
    corrected = store.correct(imported.id, "coffee")
    assert corrected.recorded_at == "2999-01-01T00:00:00.000001Z"
    retracted = store.retract(corrected.id)
    assert retracted.recorded_at == "2999-01-01T00:00:00.000002Z"
    assert retracted.supersedes == corrected.id


def frozen_clock(monkeypatch, instant):
    """Make every write without an explicit time happen at ``instant``."""
    real = store_module.normalize_time
    monkeypatch.setattr(store_module, "normalize_time",
                        lambda value=None: real(instant if value is None else value))


def test_correcting_a_write_from_the_same_clock_tick_succeeds(store, monkeypatch):
    frozen_clock(monkeypatch, "2025-06-01T12:00:00.000000Z")
    first = store.add("person", "drink", "tea", valid_from="2025-01-01")
    second = store.add("person", "city", "Riverton", valid_from="2025-01-01")
    third = store.add("person", "pet", "cat", valid_from="2025-01-01")
    # Plain writes within one clock tick stay strictly ordered.
    assert [first.recorded_at, second.recorded_at, third.recorded_at] == [
        "2025-06-01T12:00:00.000000Z", "2025-06-01T12:00:00.000001Z",
        "2025-06-01T12:00:00.000002Z"]
    corrected = store.correct(second.id, "Lakeside")
    assert corrected.recorded_at == "2025-06-01T12:00:00.000003Z"


@pytest.mark.parametrize("ahead, expected", [
    ("12:00:00.000000", "12:00:00.000001"),  # The same clock tick.
    ("12:00:00.004000", "12:00:00.004001"),  # A writer whose clock is slightly fast.
    ("12:00:00.010000", "12:00:00.010001"),  # The edge of the 10 ms window.
    ("12:00:00.010001", "12:00:00.000000"),  # Further ahead counts as an import.
    ("12:00:00.500000", "12:00:00.000000"),
    ("2999-01-01", "12:00:00.000000"),
])
def test_writes_follow_the_latest_record_only_within_ten_milliseconds(store, monkeypatch, ahead,
                                                                     expected):
    frozen_clock(monkeypatch, "2025-06-01T12:00:00.000000Z")
    store.add("person", "drink", "tea", valid_from="2025-01-01",
              recorded_at=ahead if ahead.startswith("2999") else "2025-06-01T" + ahead + "Z")
    after = store.add("person", "city", "Riverton", valid_from="2025-01-01")
    assert after.recorded_at == "2025-06-01T" + expected + "Z"
    if expected == "12:00:00.000000":
        # A write recorded "now" is visible to the next query made now.
        assert after.id in {fact.id for fact in store.snapshot()}


def test_no_revision_can_follow_the_last_representable_instant(store):
    last = store.add("person", "drink", "tea", valid_from="2025-01-01",
                     recorded_at="9999-12-31T23:59:59.999999Z")
    for revise in (lambda: store.correct(last.id, "coffee"), lambda: store.retract(last.id)):
        with pytest.raises(ValueError, match=re.escape(
                "No later time can be recorded after 9999-12-31T23:59:59.999999Z")):
            revise()
    assert [f.id for f in store.all_facts()] == [last.id]
    assert list(store.memory_dir.rglob("*.md")) == [store.path_for(last.id)]


def test_revising_a_superseded_record_names_the_right_action(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    store.correct(fact.id, "coffee")
    with pytest.raises(MemoryConflictError, match="retract the latest revision instead"):
        store.retract(fact.id)
    with pytest.raises(MemoryConflictError, match="correct the latest revision instead"):
        store.correct(fact.id, "water")


@pytest.mark.parametrize("given", ["urgent", "", {"a", "b"}, {"k": 1}, 0],
                         ids=["string", "empty-string", "set", "dict", "zero"])
def test_tags_and_links_must_be_lists_of_strings(store, given):
    for field in ("tags", "links"):
        with pytest.raises(ValueError, match="^{} must be a list of strings$".format(field)):
            store.add("person", "drink", "tea", valid_from="2025-01-01", **{field: given})
    assert list(store.memory_dir.rglob("*.md")) == []
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01", tags=None,
                     links=("Acme",))
    assert (fact.tags, fact.links) == ((), ("Acme",))


def test_correct_keeps_confidence_and_topic_unless_given(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01", confidence=0.4,
                     topic="profile")
    kept = store.correct(fact.id, "green tea")
    assert (kept.confidence, kept.topic) == (0.4, "profile")
    assert store.path_for(kept.id).parent == store.memory_dir / "profile"
    changed = store.correct(kept.id, "black tea", confidence=0.9, topic="profile/drinks")
    assert (changed.confidence, changed.topic) == (0.9, "profile/drinks")
    assert store.path_for(changed.id).parent == store.memory_dir / "profile" / "drinks"


def test_audit_trail_lists_every_revision_in_recorded_order(store):
    first = store.add("person", "drink", "tea", valid_from="2025-01-01", recorded_at="2025-01-01")
    store.add("person", "city", "Riverton", valid_from="2025-01-01", recorded_at="2025-01-02")
    second = store.correct(first.id, "coffee", recorded_at="2025-01-03")
    third = store.retract(second.id, recorded_at="2025-01-04")
    later = store.add("person", "drink", "water", valid_from="2025-02-01",
                      recorded_at="2025-01-05")
    trail = store.audit_trail("person", "drink")
    assert [f.id for f in trail] == [first.id, second.id, third.id, later.id]
    assert [f.id for f in store.history("person", "drink")] == [third.id, later.id]


def test_audit_trail_puts_each_correction_after_its_target_when_times_tie(store, monkeypatch):
    # An importer may give a record and its corrections one batch timestamp.
    # IDs that sort the wrong way round make an ID tie-break list them backwards.
    ids = iter("{:032x}".format(number) for number in (0xF0, 0x30, 0x10, 0x20))
    with monkeypatch.context() as patch:
        patch.setattr(store_module.uuid, "uuid4", lambda: types.SimpleNamespace(hex=next(ids)))
        first = store.add("person", "drink", "tea", valid_from="2025-01-01",
                          recorded_at="2025-02-01")
        second = store.correct(first.id, "coffee", recorded_at="2025-02-01")
        third = store.correct(second.id, "water", recorded_at="2025-02-01")
        other = store.add("person", "drink", "milk", valid_from="2025-03-01",
                          recorded_at="2025-02-01")
    order = [f.id for f in store.audit_trail("person", "drink")]
    assert sorted(order) == sorted([first.id, second.id, third.id, other.id])
    assert order.index(first.id) < order.index(second.id) < order.index(third.id)
    with MemoryStore(store.root) as reopened:
        assert [f.id for f in reopened.audit_trail("person", "drink")] == order


# embeddings.py


@pytest.mark.parametrize("left, right", [
    ("Stra\u00dfe", "STRASSE"),
    ("\ufb01nance", "finance"),
    (RESUME_NFC, RESUME_NFD),
    ("\uff21\uff22\uff23", "abc"),
    ("\u0130stanbul", "\u0130STANBUL"),
], ids=["sharp-s", "ligature", "nfc-nfd", "fullwidth", "dotted-i"])
def test_tokenize_folds_case_and_unicode_forms(left, right):
    assert left != right
    assert tokenize(left) == tokenize(right)
    assert len(tokenize(left)) == 1


def test_tokenize_keeps_combining_marks_inside_words():
    assert tokenize("हिन्दी भाषा") == ["हिन्दी", "भाषा"]
    assert tokenize("नदी") == ["नदी"] and tokenize("दिन") == ["दिन"]
    # Decomposed accents stay inside the word, which comes out composed.
    assert tokenize(RESUME_NFD + " and coffee") == [RESUME_NFC.lower(), "and", "coffee"]


def test_common_question_words_are_stopwords():
    assert {"the", "for", "what", "is"} <= STOPWORDS


class HubError(Exception):
    """An unrelated exception type, like the ones model libraries raise."""


def test_model_load_failure_raises_embedding_unavailable(monkeypatch):
    class Failing:
        def __init__(self, name, trust_remote_code):
            raise HubError("Repository not found: " + name)

    fake_sentence_transformers(monkeypatch, Failing)
    with pytest.raises(EmbeddingUnavailableError, match="Could not load the embedding model"):
        SentenceTransformerEmbedder("org/missing-model").embed(["text"])


def test_encode_failure_raises_embedding_unavailable(monkeypatch):
    class Broken:
        def __init__(self, name, trust_remote_code):
            pass

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            raise HubError("CUDA out of memory")

    fake_sentence_transformers(monkeypatch, Broken)
    with pytest.raises(EmbeddingUnavailableError, match="The embedding model failed"):
        SentenceTransformerEmbedder("local-model").embed(["text"])


def test_model_loads_once_even_with_concurrent_first_use(monkeypatch):
    loads = []

    class Slow:
        def __init__(self, name, trust_remote_code):
            loads.append(name)
            time.sleep(0.05)

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            return [[1.0, 0.0] for _ in texts]

    fake_sentence_transformers(monkeypatch, Slow)
    embedder = SentenceTransformerEmbedder("local-model")
    start = threading.Barrier(4)
    errors = []

    def use():
        start.wait()
        try:
            embedder.embed(["text"])
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=use) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    embedder.embed(["again"])
    assert errors == [] and loads == ["local-model"]


class BrokenInstallFinder:
    """An import hook failing the way a broken torch install does (dlopen errors)."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "sentence_transformers":
            raise OSError("dlopen(libtorch_cpu.dylib): Library not loaded")
        return None


class BrokenModule(types.ModuleType):
    """A half-initialized package whose attributes fail with OSError."""

    def __getattr__(self, name):
        raise OSError("dlopen(libtorch_cpu.dylib): Library not loaded")


def broken_sentence_transformers(monkeypatch, how):
    """Make ``import sentence_transformers`` fail with an OSError."""
    if how == "finder":
        monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
        monkeypatch.setattr(sys, "meta_path", [BrokenInstallFinder(), *sys.meta_path])
    else:
        monkeypatch.setitem(sys.modules, "sentence_transformers",
                            BrokenModule("sentence_transformers"))


@pytest.mark.parametrize("how", ["finder", "module"])
def test_any_import_failure_is_embedding_unavailable(monkeypatch, how):
    broken_sentence_transformers(monkeypatch, how)
    with pytest.raises(EmbeddingUnavailableError,
                       match=r"Could not import sentence-transformers \(OSError: dlopen"):
        SentenceTransformerEmbedder("local-model").embed(["text"])


@pytest.mark.parametrize("cached", [True, False])
def test_model_is_first_loaded_from_the_local_cache_only(monkeypatch, cached):
    attempts = []

    class Recording:
        def __init__(self, name, **options):
            attempts.append((name, options))
            if options.get("local_files_only") and not cached:
                raise OSError("not in the local cache and outgoing traffic is disabled")

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            return [[1.0, 0.0] for _ in texts]

    fake_sentence_transformers(monkeypatch, Recording)
    embedder = SentenceTransformerEmbedder("org/model")
    assert embedder.embed(["a"]) == [[1.0, 0.0]] and embedder.embed(["b"]) == [[1.0, 0.0]]
    offline = ("org/model", {"trust_remote_code": False, "local_files_only": True})
    online = ("org/model", {"trust_remote_code": False})
    assert attempts == ([offline] if cached else [offline, online])


def test_one_lock_covers_loading_and_encoding(monkeypatch):
    active, overlap = [], []

    class Unsafe:
        def __init__(self, name, **options):
            pass

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            active.append(1)
            overlap.append(len(active))
            time.sleep(0.01)
            active.pop()
            return [[1.0, 0.0] for _ in texts]

    fake_sentence_transformers(monkeypatch, Unsafe)
    embedder = SentenceTransformerEmbedder("local-model")
    start = threading.Barrier(4)

    def use():
        start.wait()
        for _ in range(3):
            embedder.embed(["text"])

    threads = [threading.Thread(target=use) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(overlap) == 12 and max(overlap) == 1


def test_model_path_with_tilde_is_expanded(monkeypatch, tmp_path):
    loaded = []

    class Recording:
        def __init__(self, name, trust_remote_code):
            loaded.append(name)

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            return [[1.0, 0.0] for _ in texts]

    fake_sentence_transformers(monkeypatch, Recording)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MEMORYOS_EMBEDDING_MODEL", "~/models/minilm")
    embedder = create_embedder()
    embedder.embed(["text"])
    assert loaded == [str(tmp_path / "models" / "minilm")]


# retrieval.py: rank fusion


def test_competition_ranks_share_a_rank_for_tied_scores():
    ranking = [("a", 1.0), ("b", 1.0), ("c", 0.5), ("d", 0.5), ("e", 0.25)]
    assert [rank for _, _, rank in retrieval_module._competition_ranks(ranking)] == [
        1, 1, 3, 3, 5]


def test_tied_graph_seeds_do_not_let_fact_ids_decide_the_top_result(store, monkeypatch):
    # The asked-for fact gets the largest ID. Ranking tied graph scores by
    # position (1, 2, 3, ...) would then put it last in the graph channel and
    # let an unrelated attribute win the fused score.
    ids = iter("{:032x}".format(number) for number in (0xF0, 1, 2, 3, 4))
    with monkeypatch.context() as patch:  # Only these writes get the chosen IDs.
        patch.setattr(store_module.uuid, "uuid4", lambda: types.SimpleNamespace(hex=next(ids)))
        employer = store.add("Alice", "employer", "Acme", valid_from="2025-01-01")
        for attribute, value in (("hobby", "chess"), ("city", "Riverton"), ("drink", "tea"),
                                 ("pet", "cat")):
            store.add("Alice", attribute, value, valid_from="2025-01-01")
    assert employer.id == "{:032x}".format(0xF0)
    hits = HybridRetriever(store, HashingEmbedder()).search("Alice employer",
                                                            as_of="2025-02-01", limit=10)
    assert hits[0].fact.id == employer.id
    assert all(hit.channels["graph"] == 1.0 for hit in hits)
    # Facts tied in every channel share one fused score.
    hobby, drink = (next(hit for hit in hits if hit.fact.attribute == name)
                    for name in ("hobby", "drink"))
    assert hobby.channels == drink.channels and hobby.score == drink.score


# retrieval.py: document vectors


def edit_value(store, fact_id, old, new):
    path = store.path_for(fact_id)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"value": "{}"'.format(old), '"value": "{}"'.format(new)),
                    encoding="utf-8")


def test_vector_cache_follows_markdown_edits_on_a_long_lived_retriever(store):
    alice = store.add("Alice", "employer", "Google", valid_from="2025-01-01")
    bob = store.add("Bob", "employer", "Microsoft", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    assert [hit.fact.id for hit in retriever.search("Google", mode="semantic")] == [alice.id]
    before = {hit.fact.id: hit.channels for hit in retriever.search("Alice Microsoft",
                                                                     mode="semantic")}
    edit_value(store, alice.id, "Google", "Microsoft")
    assert store.refresh() is True
    assert retriever.search("Google", mode="semantic") == []
    after = {hit.fact.id: hit.channels for hit in retriever.search("Alice Microsoft",
                                                                    mode="semantic")}
    assert after[alice.id] != before[alice.id]
    fresh = HybridRetriever(store, HashingEmbedder()).search("Alice Microsoft", mode="semantic")
    assert after == {hit.fact.id: hit.channels for hit in fresh}
    assert set(after) == {alice.id, bob.id}


def test_vector_cache_can_be_shared_between_retrievers(store):
    for name in ("alpha", "beta", "gamma"):
        store.add(name, "drink", "tea", valid_from="2025-01-01")
    cache = VectorCache()
    first, second = CountingHashing(), CountingHashing()
    one = HybridRetriever(store, first, vector_cache=cache)
    assert one.vectors is cache
    expected = one.search("tea", mode="semantic")
    assert len(first.calls[0]) == 4 and len(cache) == 3
    hits = HybridRetriever(store, second, vector_cache=cache).search("tea", mode="semantic")
    assert second.calls == [["tea"]]  # Only the query; documents came from the cache.
    assert [(hit.fact.id, hit.score) for hit in hits] == [
        (hit.fact.id, hit.score) for hit in expected]


def test_swapping_the_embedder_does_not_reuse_vectors(store):
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    first = Controlled("model-a", document=(1.0, 0.0))
    retriever = HybridRetriever(store, first)
    assert retriever.search("tea", mode="semantic")[0].channels["semantic"] == 1.0
    second = Controlled("model-b", document=(0.6, 0.8))
    retriever.embedder = second
    [hit] = retriever.search("tea", mode="semantic")
    assert hit.channels["semantic"] == pytest.approx(0.6)
    assert len(second.calls[0]) == 2  # The document was embedded by the new model.


def test_hashing_embedders_of_different_sizes_do_not_share_vectors(store):
    assert HashingEmbedder().name == "lexical-hash-v1-384"
    assert HashingEmbedder(64).name == "lexical-hash-v1-64"
    for name in ("alpha", "beta"):
        store.add(name, "drink", "tea", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    assert len(retriever.search("tea", mode="semantic")) == 2
    retriever.embedder = HashingEmbedder(64)
    swapped = retriever.search("alpha tea", mode="semantic")
    fresh = HybridRetriever(store, HashingEmbedder(64)).search("alpha tea", mode="semantic")
    assert [(hit.fact.id, hit.score) for hit in swapped] == [
        (hit.fact.id, hit.score) for hit in fresh]
    assert len(swapped) == 2 and len(retriever.vectors) == 4  # Two documents per size.


def test_a_batch_larger_than_the_cache_limit_keeps_its_last_entries(store):
    for name in ("alpha", "beta", "gamma"):
        store.add(name, "drink", "tea", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    retriever.cache_limit = 2
    assert len(retriever.search("tea", mode="semantic")) == 3
    assert len(retriever.vectors) == retriever.cache_limit
    assert len(retriever.search("tea", mode="semantic")) == 3
    assert len(retriever.vectors) == retriever.cache_limit
    cache = VectorCache()
    entry = (2, None, array("d", [1.0, 0.0]), 1.0, frozenset({"tea"}))
    cache.store({("m", "a"): entry, ("m", "b"): entry, ("m", "c"): entry}, limit=2)
    assert len(cache) == 2
    assert cache.get(("m", "a")) is None
    assert cache.get(("m", "b")) is not None and cache.get(("m", "c")) is not None


def test_storing_nothing_new_leaves_a_shared_cache_alone(store):
    for name in ("alpha", "beta", "gamma"):
        store.add(name, "drink", "tea", valid_from="2025-01-01")
    cache = VectorCache()
    HybridRetriever(store, HashingEmbedder(), vector_cache=cache).search("tea", mode="semantic")
    assert len(cache) == 3
    small = HybridRetriever(store, HashingEmbedder(), vector_cache=cache)
    small.cache_limit = 2
    assert len(small.search("tea", mode="semantic")) == 3  # Every document was a hit.
    assert len(cache) == 3
    cache.store({}, limit=0)
    assert len(cache) == 3


def test_vector_cache_entries_are_compact_arrays(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    text = HybridRetriever._text(fact)
    lexical = HybridRetriever(store, HashingEmbedder())
    lexical.search("tea", mode="semantic")
    dimensions, indices, values, norm, tokens = lexical.vectors.get(("lexical-hash-v1-384", text))
    # Hashing vectors are mostly zeros: only the non-zero positions are kept.
    assert dimensions == 384 and (indices.typecode, values.typecode) == ("i", "d")
    assert 0 < len(indices) == len(values) < 384 // 2
    assert norm == pytest.approx(1.0) and "tea" in tokens
    learned = HybridRetriever(store, Controlled("dense", document=(0.6, 0.8)))
    learned.search("tea", mode="semantic")
    dimensions, indices, values, norm, _ = learned.vectors.get(("dense", text))
    assert (dimensions, indices, values.typecode, list(values)) == (2, None, "d", [0.6, 0.8])
    assert norm == pytest.approx(1.0)
    assert HybridRetriever.cache_limit == 50_000


# retrieval.py: thresholds, terms and queries


def test_semantic_threshold_comes_from_the_environment(store, monkeypatch):
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    provider = Controlled("controlled", document=(0.3, 0.9539392014169457))
    assert len(HybridRetriever(store, provider).search("q", mode="semantic")) == 1
    monkeypatch.setenv("MEMORYOS_SEMANTIC_THRESHOLD", "0.5")
    retriever = HybridRetriever(store, provider)
    assert retriever.semantic_threshold == 0.5
    assert retriever.search("q", mode="semantic") == []
    explicit = HybridRetriever(store, provider, semantic_threshold=0.1)
    assert len(explicit.search("q", mode="semantic")) == 1
    monkeypatch.setenv("MEMORYOS_SEMANTIC_THRESHOLD", "  ")
    assert HybridRetriever(store, provider).semantic_threshold == 0.25


@pytest.mark.parametrize("configured", ["high", "1.5", "nan"])
def test_invalid_semantic_threshold_from_the_environment_is_rejected(store, monkeypatch,
                                                                     configured):
    monkeypatch.setenv("MEMORYOS_SEMANTIC_THRESHOLD", configured)
    with pytest.raises(ValueError, match="MEMORYOS_SEMANTIC_THRESHOLD|semantic_threshold"):
        HybridRetriever(store, HashingEmbedder())


def test_boolean_semantic_threshold_is_rejected(store):
    for flag in (True, False):
        with pytest.raises(ValueError, match="semantic_threshold"):
            HybridRetriever(store, HashingEmbedder(), semantic_threshold=flag)


def test_semantic_threshold_accepts_any_real_number(store):
    retriever = HybridRetriever(store, HashingEmbedder(), semantic_threshold=Fraction(1, 4))
    assert retriever.semantic_threshold == 0.25 and type(retriever.semantic_threshold) is float
    for rejected in (Decimal("0.25"), "0.25", True):  # Decimal is not numbers.Real.
        with pytest.raises(ValueError, match="semantic_threshold"):
            HybridRetriever(store, HashingEmbedder(), semantic_threshold=rejected)


@pytest.mark.parametrize("configured, expected", [
    ("", 0.25), (" 0.4 ", 0.4), ("-1", -1.0), ("1", 1.0), ("2", None), ("nan", None),
    ("high", None), ("-1.5", None),
])
def test_default_threshold_validates_the_environment(monkeypatch, configured, expected):
    monkeypatch.setenv("MEMORYOS_SEMANTIC_THRESHOLD", configured)
    if expected is None:
        with pytest.raises(ValueError, match="MEMORYOS_SEMANTIC_THRESHOLD must be a number "
                                             "between -1 and 1"):
            retrieval_module.default_threshold()
    else:
        assert retrieval_module.default_threshold() == expected


def test_huge_embedding_values_are_skipped_instead_of_scoring_one(store):
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    huge = Controlled("huge", document=(1e200, 1e200))  # Finite, but the norm overflows.
    retriever = HybridRetriever(store, huge, semantic_threshold=0.9)
    assert retriever.search("zebra", mode="semantic") == []


def test_keyword_channel_matches_documents_as_written(store):
    # The alias table normalizes lexical queries. Applied to records in the
    # keyword channel it made "Applied Mathematics" an application.
    degree = store.add("Alice", "degree", "Applied Mathematics", valid_from="2025-01-01")
    job = store.add("Bob", "job application", "submitted", valid_from="2025-01-01")
    invited = store.add("Acme application", "status", "interview invited",
                        valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())

    def keyword(query):
        return [hit.fact.id for hit in retriever.search(query, mode="keyword", limit=10)]

    assert sorted(keyword("application status")) == sorted([invited.id, job.id])
    assert keyword("invitation") == [] and keyword("applied") == [degree.id]
    assert keyword("invited") == [invited.id]
    # The lexical channel still treats the listed forms as one word.
    assert [hit.fact.id for hit in retriever.search("invitation", mode="semantic")] == [
        invited.id]


@pytest.mark.parametrize("document, query", [
    ("Stra\u00dfe", "STRASSE"),
    ("Strasse", "Stra\u00dfe"),
    ("\ufb01nance", "finance"),
    ("\u0130stanbul", "\u0130STANBUL"),
    ("\u0130stanbul", "i\u0307stanbul"),
    (RESUME_NFC, RESUME_NFD),
    (RESUME_NFD, RESUME_NFC),
    ("\uff21\uff22\uff23", "abc"),
], ids=["sharp-s", "sharp-s-query", "ligature", "dotted-i", "dotted-i-decomposed",
        "nfd-query", "nfd-record", "fullwidth"])
def test_keyword_and_lexical_channels_match_across_unicode_forms(store, document, query):
    assert document != query
    fact = store.add("person", "street", document, valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    for mode in ("keyword", "semantic"):
        assert [hit.fact.id for hit in retriever.search(query, mode=mode)] == [fact.id], mode


def test_devanagari_words_do_not_match_by_shared_consonants(store):
    river = store.add("नदी", "नाम", "गंगा", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    for mode in ("keyword", "semantic", "graph", "hybrid"):
        assert retriever.search("दिन", mode=mode) == [], mode
        assert [hit.fact.id for hit in retriever.search("नदी", mode=mode)] == [river.id], mode


def test_stopwords_do_not_match_every_demo_fact(store):
    seed_demo(store)
    retriever = HybridRetriever(store, HashingEmbedder())
    # Every demo record's evidence contains "for" and "the".
    for mode in ("keyword", "semantic", "hybrid"):
        hits = retriever.search("What is the preferred drink for Alex?", mode=mode, limit=10)
        assert {hit.fact.entity for hit in hits} == {"Alex"}, mode
        assert hits[0].fact.attribute == "preferred drink", mode
    hits = retriever.search("What drink does Alex enjoy?", mode="keyword", limit=10)
    assert {hit.fact.entity for hit in hits} == {"Alex"}
    assert hits[0].fact.value == "decaf coffee"
    drinks = retriever.search("What is the preferred drink?", mode="keyword", limit=10,
                              as_of="2026-09-01")
    assert [hit.fact.value for hit in drinks] == ["tea"]


def test_stopwords_do_not_use_up_the_keyword_term_limit(store):
    coffee = store.add("Bob", "drink", "coffee", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    fillers = ["filler{}".format(index) for index in range(10)]
    query = " ".join(sorted(STOPWORDS) + fillers + ["coffee"])
    assert len(set(tokenize(query))) > MAX_KEYWORD_TERMS
    assert [hit.fact.id for hit in retriever.search(query, mode="keyword")] == [coffee.id]


def test_keyword_terms_are_capped_at_64(store):
    coffee = store.add("Bob", "drink", "coffee", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    assert MAX_KEYWORD_TERMS == 64

    def query(count):
        return " ".join(["filler{}".format(index) for index in range(count)] + ["coffee"])

    assert [hit.fact.id for hit in retriever.search(query(63), mode="keyword")] == [coffee.id]
    assert retriever.search(query(64), mode="keyword") == []


def test_symbol_only_query_needs_a_learned_provider(store):
    coffee = store.add("person", "drink", "☕", valid_from="2025-01-01")
    learned = HybridRetriever(store, Controlled("controlled", document=(1.0, 0.0)))
    for mode in ("semantic", "hybrid"):
        assert [hit.fact.id for hit in learned.search("☕", mode=mode)] == [coffee.id], mode
    for mode in ("keyword", "graph"):
        assert learned.search("☕", mode=mode) == [], mode
    assert learned.search("   ", mode="semantic") == []
    lexical = HybridRetriever(store, HashingEmbedder())
    for mode in HybridRetriever.available_modes:
        assert lexical.search("☕", mode=mode) == [], mode


def test_citation_falls_back_when_the_record_disappears_after_the_snapshot(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01", topic="profile")
    other = MemoryStore(store.root)

    class Vanishing:
        """Another writer deletes the record right after the snapshot is taken."""

        root = store.root

        def snapshot(self, **arguments):
            facts = store.snapshot(**arguments)
            store.path_for(fact.id).unlink()
            assert other.refresh() is True
            return facts

        def path_for(self, fact_id):
            return store.path_for(fact_id)

    try:
        [hit] = HybridRetriever(Vanishing(), HashingEmbedder()).search("tea")
    finally:
        other.close()
    assert hit.fact.id == fact.id
    assert hit.citation == "memory/profile/{}.md".format(fact.id)


# extraction.py: conflicts and damaged queue files


def test_conflicting_approvals_raise_memory_conflict(store):
    old = store.add("user", "city", "Paris", valid_from="2026-01-01")
    queue = ProposalQueue(store)
    [stale] = queue.extract("user | city | Lyon | 2026-01-01", source="chat")
    store.correct(old.id, "London")
    with pytest.raises(MemoryConflictError, match="Memory changed"):
        queue.approve(stale.id)
    rejected = queue.reject(stale.id)
    with pytest.raises(MemoryConflictError, match="Only pending"):
        queue.approve(rejected.id)
    [fresh] = queue.extract("user | drink | tea | 2026-01-01", source="chat")
    queue.approve(fresh.id)
    with pytest.raises(MemoryConflictError, match="cannot be rejected"):
        queue.reject(fresh.id)
    assert queue.get(fresh.id).status == "approved"


def valid_proposal_file(store):
    queue = ProposalQueue(store)
    [proposal] = queue.extract("Alex | city | Riverton", source="chat",
                               observed_at="2025-01-01")
    path = queue.directory / (proposal.id + ".json")
    return queue, proposal, path, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("damage, message", [
    (lambda data: b"{", "not valid JSON"),
    (lambda data: b"", "not valid JSON"),
    (lambda data: b"\xff\xfe", "not valid JSON"),
    (lambda data: b"[]", "not a proposal file"),
    (lambda data: json.dumps({**data, "reviewed_by": "me"}).encode(), "not a proposal file"),
    (lambda data: json.dumps({k: v for k, v in data.items() if k != "entity"}).encode(),
     "missing proposal fields"),
    (lambda data: json.dumps({**data, "status": "done"}).encode(), "inconsistent"),
    (lambda data: json.dumps({**data, "kind": "merge"}).encode(), "inconsistent"),
    (lambda data: json.dumps({**data, "created_at": None}).encode(), "field type"),
    (lambda data: json.dumps({**data, "created_at": "yesterday"}).encode(), "field type"),
    # Forms some Python versions' fromisoformat accepts, outside the package's grammar.
    (lambda data: json.dumps({**data, "created_at": "20250101T000000"}).encode(), "field type"),
    (lambda data: json.dumps({**data, "created_at": "2025-01-01T08:00:00+0800"}).encode(),
     "field type"),
    (lambda data: json.dumps({**data, "created_at": "2025-01-01T08"}).encode(), "field type"),
    (lambda data: json.dumps({**data, "valid_from": "next week"}).encode(), "field type"),
    (lambda data: json.dumps({**data, "entity": 7}).encode(), "field type"),
    (lambda data: json.dumps({**data, "target_id": ["a"]}).encode(), "field type"),
    (lambda data: json.dumps({**data, "confidence": "high"}).encode(), "field type"),
    (lambda data: json.dumps({**data, "confidence": True}).encode(), "field type"),
    (lambda data: json.dumps({**data, "dated": "no"}).encode(), "field type"),
    (lambda data: json.dumps({**data, "observed": 1}).encode(), "field type"),
    (lambda data: json.dumps({**data, "warnings": "check this"}).encode(), "field type"),
    (lambda data: json.dumps({**data, "warnings": [1]}).encode(), "field type"),
    (lambda data: json.dumps({**data, "kind": "update"}).encode(), "field type"),
    (lambda data: json.dumps({**data, "kind": "correct", "target_id": ""}).encode(),
     "field type"),
    (lambda data: json.dumps({**data, "kind": "duplicate"}).encode(), "field type"),
    (lambda data: json.dumps({**data, "status": "approved"}).encode(), "field type"),
], ids=["truncated", "empty", "not-utf8", "list", "unknown-key", "missing-field",
        "bad-status", "bad-kind", "null-created-at", "bad-created-at", "compact-created-at",
        "compact-offset-created-at", "hour-only-created-at", "bad-valid-from",
        "number-entity", "list-target", "text-confidence", "boolean-confidence", "text-dated",
        "number-observed", "text-warnings", "number-warning", "update-without-target", "correct-without-target",
        "duplicate-without-target", "approved-without-result"])
def test_malformed_proposal_file_is_named_and_skipped_by_list(store, damage, message):
    queue, proposal, path, data = valid_proposal_file(store)
    [other] = queue.extract("Alex | drink | tea", source="chat", observed_at="2025-01-01")
    path.write_bytes(damage(data))
    name = "proposals/{}.json".format(proposal.id)
    # Acting on the damaged proposal itself names its file.
    for action in (lambda: queue.get(proposal.id), lambda: queue.approve(proposal.id),
                   lambda: queue.reject(proposal.id)):
        with pytest.raises(ValueError, match=message) as caught:
            action()
        assert str(caught.value).startswith(name + " ")
        assert str(store.root) not in str(caught.value)
    # Listing skips it and says why, so the rest of the queue stays usable.
    for status in ("pending", "all", None):
        assert [p.id for p in queue.list(status=status)] == [other.id]
        [problem] = queue.problems
        assert problem.startswith(name + " ") and re.search(message, problem)
    assert [p.id for p in queue.list(status="approved")] == [] and len(queue.problems) == 1
    assert queue.approve(other.id).value == "tea"
    [fresh] = queue.extract("Alex | pet | cat", source="chat", observed_at="2025-01-01")
    assert fresh.status == "pending" and len(queue.problems) == 1
    assert [f.value for f in store.all_facts()] == ["tea"]
    path.unlink()
    queue.list()
    assert queue.problems == []


@pytest.mark.parametrize("created_at", [
    "2025-01-01T00:00:00Z", "2025-01-01T00:00:00.5+00:00", "2025-01-01 00:00", "2025-01-01",
])
def test_proposal_creation_times_follow_one_grammar_on_every_python(store, created_at):
    queue, proposal, path, data = valid_proposal_file(store)
    path.write_text(json.dumps({**data, "created_at": created_at}), encoding="utf-8")
    assert queue.get(proposal.id).created_at == created_at
    assert [p.id for p in queue.list()] == [proposal.id] and queue.problems == []
    assert queue.approve(proposal.id).value == "Riverton"


def test_stray_json_files_are_ignored_by_the_queue(store):
    queue, proposal, path, _ = valid_proposal_file(store)
    # Not 32 lowercase hex characters (an uppercase copy of a real name would
    # replace that file on a case-insensitive filesystem, so use another name).
    for name in ("backup.json", "notes.json", "A" * 32 + ".json"):
        (queue.directory / name).write_text("{not even json", encoding="utf-8")
    assert [p.id for p in queue.list()] == [proposal.id]
    [again] = queue.extract("Alex | city | Riverton", source="chat", observed_at="2025-01-01")
    assert again.id == proposal.id


def test_proposal_file_from_before_the_dated_field_still_loads(store):
    queue, proposal, path, data = valid_proposal_file(store)
    assert data.pop("dated") is False  # The line itself had no date.
    path.write_text(json.dumps(data), encoding="utf-8")
    assert queue.get(proposal.id).dated is True
    assert queue.approve(proposal.id).value == "Riverton"


def test_proposal_file_from_before_the_observed_field_still_loads(store):
    """A proposal file written before the observed field loads as not observed."""
    queue, proposal, path, data = valid_proposal_file(store)
    assert data.pop("observed") is True  # The line took an explicit observation time.
    path.write_text(json.dumps(data), encoding="utf-8")
    assert queue.get(proposal.id).observed is False
    assert queue.approve(proposal.id).value == "Riverton"


def test_proposal_file_with_a_different_id_inside_is_rejected(store):
    queue, proposal, path, _ = valid_proposal_file(store)
    copy = "c" * 32
    (queue.directory / (copy + ".json")).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="inconsistent id"):
        queue.get(copy)
    with pytest.raises(ValueError, match="inconsistent id"):
        queue.approve(copy)
    assert store.all_facts() == []
    assert queue.get(proposal.id).status == "pending"


# extraction.py: approval


def test_dated_conversation_approves_in_order_over_an_existing_value(store):
    store.add("Acme application", "status", "pending", valid_from="2026-09-01")
    queue = ProposalQueue(store)
    proposals = queue.extract(
        "2026-09-15: My application at Acme is interview invited.\n"
        "2026-10-01: My application at Acme is interview completed.", source="conversation:1")
    assert [(p.kind, p.prior_value) for p in proposals] == [("update", "pending")] * 2
    for proposal in proposals:
        queue.approve(proposal.id)
    assert [f.value for f in store.history("Acme application")] == [
        "pending", "interview invited", "interview completed"]
    assert {p.status for p in queue.list(status="all")} == {"approved"}


def test_add_preview_over_an_unseen_value_is_a_conflict(store):
    queue = ProposalQueue(store)
    [proposal] = queue.extract("Acme application | status | invited | 2025-01-10",
                               source="chat:1")
    assert proposal.kind == "add"
    store.add("Acme application", "status", "pending", valid_from="2025-01-01",
              source="manual")
    with pytest.raises(MemoryConflictError, match="Memory changed"):
        queue.approve(proposal.id)
    assert [f.value for f in store.all_facts()] == ["pending"]


@pytest.mark.parametrize("setup, line", [
    (lambda store: None, "Acme application | status | pending | 2026-09-01"),
    (lambda store: store.add("user", "city", "Paris", valid_from="2026-01-01"),
     "user | city | Lyon | 2026-01-01"),
], ids=["add", "correct"])
def test_approval_after_an_interrupted_save_marks_the_existing_fact(store, monkeypatch,
                                                                    setup, line):
    setup(store)
    queue = ProposalQueue(store)
    [proposal] = queue.extract(line, source="conv:1")
    before = {f.id for f in store.all_facts()}
    real_save = ProposalQueue._save

    def crash(self, proposal):
        raise RuntimeError("process stopped after the fact was written")

    monkeypatch.setattr(ProposalQueue, "_save", crash)
    with pytest.raises(RuntimeError):
        queue.approve(proposal.id)
    monkeypatch.setattr(ProposalQueue, "_save", real_save)
    [written] = [f for f in store.all_facts() if f.id not in before]
    assert queue.get(proposal.id).status == "pending"
    assert queue.approve(proposal.id).id == written.id
    saved = queue.get(proposal.id)
    assert (saved.status, saved.result_fact_id) == ("approved", written.id)
    assert len(store.all_facts()) == len(before) + 1


@pytest.mark.parametrize("topic", ["general", "other"])
def test_correction_approval_keeps_the_target_topic_and_the_reviewed_confidence(store, topic):
    # The extraction topic is often just the default; a correction does not move
    # the corrected record to another folder.
    old = store.add("user", "preference", "tea", valid_from="2025-01-01", topic="profile/prefs")
    queue = ProposalQueue(store)
    [proposal] = queue.extract("2025-01-01: I prefer coffee.", source="chat", topic=topic)
    assert (proposal.kind, proposal.confidence, proposal.topic) == ("correct", 0.75, topic)
    fact = queue.approve(proposal.id)
    assert (fact.supersedes, fact.confidence, fact.topic) == (old.id, 0.75, "profile/prefs")
    assert store.path_for(fact.id).parent == store.memory_dir / "profile" / "prefs"
    assert not (store.memory_dir / topic).exists()


# extraction.py: duplicates, warnings and parsing


def test_undated_reextraction_without_observed_at_returns_the_pending_proposal(store):
    queue = ProposalQueue(store)
    [first] = queue.extract("My city is Paris.", source="chat")
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id and first.dated is False
    assert len(queue.list()) == 1
    # A dated line names its own effective time, whenever it is extracted.
    [dated] = queue.extract("2025-01-01: My city is Paris.", source="chat",
                            observed_at="2026-01-01")
    assert (dated.dated, dated.valid_from) == (True, "2025-01-01T00:00:00.000000Z")
    [dated_again] = queue.extract("2025-01-01: My city is Paris.", source="chat",
                                  observed_at="2026-06-01")
    assert dated_again.id == dated.id
    assert len(queue.list()) == 2


def test_an_approved_undated_statement_said_again_later_is_a_new_proposal(store):
    queue = ProposalQueue(store)
    [paris] = queue.extract("My city is Paris.", source="chat", observed_at="2025-01-01")
    queue.approve(paris.id)
    queue.approve(queue.extract("My city is London.", source="chat",
                                observed_at="2025-06-01")[0].id)
    # Back in Paris: the same words again are a new statement, not the old approval.
    [again] = queue.extract("My city is Paris.", source="chat", observed_at="2026-01-01")
    assert again.id != paris.id
    assert (again.status, again.kind, again.prior_value) == ("pending", "update", "London")
    assert again.valid_from == "2026-01-01T00:00:00.000000Z"
    queue.approve(again.id)
    assert [f.value for f in store.history("user", "city")] == ["Paris", "London", "Paris"]


def test_duplicate_detection_includes_the_topic(store):
    queue = ProposalQueue(store)
    [first] = queue.extract("Alex | city | Riverton", source="chat", observed_at="2025-01-01",
                            topic="profile")
    [second] = queue.extract("Alex | city | Riverton", source="chat", observed_at="2025-01-01",
                             topic="people")
    assert second.id != first.id and second.topic == "people"


def test_lines_for_the_same_instant_get_a_warning(store):
    queue = ProposalQueue(store)
    proposals = queue.extract("I prefer tea.\nI like jazz.\nMy city is Paris.",
                              source="chat", observed_at="2026-10-01")
    warning = ("Another line in this text gives preference for the same time; only one can be "
               "approved.")
    assert [warning in p.warnings for p in proposals] == [True, True, False]


def test_restoring_a_retracted_assertion_is_previewed_as_a_restore(store):
    old = store.add("user", "city", "Paris", valid_from="2026-01-01")
    store.retract(old.id)
    [proposal] = ProposalQueue(store).extract("user | city | Lyon | 2026-01-01", source="chat")
    assert proposal.kind == "correct" and proposal.prior_value is None
    assert proposal.warnings == ["Restores an assertion that was retracted at this effective time."]


def test_leading_bom_is_stripped_from_extracted_text(store):
    queue = ProposalQueue(store)
    proposals = queue.extract(BOM + "user | city | Paris\nMy drink is tea.", source="chat",
                              observed_at="2026-10-01")
    assert [(p.entity, p.attribute) for p in proposals] == [("user", "city"), ("user", "drink")]
    [prose] = queue.extract(BOM + "My name is Alex.", source="chat", observed_at="2026-10-01")
    assert (prose.entity, prose.attribute, prose.value) == ("user", "name", "Alex")
    assert BOM not in prose.evidence


@pytest.mark.parametrize("text", [BOM, BOM + "\n", BOM + "  \r\n\t"])
def test_text_holding_only_a_bom_is_empty(store, text):
    with pytest.raises(ValueError, match="Conversation text must not be empty"):
        ProposalQueue(store).extract(text, source="chat")


@pytest.mark.parametrize("line", [
    "| Name | Value |",
    "| --- | --- |",
    "I think a | b is weird",
    "2026-13-45: unsupported prose here",
    "2026-02-30: Alex | city | Lakeside",
    "Alex | city | Lakeside | next week",
])
def test_prose_with_pipes_or_a_bad_date_is_skipped(store, line):
    proposals = ProposalQueue(store).extract(line + "\nAlex | city | Riverton", source="chat",
                                             observed_at="2025-01-01")
    assert [(p.entity, p.attribute, p.value) for p in proposals] == [
        ("Alex", "city", "Riverton")]


@pytest.mark.parametrize("line", [
    "Alex | city | Lakeside | \u0662\u0660\u0662\u0665-\u0660\u0661-\u0660\u0661",
    "Alex | city | Lakeside | \uff12\uff10\uff12\uff15-01-01",
    "\u0662\u0660\u0662\u0665-\u0660\u0661-\u0660\u0661: My city is Lakeside.",
    # The prefix must still be recognized as a (bad) date, or it leaks into the entity.
    "\u0662\u0660\u0662\u0665-\u0660\u0661-\u0660\u0661: Alex | city | Paris",
], ids=["arabic-indic-field", "fullwidth-field", "arabic-indic-prefix", "arabic-indic-pipe"])
def test_dates_written_with_other_digits_are_not_dates(store, line):
    proposals = ProposalQueue(store).extract(line + "\nAlex | city | Riverton", source="chat",
                                             observed_at="2025-01-01")
    assert [(p.entity, p.value, p.valid_from) for p in proposals] == [
        ("Alex", "Riverton", "2025-01-01T00:00:00.000000Z")]


def test_topic_is_validated_even_when_no_line_is_supported(store):
    queue = ProposalQueue(store)
    with pytest.raises(ValueError, match="topic"):
        queue.extract("Only unsupported prose here.", source="chat", topic="../outside")
    assert queue.list(status="all") == []


def test_proposal_saves_use_hidden_temp_files_and_flush_the_folder(store, monkeypatch):
    real_mkstemp = extraction_module.tempfile.mkstemp
    temporary, flushed = [], []

    def recording_mkstemp(**arguments):
        temporary.append(arguments)
        return real_mkstemp(**arguments)

    monkeypatch.setattr(extraction_module.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(extraction_module, "_flush_directory", lambda path: flushed.append(path))
    queue = ProposalQueue(store)
    queue.extract("Alex | city | Riverton", source="chat", observed_at="2025-01-01")
    assert [(a["prefix"], Path(a["dir"])) for a in temporary] == [
        (".proposal-", queue.directory)]
    assert flushed == [queue.directory]
    assert sorted(p.name for p in queue.directory.iterdir() if p.name != ".lock") == [
        queue.list()[0].id + ".json"]


# demo.py


def test_demo_skips_slots_already_used_by_other_assertions(store):
    manual = store.add("Alex", "preferred drink", "water", valid_from="2026-08-01")
    result = seed_demo(store)
    assert result["added"] == 6
    assert result["skipped_existing"] == ["Alex / preferred drink at 2026-08-01"]
    assert store.get(manual.id).value == "water"
    again = seed_demo(store)
    assert again["added"] == 0
    assert again["skipped_existing"] == ["Alex / preferred drink at 2026-08-01"]
    assert sum(f.source == DEMO_SOURCE for f in store.all_facts()) == 6


# Optional interfaces: install hints and the MCP server


def test_missing_extras_name_the_source_checkout_install(monkeypatch, tmp_path):
    from temporal_memory_os.api import create_app
    from temporal_memory_os.mcp_server import create_server

    monkeypatch.setitem(sys.modules, "fastapi", None)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[api\]'"):
        create_app(tmp_path)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[mcp\]'"):
        create_server(tmp_path)


def run_mcp(server, scenario):
    """Run ``scenario(session, initialize_result)`` against an in-memory MCP session."""
    import anyio
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    low_level = server._mcp_server

    async def main():
        async with create_client_server_memory_streams() as (client, served):
            async with anyio.create_task_group() as group:
                group.start_soon(lambda: low_level.run(
                    served[0], served[1], low_level.create_initialization_options()))
                async with ClientSession(client[0], client[1]) as session:
                    result = await session.initialize()
                    await scenario(session, result)
                group.cancel_scope.cancel()

    anyio.run(main)


def test_mcp_reports_the_package_version_and_enumerates_modes(tmp_path):
    pytest.importorskip("mcp")
    from temporal_memory_os.mcp_server import create_server

    async def scenario(session, initialized):
        assert initialized.serverInfo.version == __version__
        [search] = [tool for tool in (await session.list_tools()).tools if tool.name == "search"]
        assert search.inputSchema["properties"]["mode"]["enum"] == [
            "keyword", "semantic", "graph", "hybrid"]
        bad = await session.call_tool("search", {"query": "tea", "mode": "bogus"})
        assert bad.isError
        assert "literal_error" in bad.content[0].text  # Rejected by schema validation.

    run_mcp(create_server(tmp_path), scenario)


def test_mcp_reuses_one_embedder_and_vector_cache(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from temporal_memory_os.mcp_server import create_server

    with MemoryStore(tmp_path) as memory:
        for name in ("alpha", "beta"):
            memory.add(name, "drink", "tea", valid_from="2025-01-01")
    created = []

    def counting_factory(model_name=None):
        created.append(CountingHashing())
        return created[-1]

    monkeypatch.setattr(embeddings_module, "create_embedder", counting_factory)

    async def scenario(session, _initialized):
        for _ in range(2):
            result = await session.call_tool("search", {"query": "tea", "mode": "semantic"})
            assert not result.isError
            assert len(json.loads(result.content[0].text)["results"]) == 2

    run_mcp(create_server(tmp_path), scenario)
    [embedder] = created
    assert [len(call) for call in embedder.calls] == [3, 1]


def test_mcp_empty_memoryos_home_uses_the_default_folder(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from temporal_memory_os.mcp_server import create_server

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYOS_HOME", "")

    async def scenario(session, _initialized):
        result = await session.call_tool("memory_history", {"entity": "Alex"})
        assert not result.isError

    run_mcp(create_server(), scenario)
    assert (tmp_path / "memory-data" / "memory").is_dir()
    assert not (tmp_path / "memory").exists()



# Round 3 follow-ups found by independent review.

def test_opening_an_unchanged_store_does_not_wait_for_a_writer(tmp_path, monkeypatch):
    with MemoryStore(tmp_path) as memory:
        fact = memory.add("person", "drink", "tea", valid_from="2025-01-01")
    holder = sqlite3.connect(tmp_path / ".memoryos" / "index.sqlite3", isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")  # What a running checkpoint holds.
    monkeypatch.setattr(store_module, "_LOCK_RETRY_SECONDS", 0.5)
    try:
        started = time.monotonic()
        with MemoryStore(tmp_path) as reader:  # Files written moments ago are still hashed.
            assert reader.get(fact.id).value == "tea"
            assert HybridRetriever(reader, HashingEmbedder()).search("person tea")
        assert time.monotonic() - started < 0.4
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def test_a_duplicated_record_id_names_both_files(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    original = store.path_for(fact.id)
    copy = original.parent / "0-copy.md"
    copy.write_text(original.read_text(encoding="utf-8").replace('"tea"', '"coffee"'),
                    encoding="utf-8")
    with pytest.raises(StoreIntegrityError) as caught:
        store.reindex()
    message = str(caught.value)
    assert fact.id in message
    assert "memory/general/0-copy.md" in message
    assert "memory/general/{}.md".format(fact.id) in message


def test_a_deleted_record_the_reviewer_saw_makes_approval_stale_not_missing(store):
    queue = ProposalQueue(store)
    london = store.add("person", "city", "London", valid_from="2026-01-01", source="chat")
    [paris] = queue.extract("person | city | Paris | 2026-03-01", source="chat")
    assert paris.kind == "update" and paris.target_id == london.id
    store.add("person", "city", "Oslo", valid_from="2026-02-01", source="chat")
    store.path_for(london.id).unlink()
    store.refresh()
    with pytest.raises(MemoryConflictError, match="Memory changed"):
        queue.approve(paris.id)


def test_audit_trail_stays_fast_on_a_long_correction_chain(store):
    head = store.add("person", "drink", "v0", valid_from="2025-01-01",
                     recorded_at="2025-01-01")
    for index in range(1, 401):
        head = store.correct(head.id, "v{}".format(index), recorded_at="2025-01-01")
    started = time.monotonic()
    trail = store.audit_trail("person", "drink")
    assert time.monotonic() - started < 0.5
    assert [fact.value for fact in trail] == ["v{}".format(index) for index in range(401)]


def test_symbol_names_seed_only_as_whole_words_and_written_names_all_seed(store):
    plus = store.add("+", "kind", "operator", valid_from="2025-01-01")
    c = store.add("C", "kind", "language", valid_from="2025-01-01")
    cpp = store.add("C++", "kind", "language", valid_from="2025-01-01")
    csharp = store.add("C#", "kind", "language", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())

    def seeds(query):
        return {hit.fact.id for hit in retriever.search(query, mode="graph", limit=10)}

    assert seeds("tell me about c++") == {cpp.id}  # Not "+" and not "C".
    assert seeds("C and C++?") == {c.id, cpp.id}
    assert seeds("compare C# and C++") == {cpp.id, csharp.id}
    assert seeds("what does + do") == {plus.id}


def test_a_future_import_does_not_break_ordering_of_new_writes(store, monkeypatch):
    store.add("person", "city", "Future", valid_from="2025-01-01",
              recorded_at="2099-01-01T00:00:00Z")
    tick = "2026-06-01T12:00:00.000000Z"
    real = store_module.normalize_time
    monkeypatch.setattr(store_module, "normalize_time",
                        lambda value=None: real(tick if value is None else value))
    first = store.add("person", "drink", "tea", valid_from="2025-01-01")
    second = store.add("person", "food", "rice", valid_from="2025-01-01")
    assert first.recorded_at == tick
    assert second.recorded_at == "2026-06-01T12:00:00.000001Z"


def test_crash_recovery_never_claims_a_correction_for_an_add(store):
    queue = ProposalQueue(store)
    [add] = queue.extract("2025-01-01: I prefer tea.", source="chat")
    assert add.kind == "add"
    coffee = store.add("user", "preference", "coffee", valid_from="2025-01-01")
    # A correction with the same value, date, source, evidence and confidence,
    # as a correction approval interrupted before saving would leave behind.
    store.correct(coffee.id, "tea", source="chat", evidence="2025-01-01: I prefer tea.",
                  confidence=add.confidence)
    with pytest.raises(MemoryConflictError):
        queue.approve(add.id)


def test_her_is_a_stopword_unless_an_entity_uses_it(store):
    assert "her" in STOPWORDS
    store.add("Alex", "drink", "tea", valid_from="2025-01-01", evidence="her favourite")
    assert HybridRetriever(store, HashingEmbedder()).search("her", mode="keyword") == []


def test_an_unchanged_open_marks_recent_files_verified_when_the_lock_is_free(tmp_path,
                                                                            monkeypatch):
    with MemoryStore(tmp_path) as memory:
        memory.add("person", "drink", "tea", valid_from="2025-01-01")
    index = tmp_path / ".memoryos" / "index.sqlite3"

    def verified():
        with closing(sqlite3.connect(index)) as connection:
            return connection.execute("SELECT verified_ns FROM sources").fetchone()[0]

    before = verified()
    monkeypatch.setattr(store_module, "_RACY_WINDOW_NS", 10 ** 18)  # Everything is recent.
    MemoryStore(tmp_path).close()
    assert verified() > before  # Marked, so later opens need not hash again.
    holder = sqlite3.connect(index, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        marked = verified()
        started = time.monotonic()
        MemoryStore(tmp_path).close()  # Lock busy: skip marking, do not wait.
        assert time.monotonic() - started < 1
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert verified() == marked


# Round 4: found while reviewing the code that mutation testing showed untested.

def test_a_damaged_record_after_a_byte_order_mark_is_reported_not_skipped(store):
    # The accented letter as one Latin-1 byte, which is not valid UTF-8.
    record = hand_written_record(value="tea").replace('"tea"', '"t\u00e9a"').encode("latin-1")
    assert b'"t\xe9a"' in record
    (store.memory_dir / "hand.md").write_bytes(b"\xef\xbb\xbf" + record)
    with pytest.raises(StoreIntegrityError, match="memory/hand.md: record is not valid UTF-8"):
        store.refresh()
    (store.memory_dir / "hand.md").write_bytes(b"\xef\xbb\xbf" + hand_written_record().encode())
    assert store.refresh() and [fact.value for fact in store.all_facts()] == ["tea"]


def test_a_link_whose_placeholder_signature_was_stored_is_still_rejected(tmp_path, monkeypatch):
    outside = tmp_path / "outside.md"
    outside.write_text("linked note\n", encoding="utf-8")
    with MemoryStore(tmp_path / "store") as memory:
        memory.add("person", "drink", "tea", valid_from="2025-01-01")
        link = memory.memory_dir / "linked.md"
        link.symlink_to(outside)
        real_scan = MemoryStore._scan

        def scan_then_replace(self):
            found = real_scan(self)
            if link.is_symlink():  # The link becomes a plain file between stat and read.
                link.unlink()
                link.write_text("plain\n", encoding="utf-8")
            return found

        with monkeypatch.context() as patch:
            patch.setattr(MemoryStore, "_scan", scan_then_replace)
            assert memory.reindex() == 1  # Stores the link's placeholder signature.
        link.unlink()
        link.symlink_to(outside)  # A link again before the next refresh.
        with pytest.raises(StoreIntegrityError, match="symbolic links"):
            memory.refresh()
    with pytest.raises(StoreIntegrityError, match="symbolic links"):
        MemoryStore(tmp_path / "store")


def test_extraction_rejects_a_source_that_is_not_valid_unicode(store):
    queue = ProposalQueue(store)
    with pytest.raises(ValueError, match="source"):
        queue.extract("Alex | city | Riverton", source="chat\udcff")
    with pytest.raises(ValueError, match="source"):
        store.add("Alex", "city", "Riverton", source="chat\udcff")  # The same rule.
    assert queue.list() == []


def test_an_empty_observation_time_folds_like_a_missing_one(store, ticking_clock):
    queue = ProposalQueue(store)
    [first] = queue.extract("My city is Paris.", source="chat", observed_at="")
    [again] = queue.extract("My city is Paris.", source="chat", observed_at="")
    [third] = queue.extract("My city is Paris.", source="chat")
    assert first.id == again.id == third.id
    assert len(queue.list()) == 1


def test_an_undated_proposal_saved_before_the_dated_field_still_folds(store, ticking_clock):
    queue = ProposalQueue(store)
    [first] = queue.extract("My city is Paris.", source="chat")
    path = queue.directory / (first.id + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["dated"]  # As written by 0.2.1.
    path.write_text(json.dumps(data), encoding="utf-8")
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id
    assert len(queue.list()) == 1


@pytest.mark.parametrize("change", ["edit", "correct", "retract"])
def test_a_newer_predecessor_does_not_hide_a_change_to_the_reviewed_record(store, change):
    queue = ProposalQueue(store)
    [first] = queue.extract("user | city | London | 2026-01-01", source="chat")
    london = queue.approve(first.id)
    oslo, paris = queue.extract("user | city | Oslo | 2026-02-01\n"
                                "user | city | Paris | 2026-03-01", source="chat")
    assert paris.kind == "update" and paris.prior_value == "London"
    queue.approve(oslo.id)  # The expected, chronological predecessor.
    if change == "edit":  # By hand, keeping the record's ID.
        path = store.path_for(london.id)
        path.write_text(path.read_text(encoding="utf-8").replace('"London"', '"Berlin"'),
                        encoding="utf-8")
    elif change == "correct":
        store.correct(london.id, "Rome", source="other")
    else:
        store.retract(london.id, source="other")
    with pytest.raises(MemoryConflictError, match="Memory changed"):
        queue.approve(paris.id)


@pytest.mark.parametrize("threshold", [10 ** 400, -10 ** 400, Fraction(10 ** 400, 3)],
                         ids=["huge", "huge-negative", "huge-fraction"])
def test_a_threshold_too_large_for_a_float_is_rejected_as_a_value_error(store, threshold):
    with pytest.raises(ValueError, match="semantic_threshold"):
        HybridRetriever(store, HashingEmbedder(), semantic_threshold=threshold)


def test_names_written_with_trailing_punctuation_seed_the_graph(store):
    dotted = store.add("St. Louis", "weather", "humid", valid_from="2025-01-01")
    plain = store.add("St Louis", "weather", "humid", valid_from="2025-01-01")
    band = store.add("!!!", "genre", "dance punk", valid_from="2025-01-01")
    yahoo = store.add("Yahoo!", "kind", "portal", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())

    def seeds(query):
        return {hit.fact.id for hit in retriever.search(query, mode="graph", limit=10)}

    assert dotted.id in seeds("St. Louis weather")
    assert dotted.id in seeds("weather in St. Louis.")
    assert seeds("St Louis weather") == {plain.id}  # The dotted name is not written out.
    assert seeds("!!! tour dates") == {band.id}
    assert seeds("Did you see !!!?") == {band.id}
    assert seeds("Is Yahoo! still around?") == {yahoo.id}


def test_an_index_from_an_older_release_reads_every_file_again(tmp_path):
    # 0.2.0 and 0.2.1 (index version 2) signed a record behind a byte-order mark as a note.
    with MemoryStore(tmp_path) as memory:
        memory.add("person", "food", "rice", valid_from="2025-01-01")
        path = memory.memory_dir / "bom.md"
        path.write_bytes(b"\xef\xbb\xbf" + hand_written_record().encode())
        status = path.stat()
    connection = index_connection(tmp_path)
    try:
        connection.execute("INSERT INTO sources VALUES (?, ?, ?, ?, ?)", (
            "memory/bom.md", status.st_size, status.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(), time.time_ns()))
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    finally:
        connection.close()
    with MemoryStore(tmp_path) as memory:
        assert sorted(fact.value for fact in memory.all_facts()) == ["rice", "tea"]


@pytest.mark.parametrize("codec, newline", [("utf-16", "\n"), ("utf-16-le", "\n"),
                                            ("utf-16-be", "\n"), ("utf-16", "\r\n")],
                         ids=["bom", "le", "be", "bom-crlf"])
def test_a_record_saved_as_utf16_is_reported_not_skipped(store, codec, newline):
    record = hand_written_record().replace("\n", newline).encode(codec)
    (store.memory_dir / "hand.md").write_bytes(record)
    with pytest.raises(StoreIntegrityError, match="memory/hand.md: record is not valid UTF-8"):
        store.refresh()


def test_a_utf16_note_that_mentions_the_marker_inline_is_still_a_note(store):
    note = ("Records start with " + MARKER + " lines.\n").encode("utf-16")
    (store.memory_dir / "note.md").write_bytes(note)
    assert store.refresh() is True and store.all_facts() == []


def test_a_value_planned_for_later_does_not_stop_undated_lines_folding(store, ticking_clock):
    queue = ProposalQueue(store)
    store.add("user", "city", "Oslo", valid_from="2030-01-01", source="chat")  # A planned move.
    [first] = queue.extract("My city is Paris.", source="chat")
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id
    assert len(queue.list()) == 1


def test_a_proposal_file_holding_an_escaped_lone_surrogate_is_reported(store, capsys):
    queue = ProposalQueue(store)
    bad, good = queue.extract("Alex | city | Riverton\nAlex | drink | tea", source="chat",
                              observed_at="2026-01-01")
    path = queue.directory / (bad.id + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["value"] = "River\ud800ton"
    path.write_text(json.dumps(data), encoding="utf-8")  # Escaped as "\ud800".
    assert [proposal.id for proposal in queue.list()] == [good.id]
    assert len(queue.problems) == 1 and "proposals/{}.json".format(bad.id) in queue.problems[0]
    assert cli.main(["--root", str(store.root), "proposals"]) == 0
    captured = capsys.readouterr()
    assert [proposal["id"] for proposal in json.loads(captured.out)] == [good.id]
    assert bad.id in captured.err


def test_extraction_rejects_text_that_is_not_valid_unicode(store, capsys):
    queue = ProposalQueue(store)
    # "\udce9" is how a POSIX command-line argument decodes the Latin-1 byte 0xE9.
    with pytest.raises(ValueError, match="Conversation text"):
        queue.extract("Alex | city | Riv\udce9rton", source="chat")
    assert queue.list() == []
    assert cli.main(["--root", str(store.root), "extract", "--text", "Alex | city | Riv\udce9rton",
                     "--source", "chat"]) == 1
    assert "Conversation text" in json.loads(capsys.readouterr().err)["error"]


def test_a_blank_observation_time_means_now_and_other_falsy_values_are_rejected(
        store, ticking_clock):
    queue = ProposalQueue(store)
    [first] = queue.extract("My city is Paris.", source="chat", observed_at="  ")
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id
    for value in (0, False, []):
        with pytest.raises(ValueError):
            queue.extract("My city is Paris.", source="chat", observed_at=value)
    assert len(queue.list()) == 1


def _graph_seeds(retriever, query):
    return {hit.fact.entity for hit in retriever.search(query, mode="graph", limit=20)}


def test_names_in_quotes_brackets_or_with_a_possessive_count_as_written_out(store):
    for name in ("C", "C#", "C++", "!!!"):
        store.add(name, "kind", "language", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    for query in ('what is "C++"?', "templates (C++) only", "C++'s templates", "¿C++?",
                  "“C++”", "[C++]"):
        assert _graph_seeds(retriever, query) == {"C++"}, query
    assert _graph_seeds(retriever, 'I like "C"') == {"C"}
    assert _graph_seeds(retriever, 'Who are "!!!"?') == {"!!!"}
    assert _graph_seeds(retriever, "C++ or C#") == {"C++", "C#"}


def test_a_name_cut_off_by_the_end_of_the_query_is_not_written_out(store):
    store.add("C# Builder", "kind", "tool", valid_from="2025-01-01")
    store.add("C++ Builder", "kind", "tool", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    # Neither name is written out, so both seed.
    assert _graph_seeds(retriever, "builder for C++") == {"C# Builder", "C++ Builder"}
    assert _graph_seeds(retriever, "is C++ Builder free?") == {"C++ Builder"}


def test_a_run_of_sentence_punctuation_also_writes_shorter_symbol_names(store):
    for name in ("!", "!!", "!!!"):
        store.add(name, "kind", "band", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    # Trailing sentence punctuation is allowed after a name, as in "st." for "st".
    assert _graph_seeds(retriever, "!!! tour dates") == {"!", "!!", "!!!"}
    assert _graph_seeds(retriever, "!! album") == {"!", "!!"}


def test_graph_search_stays_fast_on_a_long_query(store):
    for count in range(1, 101):
        store.add("+" * count, "kind", "operator", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    query = " ".join(["a"] * 5000)[:10_000]  # The longest query the HTTP API accepts.
    started = time.monotonic()
    assert retriever.search(query, mode="graph") == []
    assert time.monotonic() - started < 1.0
    assert _graph_seeds(retriever, "what do ++ and +++. do") == {"++", "+++"}



# Found by the second review of surviving mutants.

@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_opening_never_changes_a_file_that_an_index_log_file_links_to(tmp_path, suffix):
    MemoryStore(tmp_path / "store").close()
    outside = tmp_path / "shared-notes.txt"
    outside.write_text("notes others may read\n", encoding="utf-8")
    outside.chmod(0o644)
    (tmp_path / "store" / ".memoryos" / ("index.sqlite3" + suffix)).symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        MemoryStore(tmp_path / "store")
    assert outside.stat().st_mode & 0o777 == 0o644


def test_a_file_name_that_is_not_utf8_is_reported_by_name(tmp_path):
    with MemoryStore(tmp_path) as memory:
        memory.add("person", "drink", "tea", valid_from="2025-01-01")
    try:
        with open(os.fsencode(tmp_path / "memory") + b"/caf\xe9.md", "wb") as note:
            note.write(b"a plain note\n")
    except OSError:
        pytest.skip("this file system accepts only UTF-8 file names")
    with pytest.raises(StoreIntegrityError, match=r"memory/caf\\udce9\.md: the file name"):
        MemoryStore(tmp_path)


@pytest.mark.parametrize("codec", ["utf-32", "utf-32-le", "utf-32-be"])
def test_a_record_saved_as_utf32_is_reported_not_skipped(store, codec):
    (store.memory_dir / "hand.md").write_bytes(hand_written_record().encode(codec))
    with pytest.raises(StoreIntegrityError, match="memory/hand.md: record is not valid UTF-8"):
        store.refresh()


class _NoOffset(tzinfo):
    """A time zone that gives no offset, which makes a datetime naive."""

    def utcoffset(self, moment):
        return None

    def dst(self, moment):
        return None


def test_a_datetime_whose_zone_gives_no_offset_means_utc(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is Unix-only")
    monkeypatch.setenv("TZ", "HKT-8")  # A local zone that is not UTC.
    time.tzset()
    try:
        assert normalize_time(datetime(2025, 1, 1, 8, 30, tzinfo=_NoOffset())) == (
            "2025-01-01T08:30:00.000000Z")
        assert normalize_time(datetime(1, 1, 1, tzinfo=_NoOffset())) == (
            "0001-01-01T00:00:00.000000Z")
    finally:
        monkeypatch.undo()
        time.tzset()


def test_a_deeply_nested_proposal_file_is_reported_not_raised(store):
    queue = ProposalQueue(store)
    [good] = queue.extract("Alpha | status | one | 2026-01-01", source="chat")
    (queue.directory / ("f" * 32 + ".json")).write_text("[" * 100_000 + "]" * 100_000,
                                                       encoding="utf-8")
    assert [proposal.id for proposal in queue.list()] == [good.id]
    assert len(queue.problems) == 1 and "f" * 32 in queue.problems[0]
    assert queue.approve(good.id).value == "one"


def test_a_deeply_nested_record_is_reported_as_a_store_problem(store):
    record = hand_written_record().replace('"tea"', "[" * 100_000 + "]" * 100_000)
    (store.memory_dir / "hand.md").write_text(record, encoding="utf-8")
    with pytest.raises(StoreIntegrityError, match="memory/hand.md"):
        store.refresh()


def _set_valid_from(queue, proposal_id, value):
    path = queue.directory / (proposal_id + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["valid_from"] = value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_a_hand_written_date_in_a_proposal_file_still_folds_and_recovers(store, monkeypatch):
    queue = ProposalQueue(store)
    line = "user | city | Paris | 2026-01-01"
    [first] = queue.extract(line, source="chat")
    _set_valid_from(queue, first.id, "2026-01-01")  # Another accepted form of the same time.
    [again] = queue.extract(line, source="chat")
    assert again.id == first.id and len(queue.list()) == 1

    def stop(self, proposal):
        raise RuntimeError("stopped after the fact was written")

    with monkeypatch.context() as patch:
        patch.setattr(ProposalQueue, "_save", stop)
        with pytest.raises(RuntimeError):
            queue.approve(first.id)
    [written] = store.all_facts()
    assert queue.approve(first.id).id == written.id


def test_in_order_approval_accepts_a_line_restating_the_conversations_last_value(store):
    queue = ProposalQueue(store)
    january, february = queue.extract("user | city | London | 2026-01-01\n"
                                      "user | city | London | 2026-02-01", source="chat")
    assert (january.kind, february.kind) == ("add", "add")
    first = queue.approve(january.id)
    again = queue.approve(february.id)  # Confirms the value, recorded at its own date.
    assert again.id != first.id and again.valid_from == "2026-02-01T00:00:00.000000Z"
    assert [fact.value for fact in store.history("user", "city")] == ["London", "London"]


def test_in_order_approval_accepts_a_repeated_update(store):
    store.add("user", "city", "London", valid_from="2026-01-01")
    queue = ProposalQueue(store)
    february, march = queue.extract("user | city | Oslo | 2026-02-01\n"
                                    "user | city | Oslo | 2026-03-01", source="chat")
    assert (february.kind, march.kind) == ("update", "update")
    oslo = queue.approve(february.id)
    again = queue.approve(march.id)
    assert again.id != oslo.id and again.valid_from == "2026-03-01T00:00:00.000000Z"
    assert [fact.value for fact in store.history("user", "city")] == ["London", "Oslo", "Oslo"]


def test_in_order_approval_records_a_return_to_an_earlier_value(store):
    store.add("user", "city", "London", valid_from="2026-01-01")
    queue = ProposalQueue(store)
    february, march = queue.extract("user | city | Oslo | 2026-02-01\n"
                                    "user | city | London | 2026-03-01", source="chat")
    assert (february.kind, march.kind) == ("update", "duplicate")
    queue.approve(february.id)
    back = queue.approve(march.id)
    assert (back.value, back.valid_from) == ("London", "2026-03-01T00:00:00.000000Z")
    assert [fact.value for fact in store.history("user", "city")] == ["London", "Oslo", "London"]


def test_a_same_time_restatement_from_the_same_text_is_still_a_sibling_conflict(store):
    queue = ProposalQueue(store)
    tea, again = queue.extract("I prefer tea.\nI like tea.", source="chat",
                               observed_at="2026-10-01")
    queue.approve(tea.id)
    with pytest.raises(MemoryConflictError, match="already set preference"):
        queue.approve(again.id)


def test_finite_vectors_whose_norm_overflows_are_not_compared(store):
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    document = Controlled("huge", [1e154, 1e154], (1.0, 0.0))
    assert HybridRetriever(store, document).search("tea", mode="semantic") == []
    query = Controlled("huge", [1.0, 0.0], (1e154, 1e154))
    with pytest.raises(EmbeddingUnavailableError, match="too large to compare"):
        HybridRetriever(store, query).search("tea", mode="semantic")
