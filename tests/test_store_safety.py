"""Exercise file confinement, index recovery and writers that share no Python locks.

Writers in other processes are started with ``spawn`` and synchronized with a
barrier or events, so the interleaving each test describes really happens on
every run instead of depending on process start-up timing.
"""

import multiprocessing
import os
import queue
import re
import shutil
import signal
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

from temporal_memory_os import store as store_module
from temporal_memory_os.embeddings import HashingEmbedder
from temporal_memory_os.models import Fact
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.store import MemoryConflictError, MemoryStore, StoreIntegrityError

SPAWN = multiprocessing.get_context("spawn")
WAIT = 120  # Seconds; generous for slow CI machines. Passing runs take far less.


def _index(root) -> Path:
    return Path(root) / ".memoryos" / "index.sqlite3"


def _raw(root) -> sqlite3.Connection:
    """A plain SQLite connection to a store's index, outside MemoryStore."""
    return sqlite3.connect(str(_index(root)), isolation_level=None, timeout=WAIT)


def _records(root) -> list:
    return sorted(path.name for path in (Path(root) / "memory").rglob("*.md"))


def _outcome(write) -> str:
    try:
        write()
        return "written"
    except Exception as exc:  # Reported to the parent, which checks the type.
        return type(exc).__name__


def _competing_add(root, barrier, results):
    try:
        with MemoryStore(root) as store:
            barrier.wait(WAIT)  # Every writer enters add() at the same moment.
            results.put(_outcome(lambda: store.add("shared", "status", "active",
                                                   valid_from="2025-01-01")))
    except Exception as exc:
        barrier.abort()  # Release the other writers now instead of at the deadline.
        results.put(type(exc).__name__)


def _opening_writer(root, barrier, results, number):
    try:
        barrier.wait(WAIT)  # Every process creates the brand-new store together.
        with MemoryStore(root) as store:
            results.put(_outcome(lambda: store.add("writer {}".format(number), "status",
                                                   "active", valid_from="2025-01-01")))
    except Exception as exc:
        barrier.abort()  # Release the other writers now instead of at the deadline.
        results.put(type(exc).__name__)


def _waiting_add(root, opened, go, results):
    try:
        with MemoryStore(root) as store:
            opened.set()
            go.wait(WAIT)
            results.put("adding")
            results.put(_outcome(lambda: store.add("shared", "status", "active",
                                                   valid_from="2025-01-01")))
    except Exception as exc:
        results.put(type(exc).__name__)


def _finish(process, timeout: float = WAIT) -> None:
    """Wait for a process and stop it if it overruns, so no test leaves one behind.

    SIGTERM comes first, so an overrun never looks like a deliberate SIGKILL.
    """
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(10)
    if process.is_alive():
        process.kill()
        process.join()


def _collect(results, processes) -> list:
    """One outcome per process within one overall deadline; stragglers are stopped."""
    deadline = time.monotonic() + WAIT
    outcomes = []
    try:
        for _ in processes:
            try:
                outcomes.append(results.get(timeout=max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                outcomes.append("no result (exit codes {})".format(
                    [process.exitcode for process in processes]))
    finally:
        for process in processes:
            _finish(process, max(0.0, deadline - time.monotonic()))
    return outcomes


def test_independent_processes_cannot_create_duplicate_effective_facts(tmp_path):
    # Initialize before launch to keep this focused on the write transaction.
    with MemoryStore(tmp_path):
        pass
    barrier, results = SPAWN.Barrier(3), SPAWN.Queue()
    processes = [SPAWN.Process(target=_competing_add, args=(str(tmp_path), barrier, results))
                 for _ in range(3)]
    for process in processes:
        process.start()
    outcomes = _collect(results, processes)
    assert sorted(outcomes) == ["MemoryConflictError", "MemoryConflictError", "written"]
    assert len(_records(tmp_path)) == 1
    with MemoryStore(tmp_path) as store:
        assert len(store.all_facts()) == 1


def test_a_writer_waits_for_the_write_lock_held_by_another_process(tmp_path):
    with MemoryStore(tmp_path):
        pass
    opened, go, results = SPAWN.Event(), SPAWN.Event(), SPAWN.Queue()
    writer = SPAWN.Process(target=_waiting_add, args=(str(tmp_path), opened, go, results))
    writer.start()
    try:
        assert opened.wait(WAIT)
        with closing(_raw(tmp_path)) as holder:
            holder.execute("BEGIN IMMEDIATE")  # What any other writer's transaction holds.
            go.set()
            assert results.get(timeout=WAIT) == "adding"
            with pytest.raises(queue.Empty):
                results.get(timeout=1.0)
            # Blocked before writing anything, not merely before indexing it.
            assert writer.is_alive()
            assert _records(tmp_path) == []
            holder.execute("COMMIT")
        assert results.get(timeout=WAIT) == "written"
    finally:
        go.set()
        _finish(writer)
    assert writer.exitcode == 0
    with MemoryStore(tmp_path) as store:
        assert [fact.value for fact in store.all_facts()] == ["active"]


def test_processes_creating_a_new_store_together_all_succeed(tmp_path):
    root = tmp_path / "brand-new"
    barrier, results = SPAWN.Barrier(4), SPAWN.Queue()
    processes = [SPAWN.Process(target=_opening_writer, args=(str(root), barrier, results, number))
                 for number in range(4)]
    for process in processes:
        process.start()
    assert _collect(results, processes) == ["written"] * 4
    assert all(process.exitcode == 0 for process in processes)
    with MemoryStore(root) as store:
        assert sorted(fact.entity for fact in store.all_facts()) == [
            "writer {}".format(number) for number in range(4)]
    with closing(_raw(root)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3


@pytest.mark.parametrize("version", [2, 3])
def test_a_full_text_table_left_by_0_2_0_is_kept_and_never_needed(tmp_path, version):
    with MemoryStore(tmp_path) as store:
        first = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01",
                          recorded_at="2025-01-01")
        store.correct(first.id, "Lakeside", recorded_at="2025-01-05")
        store.add("Acme application", "status", "pending", valid_from="2025-01-02")
        before = store.all_facts()
    # 0.2.0 kept a persistent FTS5 table that every write updated (schema version 2).
    # Version 3 indexes can have it too: 0.2.0 processes that were already
    # running keep writing to it after a newer release opened the store.
    with closing(_raw(tmp_path)) as connection:
        connection.executescript("""
            CREATE VIRTUAL TABLE facts_fts USING fts5(
                id UNINDEXED, entity, attribute, value, evidence, source, tags, links
            );
            INSERT INTO facts_fts SELECT id, entity, attribute, value, evidence, source,
                tags, links FROM facts;
        """)
        connection.execute("PRAGMA user_version = {}".format(version))
    with MemoryStore(tmp_path) as store:
        assert store.all_facts() == before
        assert [fact.value for fact in store.snapshot(as_of="2025-03-01")] == [
            "pending", "Lakeside"]
        hits = HybridRetriever(store, embedder=HashingEmbedder()).search(
            "Lakeside", as_of="2025-03-01", mode="keyword")
        assert [hit.fact.value for hit in hits] == ["Lakeside"]
        added = store.add("Alex", "employer", "Acme", valid_from="2025-02-01")
        assert store.reindex() == len(before) + 1
    with closing(_raw(tmp_path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        # Kept, untouched, so a still-running 0.2.0 process can go on writing to it.
        assert connection.execute("SELECT count(*) FROM facts_fts").fetchone()[0] == len(before)
        connection.execute("INSERT INTO facts_fts (id, value) VALUES ('0.2.0 write', 'x')")
    with MemoryStore(tmp_path) as store:
        assert store.all_facts() == before + [added]


def test_an_index_from_a_newer_version_is_refused_with_a_way_out(tmp_path):
    with MemoryStore(tmp_path) as store:
        fact = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01")
    with closing(_raw(tmp_path)) as connection:
        connection.execute("PRAGMA user_version = 4")
    with pytest.raises(ValueError, match=r"Delete the \.memoryos folder") as caught:
        MemoryStore(tmp_path)
    assert str(tmp_path) not in str(caught.value)
    with closing(_raw(tmp_path)) as connection:  # Left exactly as it was.
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    # Following the message recovers everything from Markdown.
    shutil.rmtree(tmp_path / ".memoryos")
    with MemoryStore(tmp_path) as store:
        assert store.all_facts() == [fact]


def _set_aside(root) -> list:
    return sorted(path.name for path in (Path(root) / ".memoryos").glob("*.corrupt*"))


def test_a_corrupt_index_is_set_aside_and_rebuilt_only_on_request(tmp_path):
    with MemoryStore(tmp_path) as store:
        fact = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01")
    garbage = bytes(range(256)) * 16
    _index(tmp_path).write_bytes(garbage)
    for _ in range(2):  # Opening again changes nothing either.
        with pytest.raises(StoreIntegrityError, match="memoryos reindex") as caught:
            MemoryStore(tmp_path)
        assert "file is not a database" in str(caught.value)
        assert str(tmp_path.resolve()) not in str(caught.value)
    # Other processes may have these files open, so only an explicit repair moves them.
    assert _index(tmp_path).read_bytes() == garbage
    assert _set_aside(tmp_path) == []
    with MemoryStore(tmp_path, repair_index=True) as store:  # What `memoryos reindex` does.
        assert store.all_facts() == [fact]
        added = store.add("Alex", "employer", "Acme", valid_from="2025-01-01")
    aside = _set_aside(tmp_path)
    # Named by time, so an earlier damaged file is never overwritten.
    assert len(aside) == 1 and re.fullmatch(r"index\.sqlite3\.corrupt-\d{8}T\d{6}", aside[0])
    # Kept for inspection, never parsed.
    assert (tmp_path / ".memoryos" / aside[0]).read_bytes() == garbage
    assert _index(tmp_path).stat().st_mode & 0o777 == 0o600
    with closing(_raw(tmp_path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    # A healthy index is never set aside, even when repair is allowed.
    with MemoryStore(tmp_path, repair_index=True) as store:
        assert store.all_facts() == [fact, added]
    assert _set_aside(tmp_path) == aside


def _many_records(root) -> list:
    """Written directly and indexed by one rebuild: enough rows for many index pages."""
    facts = [Fact(id="{:032x}".format(number), entity="person {}".format(number),
                  attribute="note", value="value {} ".format(number) * 20,
                  valid_from="2025-01-01", recorded_at="2025-01-01") for number in range(60)]
    general = Path(root) / "memory" / "general"
    general.mkdir(parents=True)
    for fact in facts:
        (general / (fact.id + ".md")).write_text(MemoryStore._render(fact), encoding="utf-8")
    with MemoryStore(root) as store:
        assert store.all_facts() == facts
    return facts


def test_an_index_damaged_past_its_header_is_reported_then_repaired_on_request(tmp_path):
    facts = _many_records(tmp_path)
    data = bytearray(_index(tmp_path).read_bytes())
    page = int.from_bytes(data[16:18], "big") or 65536
    assert len(data) >= 8 * page  # The premise: the index spans many pages.
    data[page:] = b"\xa5" * (len(data) - page)  # Every page after the first, as a crash may.
    _index(tmp_path).write_bytes(bytes(data))
    with closing(_raw(tmp_path)) as connection:  # The header and schema still read fine.
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    with pytest.raises(StoreIntegrityError, match="memoryos reindex") as caught:
        MemoryStore(tmp_path)
    assert "malformed" in str(caught.value)
    assert _set_aside(tmp_path) == []
    with MemoryStore(tmp_path, repair_index=True) as store:
        assert store.all_facts() == facts
    assert [name.split("-")[0] for name in _set_aside(tmp_path)] == ["index.sqlite3.corrupt"]
    with MemoryStore(tmp_path) as store:
        assert store.all_facts() == facts


@pytest.mark.parametrize("damaged", ["facts", "facts_temporal"])
def test_reindex_repairs_damage_in_pages_that_opening_never_reads(tmp_path, damaged):
    """A repair sets aside an index whose record or secondary index pages are damaged."""
    facts = _many_records(tmp_path)
    with closing(_raw(tmp_path)) as connection:
        page = connection.execute("SELECT rootpage FROM sqlite_master WHERE name = ?",
                                  (damaged,)).fetchone()[0]
        size = connection.execute("PRAGMA page_size").fetchone()[0]
    data = bytearray(_index(tmp_path).read_bytes())
    data[(page - 1) * size:page * size] = b"\xa5" * size  # A bad sector or a torn copy.
    _index(tmp_path).write_bytes(bytes(data))
    with MemoryStore(tmp_path):  # The premise: opening and refresh never read that page.
        pass
    assert _set_aside(tmp_path) == []
    with MemoryStore(tmp_path, repair_index=True) as store:  # What `memoryos reindex` does.
        assert store.reindex() == len(facts)
        assert store.all_facts() == facts
    assert [name.split("-")[0] for name in _set_aside(tmp_path)] == ["index.sqlite3.corrupt"]
    with MemoryStore(tmp_path) as store:
        assert sorted(store.snapshot(as_of="2025-06-01"), key=lambda fact: fact.id) == facts


def _damage_index_text(root, old: bytes, new: bytes) -> None:
    """Replace the only copy of ``old`` in the index file; pages stay structurally valid."""
    with closing(_raw(root)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # Everything in the main file.
    raw = _index(root).read_bytes()
    assert raw.count(old) == 1
    _index(root).write_bytes(raw.replace(old, new))


def _quick_check(root) -> list:
    with closing(_raw(root)) as connection:
        return connection.execute("PRAGMA quick_check").fetchall()


# A damaged column name in the schema that still parses: quick_check accepts it.
COLUMN_DAMAGE = pytest.mark.parametrize("old, new", [
    (b" payload TEXT NOT NULL", b" payloa\xff TEXT NOT NULL"),  # Not valid UTF-8.
    (b" value TEXT NOT NULL", b" valux TEXT NOT NULL"),  # Read by no query, only written.
    (b" verified_ns INTEGER NOT NULL", b" verified_nx INTEGER NOT NULL"),  # Read on opening.
], ids=["facts.payload", "facts.value", "sources.verified_ns"])


@COLUMN_DAMAGE
def test_reindex_repairs_a_damaged_column_name_that_quick_check_accepts(tmp_path, old, new):
    facts = _many_records(tmp_path)
    _damage_index_text(tmp_path, old, new)
    assert _quick_check(tmp_path) == [("ok",)]  # The premise: the page check finds nothing.
    with MemoryStore(tmp_path, repair_index=True) as store:  # What `memoryos reindex` does.
        assert store.reindex() == len(facts)
        assert store.all_facts() == facts
    assert [name for name in _set_aside(tmp_path) if name.startswith("index.sqlite3.corrupt-")]
    with MemoryStore(tmp_path) as store:
        assert store.all_facts() == facts
        assert store.add("person 0", "employer", "Acme", valid_from="2025-01-01").value == "Acme"


@COLUMN_DAMAGE
def test_a_damaged_column_name_is_reported_as_damage_and_left_in_place(tmp_path, old, new):
    """A normal open does not check the schema, but the error names the repair."""
    with MemoryStore(tmp_path) as store:
        fact = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01")
    _damage_index_text(tmp_path, old, new)
    try:
        store = MemoryStore(tmp_path)
    except StoreIntegrityError as error:  # The damaged column is read on opening.
        assert "run `memoryos reindex`" in str(error) and "no such column" in str(error)
    else:
        # Reading fails on a damaged payload column; writing, on any column of facts.
        with store, pytest.raises(sqlite3.DatabaseError) as caught:
            store.get(fact.id)
            store.add("Alex", "employer", "Acme", valid_from="2025-01-01")
        # The CLI, the HTTP API (503) and the MCP tools add the repair hint to this.
        assert store_module._damaged(caught.value)
    assert _set_aside(tmp_path) == []


def test_only_a_repair_checks_every_page_and_the_schema(tmp_path, monkeypatch):
    """quick_check reads every page of the index, so a normal open runs neither check."""
    MemoryStore(tmp_path).close()
    statements: list = []
    connect = MemoryStore._connect

    def traced(self):
        connection = connect(self)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(MemoryStore, "_connect", traced)
    MemoryStore(tmp_path).close()
    assert statements and not [sql for sql in statements if "quick_check" in sql
                               or "table_info" in sql]
    MemoryStore(tmp_path, repair_index=True).close()
    assert [sql for sql in statements if "quick_check" in sql or "table_info" in sql] == [
        "PRAGMA quick_check", "PRAGMA table_info(facts)", "PRAGMA table_info(sources)"]


def test_a_repair_sets_aside_an_index_whose_page_check_quotes_a_missing_column(tmp_path):
    """Only SQLite's own missing-column error stops a repair. A page-check report that
    happens to contain those words (here, a table's name) is damage like any other."""
    state = tmp_path / ".memoryos"
    state.mkdir()
    with closing(sqlite3.connect(str(state / "index.sqlite3"))) as connection:
        connection.execute('CREATE TABLE "t has no column named c" (a INTEGER CHECK (a > 0))')
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute('INSERT INTO "t has no column named c" VALUES (0)')
        connection.execute("PRAGMA user_version = 3")
        connection.commit()
    with MemoryStore(tmp_path, repair_index=True) as memory:
        assert memory.all_facts() == []
    assert [path.name for path in state.iterdir() if ".corrupt-" in path.name]


def test_only_sqlites_own_wording_counts_as_a_missing_column():
    assert store_module._missing_column(sqlite3.OperationalError("no such column: payload"))
    assert store_module._missing_column(
        sqlite3.OperationalError("table facts has no column named payload"))
    assert not store_module._missing_column(sqlite3.DatabaseError(
        "database disk image is malformed (CHECK constraint failed in t has no column named c)"))


@pytest.mark.parametrize("old, new, cause", [
    (b'"value": "Riverton"', b'"value"; "Riverton"', "Expecting ':' delimiter"),
    (b'"value": "Riverton"', b'"valux": "Riverton"', "Missing required fact fields: value"),
], ids=["not JSON", "not a record"])
def test_a_damaged_record_payload_in_the_index_gets_the_repair_hint(tmp_path, old, new, cause):
    """One damaged byte in a stored payload that is still valid UTF-8 is index damage,
    not invalid input; SQLite does not checksum cell contents."""
    with MemoryStore(tmp_path) as store:
        fact = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01")
    _damage_index_text(tmp_path, old, new)
    assert _quick_check(tmp_path) == [("ok",)]
    with MemoryStore(tmp_path) as store:
        reads = (lambda: store.get(fact.id), store.all_facts, lambda: store.history("Alex"),
                 lambda: store.add("Alex", "home city", "Lakeside", valid_from="2025-02-01"))
        for read in reads:
            with pytest.raises(StoreIntegrityError) as caught:
                read()
            message = str(caught.value)
            assert message.startswith("The index .memoryos/index.sqlite3 is damaged (record "
                                      + fact.id + " cannot be read: " + cause)
            assert "run `memoryos reindex`" in message
        assert _records(tmp_path) == [fact.id + ".md"]  # The refused write left nothing.
        assert store.reindex() == 1  # Rebuilt from Markdown, without setting anything aside.
        assert store.get(fact.id) == fact
    assert _set_aside(tmp_path) == []


def test_new_topic_folders_are_flushed_with_the_folder_that_holds_them(tmp_path, monkeypatch):
    flushed = []
    monkeypatch.setattr(store_module, "_flush_directory", flushed.append)
    with MemoryStore(tmp_path) as store:
        memory = store.memory_dir
        store.add("Alex", "home city", "Riverton", valid_from="2025-01-01",
                  topic="projects/2024/q1")
        # Each new folder's entry lives in its parent; the record's entry in its folder.
        assert {memory, memory / "projects", memory / "projects" / "2024"} <= set(flushed)
        assert flushed[-1] == memory / "projects" / "2024" / "q1"
        flushed.clear()
        store.add("Alex", "employer", "Acme", valid_from="2025-01-01", topic="projects/2024/q1")
        assert flushed == [memory / "projects" / "2024" / "q1"]  # No new folders this time.
        flushed.clear()
        store.add("Alex", "school", "Northside", valid_from="2025-01-01", topic="projects/2025")
        assert flushed == [memory / "projects", memory / "projects" / "2025"]


def test_a_topic_spelled_unlike_its_existing_folder_is_refused(tmp_path):
    """On a case-insensitive disk, a topic spelled unlike its existing folder is refused."""
    with MemoryStore(tmp_path) as store:
        first = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01",
                          topic="Work/Q1")
        if not (store.memory_dir / "work").is_dir():
            pytest.skip("needs a case-insensitive file system")
        for topic, existing in (("work", "Work"), ("Work/q1", "Work/Q1")):
            with pytest.raises(ValueError, match="{!r} already exists as {!r} in another letter "
                                                 "case".format(topic, existing)):
                store.add("Alex", "employer", "Acme", valid_from="2025-01-01", topic=topic)
        with pytest.raises(ValueError, match="letter case"):
            store.correct(first.id, "Lakeside", topic="work/Q1")
        # Nothing was written, and the existing spelling still works without a rebuild.
        assert _records(tmp_path) == [first.id + ".md"]
        added = store.add("Alex", "employer", "Acme", valid_from="2025-01-01", topic="Work")
        assert store.refresh() is False
        assert store.all_facts() == [first, added]
    assert os.listdir(tmp_path / "memory") == ["Work"]


def test_revisions_of_a_record_filed_under_another_spelling_follow_its_folder(tmp_path):
    """A record whose topic is spelled unlike its folder is corrected and retracted there."""
    # As earlier releases filed it on a case-insensitive disk: topic "work" in "Work/".
    fact = Fact(id="a" * 32, entity="Alex", attribute="home city", value="Riverton",
                valid_from="2025-01-01", recorded_at="2025-01-01", topic="work")
    work = tmp_path / "memory" / "Work"
    work.mkdir(parents=True)
    (work / (fact.id + ".md")).write_text(MemoryStore._render(fact), encoding="utf-8")
    with MemoryStore(tmp_path) as store:
        corrected = store.correct(fact.id, "Lakeside", recorded_at="2025-01-02")
        retracted = store.retract(corrected.id, recorded_at="2025-01-03")
        assert corrected.topic == retracted.topic == "Work"
        assert store.refresh() is False  # The index paths match the folder as spelled.
        assert [found.id for found in store.all_facts()] == [fact.id, corrected.id, retracted.id]
    assert os.listdir(tmp_path / "memory") == ["Work"]
    assert sorted(os.listdir(work)) == sorted(found.id + ".md"
                                              for found in (fact, corrected, retracted))


def _revise_both_in_the_topic_folder(root, employer, city):
    """Retract one record and correct the other; neither call names a topic, as in
    the CLI, the HTTP API and correction approvals. Both land in the "Work" folder."""
    with MemoryStore(root) as store:
        retracted = store.retract(employer.id, recorded_at="2025-01-02")
        corrected = store.correct(city.id, "Lakeside", recorded_at="2025-01-02")
        assert retracted.topic == corrected.topic == "Work"
        assert store.refresh() is False  # Filed under the folder's own spelling.
        assert {store.path_for(found.id).parent for found in (retracted, corrected)} == {
            store.memory_dir / "Work"}
        assert sorted(found.id for found in store.all_facts()) == sorted(
            found.id for found in (employer, city, retracted, corrected))


def _filed(folder, *facts):
    folder.mkdir(parents=True, exist_ok=True)
    for fact in facts:
        (folder / (fact.id + ".md")).write_text(MemoryStore._render(fact), encoding="utf-8")


def test_a_record_sorted_into_a_subfolder_of_its_topic_folder_spelled_otherwise_is_revised(
        tmp_path):
    """Topic "work", filed in "Work/" by an earlier release, then sorted into "Work/2025/"."""
    employer, city = (Fact(id=letter * 32, entity="Alex", attribute=attribute, value=value,
                           valid_from="2025-01-01", recorded_at="2025-01-01", topic="work")
                      for letter, attribute, value in (("a", "employer", "Acme"),
                                                       ("b", "home city", "Riverton")))
    _filed(tmp_path / "memory" / "Work" / "2025", employer, city)
    if not (tmp_path / "memory" / "work").is_dir():
        pytest.skip("needs a case-insensitive file system")
    _revise_both_in_the_topic_folder(tmp_path, employer, city)
    assert os.listdir(tmp_path / "memory") == ["Work"]


def test_a_moved_record_whose_topic_folder_was_renamed_in_another_letter_case_is_revised(
        tmp_path):
    """A record moved out of "work/", which was renamed "Work/" afterwards, keeps its topic."""
    with MemoryStore(tmp_path) as store:
        employer = store.add("Alex", "employer", "Acme", valid_from="2025-01-01",
                             recorded_at="2025-01-01", topic="work")
        city = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01",
                         recorded_at="2025-01-01", topic="work")
    memory = tmp_path / "memory"
    (memory / "archive").mkdir()
    for fact in (employer, city):
        (memory / "work" / (fact.id + ".md")).rename(memory / "archive" / (fact.id + ".md"))
    (memory / "work").rename(memory / "Work")
    if not (memory / "work").is_dir():
        pytest.skip("needs a case-insensitive file system")
    _revise_both_in_the_topic_folder(tmp_path, employer, city)
    assert sorted(os.listdir(memory)) == ["Work", "archive"]


def _killed_writer(root, point):
    """Add a record and die from SIGKILL at ``point``; nothing gets to clean up."""
    from temporal_memory_os import store as store_module

    def die(*_arguments, **_keywords):
        os.kill(os.getpid(), signal.SIGKILL)

    store = MemoryStore(root)
    if point == "before_rename":
        # The temporary file is written and flushed; the rename never happens.
        store_module.os.replace = die
    else:
        # The record is in place; the index row and COMMIT never happen.
        store._index_record = die
    store.add("Alex", "home city", "Lakeside", valid_from="2025-02-01")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("point", ["before_rename", "after_rename"])
def test_a_writer_killed_mid_write_leaves_a_store_that_opens_cleanly(tmp_path, point):
    with MemoryStore(tmp_path) as store:
        kept = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01")
    writer = SPAWN.Process(target=_killed_writer, args=(str(tmp_path), point))
    writer.start()
    _finish(writer)
    assert writer.exitcode == -signal.SIGKILL
    general = tmp_path / "memory" / "general"
    leftovers = sorted(general.glob(".memoryos-*.tmp"))
    assert len(leftovers) == (1 if point == "before_rename" else 0)
    assert len(_records(tmp_path)) == (1 if point == "before_rename" else 2)
    # A complete, valid record under a hidden name, as a copy tool or an
    # interrupted sync can leave one. Only the dot keeps it out of the store.
    stray = Fact(id="e" * 32, entity="Alex", attribute="employer", value="never committed",
                 valid_from="2025-01-01", recorded_at="2025-01-01")
    (general / ".memoryos-stray.md").write_text(MemoryStore._render(stray), encoding="utf-8")
    with MemoryStore(tmp_path) as store:
        facts = store.all_facts()
        assert stray.id not in {fact.id for fact in facts}
        if point == "before_rename":
            assert facts == [kept]
        else:
            # The renamed record is the source of truth and is indexed once.
            assert [fact.value for fact in facts] == ["Riverton", "Lakeside"]
            with pytest.raises(MemoryConflictError):
                store.add("Alex", "home city", "Lakeside", valid_from="2025-02-01")
        assert store.reindex() == len(facts)
        assert store.all_facts() == facts


def test_symlink_topic_cannot_create_directories_outside_memory(tmp_path):
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    outside.mkdir()
    with MemoryStore(root) as store:
        (store.memory_dir / "escape").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="symbolic link"):
            store.add("person", "preference", "tea", topic="escape/must-not-exist")
        assert not (outside / "must-not-exist").exists()
        assert store.all_facts() == []


@pytest.mark.parametrize("folder", ["memory", ".memoryos"])
def test_store_folders_cannot_be_symlinks(tmp_path, folder):
    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / folder).symlink_to(outside, target_is_directory=True)
    # The folders on disk need attention, not the request (the API answers 503).
    with pytest.raises(StoreIntegrityError, match="symbolic links"):
        MemoryStore(root)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_a_linked_index_file_is_reported_as_a_problem_on_disk(tmp_path, suffix):
    """A symbolic link in place of the index or its log files raises StoreIntegrityError."""
    root = tmp_path / "store"
    MemoryStore(root).close()
    outside = tmp_path / "outside.sqlite3"
    outside.write_bytes(b"")
    link = Path(str(_index(root)) + suffix)
    link.unlink(missing_ok=True)
    link.symlink_to(outside)
    with pytest.raises(StoreIntegrityError, match="The index database must not be a symbolic link"):
        MemoryStore(root)
    assert outside.read_bytes() == b""


def test_invalid_source_does_not_partially_replace_index(tmp_path):
    with MemoryStore(tmp_path) as store:
        fact = store.add("person", "preference", "tea", valid_from="2025-01-01")
        path = store.path_for(fact.id)
        path.write_text(path.read_text().replace('"confidence": 1.0', '"confidence": 7'),
                        encoding="utf-8")
        with pytest.raises(ValueError, match="confidence"):
            store.reindex()
        assert store.get(fact.id).confidence == 1.0
        assert store.get(fact.id).value == "tea"


def test_literal_metadata_markers_roundtrip_as_ordinary_data(tmp_path):
    marker = "<!-- memoryos:fact:v1 -->"
    with MemoryStore(tmp_path) as store:
        fact = store.add(
            "A title with " + marker,
            "note",
            "literal " + marker + " inside the value",
            valid_from="2025-01-01",
            evidence="Evidence can mention " + marker + " too.\n" + marker,
        )
        assert store.reindex() == 1
        assert store.get(fact.id) == fact
    with MemoryStore(tmp_path) as store:
        assert store.get(fact.id) == fact


def test_duplicate_structural_metadata_markers_still_fail(tmp_path):
    with MemoryStore(tmp_path) as store:
        fact = store.add("person", "note", "tea", valid_from="2025-01-01")
        path = store.path_for(fact.id)
        path.write_text(path.read_text() + "\n<!-- memoryos:fact:v1 -->\n", encoding="utf-8")
        with pytest.raises(ValueError, match="one canonical"):
            store.reindex()
