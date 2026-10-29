"""Filesystem replacement and nested approval transaction regressions."""

import os
import signal
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from temporal_memory_os.store import MemoryStore, StoreIntegrityError


@contextmanager
def io_deadline():
    """A regressed FIFO read must fail the test rather than hang the test runner."""
    def expired(_signum, _frame):
        raise AssertionError("Memory read blocked beyond the I/O test deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    timer = signal.setitimer(signal.ITIMER_REAL, 10)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("operation", ["refresh", "reindex", "add"])
def test_open_store_refuses_replaced_memory_root(tmp_path, operation):
    with MemoryStore(tmp_path / "store") as store:
        original = store.add("person", "drink", "tea", valid_from="2025-01-01")
        outside = tmp_path / "outside"
        store.memory_dir.rename(outside)
        store.memory_dir.symlink_to(outside, target_is_directory=True)
        with pytest.raises(StoreIntegrityError, match="symbolic links"):
            if operation == "add":
                store.add("person", "drink", "coffee", valid_from="2025-02-01")
            else:
                getattr(store, operation)()
        assert [p.stem for p in outside.rglob("*.md")] == [original.id]
        assert store.all_facts() == [original]


@pytest.mark.parametrize("operation", ["refresh", "reindex", "reopen"])
def test_markdown_fifo_is_reported_without_reading_it(tmp_path, monkeypatch, operation):
    with MemoryStore(tmp_path) as store:
        fifo = store.memory_dir / "blocked.md"
        os.mkfifo(fifo)
        real_read = Path.read_bytes

        def never_read_fifo(path):
            if path == fifo:
                raise AssertionError("attempted a blocking read of a named pipe")
            return real_read(path)

        monkeypatch.setattr(Path, "read_bytes", never_read_fifo)
        with pytest.raises(StoreIntegrityError, match="memory/blocked.md.*regular files"):
            if operation == "reopen":
                MemoryStore(tmp_path).close()
            else:
                getattr(store, operation)()


@pytest.mark.parametrize("replacement", ["fifo", "symlink"])
@pytest.mark.parametrize("operation", ["refresh", "reindex"])
def test_record_replaced_after_scan_cannot_block_or_read_link(
        tmp_path, monkeypatch, replacement, operation):
    with MemoryStore(tmp_path / "store") as store:
        original = store.add("person", "drink", "tea", valid_from="2025-01-01")
        path = store.path_for(original.id)
        outside = tmp_path / "outside.md"
        outside.write_bytes(path.read_bytes())
        real_scan = store._scan
        changed = False

        def scan_then_replace():
            nonlocal changed
            scan = real_scan()
            if not changed:
                changed = True
                path.unlink()
                if replacement == "fifo":
                    os.mkfifo(path)
                else:
                    path.symlink_to(outside)
            return scan

        monkeypatch.setattr(store, "_scan", scan_then_replace)
        message = "regular files" if replacement == "fifo" else "symbolic links"
        with io_deadline(), pytest.raises(StoreIntegrityError, match=message) as caught:
            getattr(store, operation)()
        assert path.relative_to(store.root).as_posix() in str(caught.value)
        assert store.get(original.id) == original


@pytest.mark.parametrize("ancestor", ["memory", "topic"])
def test_ancestor_link_replaced_after_scan_is_rejected(tmp_path, monkeypatch, ancestor):
    with MemoryStore(tmp_path / "store") as store:
        original = store.add("person", "drink", "tea", valid_from="2025-01-01",
                             topic="nested/leaf")
        path = store.path_for(original.id)
        replaced = store.memory_dir if ancestor == "memory" else store.memory_dir / "nested"
        real_scan = store._scan
        changed = False

        def scan_then_replace():
            nonlocal changed
            scan = real_scan()
            if not changed:
                changed = True
                outside = tmp_path / "outside"
                replaced.rename(outside)
                replaced.symlink_to(outside, target_is_directory=True)
            return scan

        monkeypatch.setattr(store, "_scan", scan_then_replace)
        with pytest.raises(StoreIntegrityError, match="symbolic links") as caught:
            store.refresh()
        assert path.relative_to(store.root).as_posix() in str(caught.value)
        assert store.get(original.id) == original


def test_nested_writes_commit_as_one_transaction(tmp_path):
    with MemoryStore(tmp_path) as store, MemoryStore(tmp_path) as observer:
        with store._transaction():
            first = store.add("person", "drink", "tea", valid_from="2025-01-01")
            second = store.correct(first.id, "coffee")
            assert observer.all_facts() == []
            assert store.get(second.id) == second
        assert observer.all_facts() == [first, second]
        assert store.refresh() is False
    with MemoryStore(tmp_path) as reopened:
        assert reopened.all_facts() == [first, second]


def test_outer_failure_rolls_back_successful_nested_writes_and_files(tmp_path):
    with MemoryStore(tmp_path) as store:
        kept = store.add("person", "city", "Paris", valid_from="2025-01-01")
        with pytest.raises(RuntimeError, match="cancel approval"):
            with store._transaction():
                first = store.add("person", "drink", "tea", valid_from="2025-01-01")
                store.correct(first.id, "coffee")
                raise RuntimeError("cancel approval")
        assert store.all_facts() == [kept]
        assert [p.stem for p in store.memory_dir.rglob("*.md")] == [kept.id]
        assert store.refresh() is False
        assert not store.connection.in_transaction


def test_caught_inner_failure_only_rolls_back_its_own_writes(tmp_path):
    with MemoryStore(tmp_path) as store:
        with store._transaction():
            kept = store.add("person", "city", "Paris", valid_from="2025-01-01")
            with pytest.raises(RuntimeError, match="cancel inner write"):
                with store._transaction():
                    store.add("person", "drink", "tea", valid_from="2025-01-01")
                    raise RuntimeError("cancel inner write")
            last = store.add("person", "language", "English", valid_from="2025-01-01")
        assert store.all_facts() == [kept, last]
        assert {p.stem for p in store.memory_dir.rglob("*.md")} == {kept.id, last.id}
        assert store.refresh() is False


def test_nested_index_failure_cleans_files_and_allows_outer_commit(tmp_path, monkeypatch):
    with MemoryStore(tmp_path) as store:
        with store._transaction():
            kept = store.add("person", "city", "Paris", valid_from="2025-01-01")

            def fail_index(_fact, _path):
                raise RuntimeError("index write failed")

            with monkeypatch.context() as patch:
                patch.setattr(store, "_index_record", fail_index)
                with pytest.raises(RuntimeError, match="index write failed"):
                    store.add("person", "drink", "tea", valid_from="2025-01-01")
        assert store.all_facts() == [kept]
        assert [p.stem for p in store.memory_dir.rglob("*.md")] == [kept.id]
        assert store.refresh() is False


def test_failed_read_commit_leaves_store_usable(tmp_path):
    with MemoryStore(tmp_path) as store:
        original = store.add("person", "drink", "tea", valid_from="2025-01-01")
        connection = store.connection

        class FailOneCommit:
            failed = False

            def execute(self, statement, *args):
                if statement == "COMMIT" and not self.failed:
                    self.failed = True
                    raise sqlite3.OperationalError("injected read commit failure")
                return connection.execute(statement, *args)

            def __getattr__(self, name):
                return getattr(connection, name)

        store.connection = FailOneCommit()
        with pytest.raises(sqlite3.OperationalError, match="read commit failure"):
            store.refresh()
        assert not connection.in_transaction
        assert store.refresh() is False
        assert store.get(original.id) == original
        added = store.add("person", "drink", "coffee", valid_from="2025-02-01")
        assert store.get(added.id) == added


@pytest.mark.parametrize("phase", ["insert", "savepoint", "release", "rollback_to"])
def test_lost_nested_transaction_cannot_commit_later_writes(tmp_path, phase):
    """A caught SQLite failure cannot silently turn later writes into autocommits."""
    with MemoryStore(tmp_path) as store, MemoryStore(tmp_path) as observer:
        connection = store.connection

        class BrokenNestedTransaction:
            failed = False

            def execute(self, statement, *args):
                trigger = {
                    "insert": "INSERT INTO facts",
                    "savepoint": "SAVEPOINT ",
                    "release": "RELEASE SAVEPOINT ",
                    "rollback_to": "ROLLBACK TO SAVEPOINT ",
                }[phase]
                if statement.startswith(trigger) and not self.failed:
                    self.failed = True
                    if phase != "rollback_to":
                        connection.execute("ROLLBACK")  # SQLite's automatic IOERR/FULL rollback.
                    raise sqlite3.OperationalError("injected SQLite failure")
                return connection.execute(statement, *args)

            def __getattr__(self, name):
                return getattr(connection, name)

        with pytest.raises(sqlite3.OperationalError, match="transaction was aborted"):
            with store._transaction():
                store.add("person", "city", "Paris", valid_from="2025-01-01")
                store.connection = BrokenNestedTransaction()
                with pytest.raises(sqlite3.OperationalError, match="injected SQLite failure"):
                    with store._transaction():
                        store.add("person", "drink", "tea", valid_from="2025-01-01")
                        if phase == "rollback_to":
                            raise ValueError("force nested rollback")
                with pytest.raises(sqlite3.OperationalError, match="transaction was aborted"):
                    store.add("person", "language", "English", valid_from="2025-01-01")
        assert observer.all_facts() == []
        assert store.all_facts() == []
        assert list(store.memory_dir.rglob("*.md")) == []
        assert not connection.in_transaction
        assert store.refresh() is False
        retry = store.add("person", "city", "Rome", valid_from="2025-01-01")
        assert observer.all_facts() == [retry]
