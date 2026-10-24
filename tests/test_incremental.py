"""The index is trusted only while it matches the Markdown files on disk."""

import os
import shutil
import sqlite3
from contextlib import closing

import pytest

from temporal_memory_os import store as store_module
from temporal_memory_os.embeddings import HashingEmbedder
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.store import MemoryConflictError, MemoryStore


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path) as memory:
        yield memory


def test_refresh_is_a_no_op_when_nothing_changed(store):
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    assert store.refresh() is False
    assert store.refresh() is False


def test_write_after_external_edit_sees_the_edited_record(store):
    tea = store.add("person", "drink", "tea", valid_from="2025-01-01")
    path = store.path_for(tea.id)
    path.write_text(path.read_text(encoding="utf-8").replace('"tea"', '"green tea"'),
                    encoding="utf-8")
    store.add("person", "city", "Riverton", valid_from="2025-01-01")
    assert store.get(tea.id).value == "green tea"


def test_write_after_invalid_external_edit_fails_without_partial_index(store):
    tea = store.add("person", "drink", "tea", valid_from="2025-01-01")
    path = store.path_for(tea.id)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"confidence": 1.0', '"confidence": 9'), encoding="utf-8")
    with pytest.raises(ValueError, match="confidence"):
        store.add("person", "city", "Riverton", valid_from="2025-01-01")
    assert [fact.value for fact in store.all_facts()] == ["tea"]
    assert list(store.memory_dir.rglob("*.md")) == [path]


def test_same_size_edit_with_restored_mtime_is_detected_by_hash(store):
    tea = store.add("person", "drink", "tea", valid_from="2025-01-01")
    path = store.path_for(tea.id)
    before = path.stat()
    path.write_text(path.read_text(encoding="utf-8").replace('"tea"', '"tee"'), encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert store.refresh() is True
    assert store.get(tea.id).value == "tee"


def test_deleted_and_copied_records_trigger_rebuild(store):
    tea = store.add("person", "drink", "tea", valid_from="2025-01-01")
    city = store.add("person", "city", "Riverton", valid_from="2025-01-01")
    store.path_for(city.id).unlink()
    assert store.refresh() is True
    with pytest.raises(KeyError):
        store.get(city.id)
    # A duplicated file reuses the ID inside it, which full validation rejects.
    shutil.copy(store.path_for(tea.id), store.memory_dir / "copy.md")
    with pytest.raises(ValueError, match="Duplicate memory id"):
        store.add("person", "city", "Lakeside", valid_from="2025-02-01")


def test_unmarked_notes_are_tracked_but_not_indexed(store):
    note = store.memory_dir / "notes" / "ideas.md"
    note.parent.mkdir()
    note.write_text("# Ordinary note\n", encoding="utf-8")
    assert store.refresh() is True
    assert store.all_facts() == []
    assert store.refresh() is False


def test_index_without_source_signature_is_rebuilt(tmp_path):
    with MemoryStore(tmp_path) as store:
        fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    # Simulate an index from version 1, which recorded no source signature.
    connection = sqlite3.connect(tmp_path / ".memoryos" / "index.sqlite3")
    connection.execute("DROP TABLE sources")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()
    with MemoryStore(tmp_path) as store:
        assert store.get(fact.id).value == "tea"
        assert store.refresh() is False


def test_newer_index_version_is_refused(tmp_path):
    MemoryStore(tmp_path).close()
    connection = sqlite3.connect(tmp_path / ".memoryos" / "index.sqlite3")
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="newer"):
        MemoryStore(tmp_path)


def test_scoped_write_validation_still_rejects_conflicts(store):
    first = store.add("person", "drink", "tea", valid_from="2025-01-01")
    store.add("other", "drink", "tea", valid_from="2025-01-01")  # Different history.
    with pytest.raises(MemoryConflictError):
        store.add("person", "drink", "coffee", valid_from="2025-01-01")
    store.correct(first.id, "green tea")
    with pytest.raises(MemoryConflictError, match="already been superseded"):
        store.correct(first.id, "black tea")


def test_entity_filter_matches_full_snapshot(store):
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    store.add("person", "drink", "coffee", valid_from="2025-03-01")
    store.add("friend", "drink", "water", valid_from="2025-01-01")
    full = [fact for fact in store.snapshot(as_of="2025-02-01") if fact.entity == "person"]
    assert store.snapshot(as_of="2025-02-01", entity="person") == full


def test_embedding_cache_eviction_keeps_results_correct(store):
    for name in ("alpha", "beta", "gamma"):
        store.add(name, "drink", "tea", valid_from="2025-01-01")
    retriever = HybridRetriever(store, embedder=HashingEmbedder())
    retriever.cache_limit = 2
    first = retriever.search("tea drink", as_of="2025-02-01", mode="semantic")
    second = retriever.search("tea drink", as_of="2025-02-01", mode="semantic")
    assert len(first) == 3
    assert [hit.fact.id for hit in first] == [hit.fact.id for hit in second]


def test_rebuild_by_another_instance_never_serves_a_stale_cached_record(tmp_path):
    with MemoryStore(tmp_path) as first, MemoryStore(tmp_path) as second:
        tea = first.add("person", "drink", "tea", valid_from="2025-01-01")
        assert first.get(tea.id).value == "tea"  # Now cached by the first instance.
        path = first.path_for(tea.id)
        path.write_text(path.read_text(encoding="utf-8").replace('"tea"', '"mate"'),
                        encoding="utf-8")
        assert second.refresh() is True
        # The shared index already matches the files, so no rebuild is needed here,
        # yet the changed payload must replace the cached record.
        assert first.refresh() is False
        assert first.get(tea.id).value == "mate"


def test_symlinked_folder_inside_memory_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    with MemoryStore(tmp_path / "store") as store:
        (store.memory_dir / "linked").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="symbolic links"):
            store.refresh()


def test_reading_an_unchanged_store_never_waits_for_a_writer(tmp_path, monkeypatch):
    with MemoryStore(tmp_path) as store:
        tea = store.add("person", "drink", "tea", valid_from="2025-01-01")
    index = tmp_path / ".memoryos" / "index.sqlite3"
    # Files verified well after they last changed, the usual state seconds after a
    # write, are trusted without hashing them again (which would take the write lock).
    with closing(sqlite3.connect(str(index))) as connection:
        connection.execute("UPDATE sources SET verified_ns = mtime_ns + 60000000000")
        connection.commit()
    monkeypatch.setattr(store_module, "_LOCK_RETRY_SECONDS", 1.0)  # Any wait fails fast.
    with MemoryStore(tmp_path) as opened_before, \
            closing(sqlite3.connect(str(index), isolation_level=None)) as writer:
        writer.execute("BEGIN IMMEDIATE")  # What a write or a Git checkpoint holds.
        try:
            assert opened_before.refresh() is False
            with MemoryStore(tmp_path) as opened_during:
                assert opened_during.get(tea.id).value == "tea"
                assert [fact.value for fact in opened_during.snapshot(as_of="2025-02-01")] == [
                    "tea"]
        finally:
            writer.execute("ROLLBACK")
