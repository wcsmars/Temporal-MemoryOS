"""Exercise file confinement and writers that do not share Python locks."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest

from temporal_memory_os.store import MemoryStore


def _competing_add(root):
    with MemoryStore(root) as store:
        try:
            store.add("shared", "status", "active", valid_from="2025-01-01")
            return "written"
        except ValueError:
            return "conflict"


def test_independent_processes_cannot_create_duplicate_effective_facts(tmp_path):
    # Initialize before launch to keep this focused on the write transaction.
    with MemoryStore(tmp_path):
        pass
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_competing_add, [str(tmp_path)] * 3))
    assert results.count("written") == 1
    assert results.count("conflict") == 2
    with MemoryStore(tmp_path) as store:
        assert len(store.all_facts()) == 1


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
    with pytest.raises(ValueError, match="symbolic links"):
        MemoryStore(root)
    assert list(outside.iterdir()) == []


def test_invalid_source_does_not_partially_replace_index(tmp_path):
    with MemoryStore(tmp_path) as store:
        fact = store.add("person", "preference", "tea", valid_from="2025-01-01")
        path = store.path_for(fact.id)
        path.write_text(path.read_text().replace('"confidence": 1.0', '"confidence": 7'), encoding="utf-8")
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
