"""Rejection paths, damaged records, failure cleanup and the in-process CLI."""

import json
import math
import re
import sys
import types

import pytest

from temporal_memory_os import cli
from temporal_memory_os.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from temporal_memory_os.models import Fact, normalize_time, validate_topic
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.store import MemoryConflictError, MemoryStore


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "store") as memory:
        yield memory


def rewrite(store, fact_id, **changes):
    """Edit a record's canonical JSON by hand, as a user with a text editor would."""
    path = store.path_for(fact_id)
    text = path.read_text(encoding="utf-8")
    block = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    data = json.loads(block.group(1))
    data.update(changes)
    path.write_text(text.replace(block.group(1), json.dumps(data, indent=2)), encoding="utf-8")


VALID = dict(id="a" * 32, entity="person", attribute="drink", value="tea",
             valid_from="2025-01-01", recorded_at="2025-01-01")


@pytest.mark.parametrize("changes, message", [
    (dict(id="../escape"), "id"),
    (dict(id=""), "id"),
    (dict(deleted="yes"), "deleted"),
    (dict(entity="  "), "entity"),
    (dict(attribute=7), "attribute"),
    (dict(value=""), "value"),
    (dict(evidence=None), "evidence"),
    (dict(valid_from=None), "required"),
    (dict(valid_from="yesterday"), "ISO"),
    (dict(valid_from="2025-13-01"), "Invalid ISO"),
    (dict(valid_from=20250101), "ISO string"),
    (dict(valid_to="2025-01-01"), "later than"),
    (dict(valid_to="2024-12-31"), "later than"),
    (dict(confidence=True), "confidence"),
    (dict(confidence=math.nan), "confidence"),
    (dict(confidence=1.5), "confidence"),
    (dict(confidence="high"), "confidence"),
    (dict(tags="tag"), "tags"),
    (dict(links=["ok", ""]), "links"),
    (dict(topic="../outside"), "topic"),
    (dict(topic="a" * 600), "topic"),
    (dict(supersedes="a" * 32), "itself"),
    (dict(supersedes="bad id!"), "supersedes"),
])
def test_invalid_fact_fields_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        Fact(**{**VALID, **changes})


def test_fact_from_dict_rejects_malformed_payloads():
    with pytest.raises(ValueError, match="JSON object"):
        Fact.from_dict(["not", "a", "dict"])
    with pytest.raises(ValueError, match="Missing required"):
        Fact.from_dict({"id": "a" * 32})
    with pytest.raises(ValueError, match="Invalid fact fields"):
        Fact.from_dict({**VALID, "colour": "blue"})


def test_fact_defaults_normalize_and_deduplicate():
    fact = Fact(**VALID, tags=["x", "x", "y"], confidence=1)
    assert fact.tags == ["x", "y"]
    assert isinstance(fact.confidence, float)
    assert fact.valid_from == "2025-01-01T00:00:00.000000Z"
    assert normalize_time("2025-01-01T08:00:00+08:00") == fact.valid_from
    assert validate_topic("a/b-c/d.e") == "a/b-c/d.e"


@pytest.fixture
def chain(store):
    """A record, its correction, and an unrelated record."""
    first = store.add("person", "drink", "tea", valid_from="2025-01-01", recorded_at="2025-01-01")
    second = store.correct(first.id, "green tea", recorded_at="2025-01-02")
    other = store.add("person", "city", "Riverton", valid_from="2025-01-01",
                      recorded_at="2025-01-01")
    return first, second, other


@pytest.mark.parametrize("edit, message", [
    (lambda first, second, other: (second.id, dict(supersedes="f" * 32)), "Missing superseded"),
    (lambda first, second, other: (second.id, dict(valid_from="2025-02-01")), "retain"),
    (lambda first, second, other: (second.id, dict(recorded_at="2024-12-01")), "recorded before"),
    (lambda first, second, other: (other.id, dict(attribute="drink", supersedes=first.id,
                                                  recorded_at="2025-01-03")), "Two corrections"),
    (lambda first, second, other: (other.id, dict(attribute="drink")), "Duplicate effective"),
    (lambda first, second, other: (other.id, dict(id=first.id)), "Duplicate memory id"),
])
def test_damaged_histories_fail_the_rebuild_and_keep_the_old_index(store, chain, edit, message):
    target, changes = edit(*chain)
    before = store.all_facts()
    rewrite(store, target, **changes)
    with pytest.raises(ValueError, match=message):
        store.reindex()
    assert store.all_facts() == before


def test_cyclic_corrections_are_rejected(store):
    first = store.add("person", "drink", "tea", valid_from="2025-01-01", recorded_at="2025-01-01")
    second = store.correct(first.id, "green tea", recorded_at="2025-01-01")
    rewrite(store, first.id, supersedes=second.id)
    with pytest.raises(ValueError, match="Cycle"):
        store.reindex()


def test_symlinked_record_and_index_are_rejected(tmp_path):
    root = tmp_path / "store"
    with MemoryStore(root) as memory:
        fact = memory.add("person", "drink", "tea", valid_from="2025-01-01")
        (memory.memory_dir / "alias.md").symlink_to(memory.path_for(fact.id))
        with pytest.raises(ValueError, match="symbolic links"):
            memory.refresh()
        (memory.memory_dir / "alias.md").unlink()
    index = root / ".memoryos" / "index.sqlite3"
    index.rename(root / "elsewhere.sqlite3")
    index.symlink_to(root / "elsewhere.sqlite3")
    with pytest.raises(ValueError, match="index database"):
        MemoryStore(root)


def test_failed_index_update_removes_the_new_file(store, monkeypatch):
    kept = store.add("person", "city", "Riverton", valid_from="2025-01-01")

    def broken(*_args):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "_index_record", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        store.add("person", "drink", "tea", valid_from="2025-01-01")
    monkeypatch.undo()
    assert [path.stem for path in store.memory_dir.rglob("*.md")] == [kept.id]
    assert store.refresh() is False  # The signature still matches: nothing leaked.
    assert store.add("person", "drink", "tea", valid_from="2025-01-01").value == "tea"


def test_revision_errors(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    with pytest.raises(KeyError):
        store.correct("0" * 32, "coffee")
    with pytest.raises(KeyError):
        store.get("0" * 32)
    with pytest.raises(KeyError):
        store.path_for("0" * 32)
    tombstone = store.retract(fact.id)
    with pytest.raises(MemoryConflictError, match="already been retracted"):
        store.retract(tombstone.id)
    restored = store.correct(tombstone.id, "tea again")
    assert [f.value for f in store.snapshot(as_of="2025-02-01")] == ["tea again"]
    assert restored.supersedes == tombstone.id


def test_sentence_transformer_wrapper_loads_lazily_and_normalizes(monkeypatch):
    calls = {}

    class FakeModel:
        def __init__(self, name, trust_remote_code):
            calls["init"] = (name, trust_remote_code)

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            calls["encode"] = (list(texts), normalize_embeddings, show_progress_bar)
            return [[0.6, 0.8] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeModel))
    embedder = SentenceTransformerEmbedder("local-model")
    assert embedder.embed([]) == [] and "init" not in calls
    assert embedder.embed(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]
    assert calls == {"init": ("local-model", False), "encode": (["a", "b"], True, False)}
    assert embedder.name == "sentence-transformers:local-model"


def test_sentence_transformer_wrapper_explains_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match=r"temporal-memory-os\[semantic\]"):
        SentenceTransformerEmbedder("any").embed(["text"])
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder("")
    with pytest.raises(ValueError):
        HashingEmbedder(dimensions=8)


@pytest.mark.parametrize("vectors, message", [
    ([[1.0, 0.0], [math.inf, 0.0]], "finite"),
    ([[1.0, 0.0], [1.0, 0.0, 0.0]], "dimensions"),
    ([[1.0, 0.0]], "wrong number"),
])
def test_misbehaving_embedding_providers_are_reported(store, vectors, message):
    store.add("person", "drink", "tea", valid_from="2025-01-01")

    class Provider:
        semantic = True
        name = "broken"

        def embed(self, texts):
            return vectors

    with pytest.raises(ValueError, match=message):
        HybridRetriever(store, Provider()).search("tea", mode="semantic")


def test_retriever_accepts_any_snapshot_source_and_rejects_bad_settings(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01", topic="profile")

    class SnapshotOnly:
        def snapshot(self, as_of=None, known_at=None, entity=None):
            return store.snapshot(as_of, known_at, entity)

    hits = HybridRetriever(SnapshotOnly(), HashingEmbedder()).search("person tea")
    assert hits[0].citation == "memory/profile/{}.md".format(fact.id)
    for threshold in (math.nan, 1.5, -2.0):
        with pytest.raises(ValueError):
            HybridRetriever(store, HashingEmbedder(), semantic_threshold=threshold)
    with pytest.raises(ValueError):
        HybridRetriever(store, HashingEmbedder()).search(None)
    with pytest.raises(ValueError):
        HybridRetriever(store, HashingEmbedder()).search("tea", entity=" ")


def run(capsys, root, *arguments):
    code = cli.main(["--root", str(root), *arguments])
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out.strip() else None, captured.err


def test_cli_covers_the_record_lifecycle(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MEMORYOS_EMBEDDING_MODEL", raising=False)
    root = tmp_path / "memory-store"
    code, added, _ = run(capsys, root, "add", "Acme application", "status", "pending",
                         "--from", "2025-01-01", "--tag", "career", "--link", "Acme",
                         "--topic", "career/applications", "--evidence", "Synthetic.")
    assert code == 0 and added["links"] == ["Acme"]
    _, corrected, _ = run(capsys, root, "correct", added["id"], "interview invited",
                          "--source", "synthetic:correction")
    assert corrected["supersedes"] == added["id"]
    _, history, _ = run(capsys, root, "history", "Acme application", "--attribute", "status")
    assert [fact["value"] for fact in history] == ["interview invited"]
    _, retracted, _ = run(capsys, root, "retract", corrected["id"])
    assert retracted["deleted"] is True
    assert run(capsys, root, "search", "Acme status", "--as-of", "2025-02-01")[1] == []
    assert run(capsys, root, "reindex")[1] == {"indexed": 3}
    _, proposals, _ = run(capsys, root, "extract", "--text", "Alex | city | Riverton",
                          "--source", "synthetic:cli", "--observed-at", "2025-01-01")
    assert len(run(capsys, root, "proposals")[1]) == 1
    _, rejected, _ = run(capsys, root, "reject", proposals[0]["id"])
    assert rejected["status"] == "rejected"
    assert run(capsys, root, "proposals", "--status", "all")[1][0]["status"] == "rejected"
    code, output, error = run(capsys, root, "approve", "f" * 32)
    assert code == 1 and output is None and "not found" in json.loads(error)["error"]


@pytest.mark.skipif(not __import__("shutil").which("git"), reason="Git unavailable")
def test_cli_git_commands_report_missing_identity(tmp_path, capsys, monkeypatch):
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                 "GIT_COMMITTER_EMAIL", "EMAIL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "empty-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "private-memory"
    assert run(capsys, root, "git-init")[1]["initialized"] is True
    assert run(capsys, root, "git-log")[1] == []
    run(capsys, root, "demo")
    code, _, error = run(capsys, root, "checkpoint", "--message", "Initial memory")
    assert code == 1 and "user.name" in json.loads(error)["error"]
    # Environment identities are explicit and still accepted.
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Memory Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "memory-test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Memory Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "memory-test@example.invalid")
    code, committed, _ = run(capsys, root, "checkpoint", "--message", "Initial memory")
    assert code == 0 and committed["committed"] is True
    assert [entry["message"] for entry in run(capsys, root, "git-log")[1]] == ["Initial memory"]
