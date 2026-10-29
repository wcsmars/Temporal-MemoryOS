"""Regressions from the 29 October audit of retrieval and proposal approval."""

import sqlite3

import pytest

from temporal_memory_os.embeddings import (
    EmbeddingUnavailableError, HashingEmbedder, SentenceTransformerEmbedder,
)
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path) as memory:
        yield memory


@pytest.mark.parametrize("mode", ["graph", "hybrid"])
@pytest.mark.parametrize("query", ["☕", '"☕"?'])
def test_symbol_only_entity_seeds_graph_and_one_hop(store, mode, query):
    seed = store.add("☕", "type", "cafe", links=["Cafe"], valid_from="2025-01-01")
    adjacent = store.add("Cafe", "city", "Paris", links=["Paris"], valid_from="2025-01-01")
    store.add("Paris", "country", "France", valid_from="2025-01-01")

    class NoEmbedding(HashingEmbedder):
        def embed(self, texts):
            raise AssertionError("A symbol-only query must not run word hashing")

    results = HybridRetriever(store, NoEmbedding()).search(query, mode=mode)
    assert [hit.fact.id for hit in results] == [seed.id, adjacent.id]
    assert [hit.channels for hit in results] == [{"graph": 1.0}, {"graph": 0.5}]


def test_symbol_only_graph_query_preserves_temporal_filtering(store):
    old = store.add("☕", "city", "Paris", valid_from="2025-01-01",
                    recorded_at="2025-01-01")
    store.add("☕", "city", "London", valid_from="2025-02-01", recorded_at="2025-02-01")
    [hit] = HybridRetriever(store).search("☕", mode="graph", as_of="2025-01-15")
    assert hit.fact.id == old.id


@pytest.mark.parametrize("output", [None, [None], [["invalid"]], [[10 ** 1000]]])
def test_model_result_conversion_failures_are_embedding_errors(output):
    class BrokenModel:
        def encode(self, *args, **kwargs):
            return output

    provider = SentenceTransformerEmbedder("test-model")
    provider._model = BrokenModel()
    with pytest.raises(EmbeddingUnavailableError, match="embedding model failed") as caught:
        provider.embed(["query"])
    assert isinstance(caught.value.__cause__, (TypeError, ValueError, OverflowError))


def test_approval_prevents_a_direct_writer_changing_a_validated_preview(store, monkeypatch):
    original = store.add("user", "city", "London", valid_from="2025-01-01")
    queue = ProposalQueue(store)
    [proposal] = queue.extract("user | city | Paris | 2025-03-01", source="chat")
    classify = queue._classify
    blocked = []
    with MemoryStore(store.root) as other:
        other.connection.execute("PRAGMA busy_timeout = 0")

        def change_after_preview(*args):
            preview = classify(*args)
            # This connection would otherwise commit after the preview was checked,
            # silently replacing the prior value that the user approved against.
            try:
                other.correct(original.id, "Berlin", source="another-writer")
            except sqlite3.OperationalError as error:
                assert "locked" in str(error)
                blocked.append(True)
            return preview

        monkeypatch.setattr(queue, "_classify", change_after_preview)
        assert queue.approve(proposal.id).value == "Paris"
        assert blocked == [True]
        # The lock is released on success and other writes can continue normally.
        assert other.correct(original.id, "Berlin", source="another-writer").value == "Berlin"


def test_failed_approval_commit_leaves_pending_proposal_and_no_new_fact(store):
    queue = ProposalQueue(store)
    [proposal] = queue.extract("user | city | Paris | 2025-03-01", source="chat")

    class FailingCommit:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, *args):
            if sql == "COMMIT" and self.connection.execute(
                    "SELECT COUNT(*) FROM facts").fetchone()[0]:
                raise sqlite3.OperationalError("disk I/O error")
            return self.connection.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    connection = store.connection
    store.connection = FailingCommit(connection)
    try:
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            queue.approve(proposal.id)
    finally:
        store.connection = connection
    assert queue.get(proposal.id).status == "pending"
    assert store.all_facts() == []
    assert list(store.memory_dir.rglob("*.md")) == []
    assert queue.approve(proposal.id).value == "Paris"
