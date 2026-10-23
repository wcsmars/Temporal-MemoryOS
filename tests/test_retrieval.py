"""Retrieval behavior at temporal, provenance and graph boundaries."""

import pytest

from temporal_memory_os.embeddings import HashingEmbedder, SentenceTransformerEmbedder, create_embedder
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.store import MemoryStore


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    monkeypatch.delenv("MEMORYOS_EMBEDDING_MODEL", raising=False)


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path) as memory:
        yield memory


def test_historical_search_cannot_return_later_state(store):
    pending = store.add("Northstar", "application_status", "pending", valid_from="2024-09-01", source="email:1")
    invitation = store.add("Northstar", "application_status", "invited", valid_from="2024-09-15", source="email:2")
    store.add("Northstar", "application_status", "completed", valid_from="2024-10-01", source="email:3")
    retriever = HybridRetriever(store)
    early = retriever.search("Northstar application status", as_of="2024-09-10")
    middle = retriever.search("Northstar application status", as_of="2024-09-20")
    assert [result.fact.id for result in early] == [pending.id]
    assert [result.fact.id for result in middle] == [invitation.id]
    assert early[0].fact.source == "email:1"
    assert early[0].citation.endswith(pending.id + ".md")
    assert early[0].to_dict()["fact"]["value"] == "pending"


def test_knowledge_cutoff_excludes_correction_not_yet_recorded(store):
    original = store.add(
        "Alice", "location", "London", valid_from="2024-01-01",
        recorded_at="2024-01-02", source="conversation:1",
    )
    corrected = store.correct(original.id, "Paris", recorded_at="2024-02-01", source="conversation:2")
    retriever = HybridRetriever(store)
    then = retriever.search("Alice location", as_of="2024-01-15", known_at="2024-01-20")
    now = retriever.search("Alice location", as_of="2024-01-15", known_at="2024-02-02")
    assert then[0].fact.id == original.id
    assert now[0].fact.id == corrected.id


def test_graph_expands_one_hop_without_leaking_future_relationships(store):
    alice = store.add("Alice", "employer", "Acme", links=["Acme"], valid_from="2024-01-01")
    acme = store.add("Acme", "office", "London", links=["London"], valid_from="2024-01-01")
    london = store.add("London", "country", "UK", valid_from="2024-01-01")
    store.add("Alice", "employer", "FutureCorp", links=["FutureCorp"], valid_from="2025-01-01")
    future = store.add("FutureCorp", "office", "Paris", valid_from="2024-01-01")
    results = HybridRetriever(store).search("Alice", as_of="2024-06-01", mode="graph")
    ids = {result.fact.id for result in results}
    assert ids == {alice.id, acme.id}
    assert london.id not in ids
    assert future.id not in ids


def test_keyword_fts_handles_user_query_syntax_as_words(store):
    coffee = store.add("Alice", "drink", "coffee", valid_from="2024-01-01")
    retriever = HybridRetriever(store)
    assert retriever.search('"coffee" OR -- (', mode="keyword")[0].fact.id == coffee.id
    assert retriever.search("*** \" -- ()", mode="keyword") == []
    assert retriever.search("unrelatedxyz", mode="keyword") == []
    assert retriever.search("unrelatedxyz", mode="semantic") == []


def test_offline_embedding_is_labeled_lexical(store):
    store.add("Northstar", "status", "invited", valid_from="2024-01-01")
    results = HybridRetriever(store).search("Northstar invitation", mode="semantic")
    assert results
    assert set(results[0].channels) == {"lexical_embedding"}
    assert HashingEmbedder.semantic is False


def test_real_semantic_provider_can_match_without_shared_words(store):
    fact = store.add("person", "drink", "espresso", valid_from="2024-01-01")

    class ControlledSemanticProvider:
        semantic = True
        name = "test-fixture"

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    results = HybridRetriever(store, ControlledSemanticProvider()).search("caffeine", mode="semantic")
    assert results[0].fact.id == fact.id
    assert results[0].channels == {"semantic": 1.0}


def test_entity_scope_and_bad_input(store):
    store.add("Alice", "drink", "coffee", valid_from="2024-01-01")
    bob = store.add("Bob", "drink", "coffee", valid_from="2024-01-01")
    retriever = HybridRetriever(store)
    assert [result.fact.id for result in retriever.search("coffee", entity="bob")] == [bob.id]
    for limit in (0, -1, 101, True, 1.5):
        with pytest.raises(ValueError):
            retriever.search("coffee", limit=limit)
    with pytest.raises(ValueError):
        retriever.search("coffee", mode="unknown")


def test_embedder_factory_environment_selection_is_lazy(store, monkeypatch):
    assert isinstance(create_embedder(), HashingEmbedder)
    monkeypatch.setenv("MEMORYOS_EMBEDDING_MODEL", " local-model-directory ")
    configured = create_embedder()
    assert isinstance(configured, SentenceTransformerEmbedder)
    assert configured.model_name == "local-model-directory"
    assert configured._model is None
    retriever = HybridRetriever(store)
    assert retriever.embedding_channel == "semantic"
    assert retriever.embedder.model_name == "local-model-directory"
    assert retriever.embedder._model is None
    explicit = create_embedder("another-model")
    assert explicit.model_name == "another-model"
    assert explicit._model is None
    override = HashingEmbedder()
    assert HybridRetriever(store, embedder=override).embedder is override
    monkeypatch.setenv("MEMORYOS_EMBEDDING_MODEL", " ")
    assert isinstance(create_embedder(), HashingEmbedder)
