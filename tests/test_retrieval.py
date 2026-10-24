"""Retrieval behavior at temporal, provenance and graph boundaries."""

import types

import pytest

import temporal_memory_os.store as store_module
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


# Matching rules pinned in 0.2.2


@pytest.mark.parametrize("query, entity", [
    ("Zurich", "José"), ("ZURICH", "José"), ("Jose", "José"), ("jose", "José"),
    ("Istanbul", "Ayşe"), ("Ayse", "Ayşe"), ("Zürich", "José"),
])
def test_keyword_search_ignores_accents(store, query, entity):
    store.add("José", "city", "Zürich", valid_from="2024-01-01")
    store.add("Ayşe", "city", "İstanbul", valid_from="2024-01-01")
    store.add("Bob", "city", "Lima", valid_from="2024-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    for mode in ("keyword", "hybrid"):
        assert [hit.fact.entity for hit in retriever.search(query, mode=mode)] == [entity], mode


@pytest.mark.parametrize("query, expected", [
    ("या", None),                       # "or", inside "vidya"
    ("दी", None),                       # inside "hindi"
    ("हिन", None),                 # inside "hindi"
    ("क", None),                             # the first letter of "kaam"
    ("काम", "Amit"),               # "kaam" itself, not "kom" or "kim"
    ("कोम", "Neha"),
    ("ไก่", "Somchai"),            # Thai: differs only in the tone mark
], ids=["ya", "di", "hin", "ka", "kaam", "kom", "thai-tone"])
def test_keyword_search_keeps_combining_marks_inside_words(store, query, expected):
    store.add("person", "subject", "विद्या", valid_from="2024-01-01")
    store.add("person", "language", "हिन्दी",
              valid_from="2024-01-01")
    store.add("Amit", "work", "काम", valid_from="2024-01-01")
    store.add("Neha", "work", "कोम", valid_from="2024-01-01")
    store.add("Sunil", "work", "किम", valid_from="2024-01-01")
    store.add("Somchai", "food", "ไก่", valid_from="2024-01-01")
    store.add("Malee", "food", "ไก๋", valid_from="2024-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    for mode in HybridRetriever.available_modes:
        hits = [hit.fact.entity for hit in retriever.search(query, mode=mode, limit=10)]
        assert hits == ([] if expected is None or mode == "graph" else [expected]), mode


def fixed_ids(monkeypatch, *numbers):
    """Give the next writes these IDs, in order."""
    ids = iter("{:032x}".format(number) for number in numbers)
    monkeypatch.setattr(store_module.uuid, "uuid4", lambda: types.SimpleNamespace(hex=next(ids)))


def test_entity_names_made_of_stopwords_still_count(store, monkeypatch):
    # The stopword-named entities get the largest IDs, so if their names were
    # dropped from the query, an ID tie-break would rank them last.
    fixed_ids(monkeypatch, 0xF3, 0xF2, 0xF1, 1, 2, 3, 4)
    for entity, attribute, value in (
            ("Will", "employer", "Acme"), ("WHO", "budget", "6.8 billion USD"),
            ("The Who", "lead singer", "Roger Daltrey"), ("Bob", "employer", "Globex"),
            ("UNICEF", "budget", "9 billion USD"), ("Queen", "lead singer", "Freddie Mercury"),
            ("Carol", "employer", "Initech")):
        store.add(entity, attribute, value, valid_from="2024-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    for query, entity in (("Will employer", "Will"), ("WHO budget", "WHO"),
                          ("The Who lead singer", "The Who")):
        for mode in ("keyword", "semantic", "hybrid"):
            [top] = retriever.search(query, mode=mode, limit=1)
            assert top.fact.entity == entity, (query, mode)


def test_a_query_of_only_stopwords_matches_nothing(store):
    store.add("Alex", "drink", "tea", valid_from="2024-01-01",
              evidence="What is the drink he has for the mornings? Just tea, she said about him.")
    retriever = HybridRetriever(store, HashingEmbedder())
    for query in ("what is the", "for the", "Who is it?", "just about him", "he and she",
                  "hers or his"):
        for mode in HybridRetriever.available_modes:
            assert retriever.search(query, mode=mode) == [], (query, mode)


def test_question_words_do_not_change_lexical_scores(store):
    store.add("Alex", "drink", "tea", valid_from="2024-01-01",
              evidence="what does he do with it")
    store.add("Alex", "snack", "chips", valid_from="2024-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())

    def scores(query):
        return [(hit.fact.value, hit.channels) for hit in retriever.search(
            query, mode="semantic", limit=10)]

    plain = scores("Alex")
    assert len(plain) == 2
    assert scores("what does Alex do") == plain == scores("What is Alex?")


def test_entity_filter_and_links_compare_normalized_names(store):
    composed, decomposed = "José", "José"
    jose = store.add(composed, "city", "Lima", valid_from="2024-01-01")
    acme = store.add("Acme", "founder", "someone", valid_from="2024-01-01", links=[decomposed])
    retriever = HybridRetriever(store, HashingEmbedder())
    for name in (decomposed, "JOSÉ", "ＪＯＳÉ", " josé "):
        assert [hit.fact.id for hit in retriever.search("Lima", entity=name)] == [jose.id], name
    graph = {hit.fact.id: hit.channels["graph"]
             for hit in retriever.search(composed, mode="graph")}
    assert graph == {jose.id: 1.0, acme.id: 0.5}


def test_graph_seeds_the_name_written_in_the_query(store):
    c = store.add("C", "paradigm", "procedural", valid_from="2024-01-01")
    sharp = store.add("C#", "paradigm", "object oriented", valid_from="2024-01-01")
    plus = store.add("C++", "paradigm", "multi paradigm", valid_from="2024-01-01")
    cup = store.add("☕", "meaning", "coffee", valid_from="2024-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    for query, expected in (("C paradigm", c), ("C# paradigm", sharp),
                            ("tell me about c++", plus), ("Which ☕ is this?", cup)):
        hits = retriever.search(query, mode="graph", limit=10)
        assert [(hit.fact.id, hit.channels["graph"]) for hit in hits] == [(expected.id, 1.0)], query
    # A symbol-only name is never seeded by a query that does not write it.
    assert cup.id not in {hit.fact.id for hit in retriever.search("coffee meaning",
                                                                   mode="graph", limit=10)}
