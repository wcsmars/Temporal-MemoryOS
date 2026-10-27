"""Retrieval behavior at temporal, provenance and graph boundaries."""

import math
import sys
import time
import types
from array import array

import pytest

import temporal_memory_os.retrieval as retrieval_module
import temporal_memory_os.store as store_module
from temporal_memory_os.demo import seed_demo
from temporal_memory_os.embeddings import (EmbeddingUnavailableError, HashingEmbedder,
                                           SentenceTransformerEmbedder, create_embedder,
                                           normalized_tokens)
from temporal_memory_os.models import Fact
from temporal_memory_os.retrieval import HybridRetriever, VectorCache
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


# Stopwords, accents, the vector cache and model output


def test_a_stopword_in_an_entity_name_counts_only_when_the_query_names_it(store):
    """"The Who" and "Bank of England" do not make "the" or "of" match every fact."""
    seed_demo(store)
    store.add("The Who", "genre", "rock", valid_from="2026-01-01")
    store.add("Bank of England", "founded", "1694", valid_from="2026-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    for mode in ("keyword", "semantic", "hybrid"):
        for question in ("What is the preferred drink for Alex?",
                         "What is a preferred drink of Alex?"):
            hits = retriever.search(question, mode=mode, limit=10, as_of="2026-10-10")
            assert {hit.fact.entity for hit in hits} == {"Alex"}, (mode, question)
        for query, entity in (("The Who genre", "The Who"),
                              ("When was the Bank of England founded?", "Bank of England")):
            [top] = retriever.search(query, mode=mode, limit=1, as_of="2026-10-10")
            assert top.fact.entity == entity, (mode, query)


class Facts:
    """A fact source that returns the same facts at every time."""

    def __init__(self, facts):
        self.facts = facts

    def snapshot(self, as_of=None, known_at=None, entity=None):
        return list(self.facts)


def names_written_out(query, facts):
    """The plain rule: every word of each entity name the query writes out word for word."""
    written = " {} ".format(" ".join(normalized_tokens(query)))
    keep = set()
    for fact in facts:
        name = normalized_tokens(fact.entity)
        if name and " {} ".format(" ".join(name)) in written:
            keep.update(name)
    return keep


def test_a_long_repetitive_query_and_many_names_with_stopwords_stay_fast(monkeypatch):
    """Finding the names a query writes out costs at most its length times the longest name,
    and gives the same results as the plain rule."""
    facts = [Fact("%032x" % number, "a " * 240 + "x%05d" % number, "genre", "rock",
                  valid_from="2025-01-01") for number in range(1000)]
    facts += [Fact("%032x" % (1000 + number), name, "founded", year, valid_from="2025-01-01")
              for number, (name, year) in enumerate([("The Who", "1964"),
                                                     ("Bank of England", "1694"),
                                                     ("The The", "1979")])]
    # Close to the API's 10,000-character limit; it writes out three of the names.
    query = "a " * 4700 + "x00007 the who of bank of england"
    retriever = HybridRetriever(Facts(facts), HashingEmbedder())
    for mode in ("keyword", "graph"):
        started = time.perf_counter()
        retriever.search(query, mode=mode)
        # Typically under 0.1 seconds; matching each name at every position took about 6.
        assert time.perf_counter() - started < 2.0, mode

    def results():
        return {mode: [(hit.fact.id, hit.score, hit.channels)
                       for hit in retriever.search(query, mode=mode, limit=20)]
                for mode in ("keyword", "semantic", "hybrid")}

    found = results()
    assert {hit[0] for hit in found["keyword"]} >= {"%032x" % 7, "%032x" % 1000, "%032x" % 1001}
    monkeypatch.setattr(retrieval_module, "_named_words", names_written_out)
    assert results() == found


def test_names_that_are_prefixes_of_each_other_stay_fast(monkeypatch):
    """Each written-out name is collected once, even when names extend one another.

    Four times the names and query cost about 15 times as much (the query's length times
    the longest name); collecting every name at every match cost about 50 times as much.
    The growth is compared rather than the time, so a slow machine cannot fail the test.
    """
    def setup(count):
        facts = [Fact("%032x" % number, " ".join(["the"] * (number + 1)), "genre", "rock",
                      valid_from="2025-01-01") for number in range(count)]
        return HybridRetriever(Facts(facts), HashingEmbedder()), "the " * (6 * count)

    def fastest(retriever, query):
        times = []
        for _ in range(3):
            started = time.perf_counter()
            retriever.search(query, mode="keyword", limit=5)
            times.append(time.perf_counter() - started)
        return min(times)

    retriever, query = setup(400)
    assert fastest(retriever, query) < 30 * fastest(*setup(100))
    found = [hit.fact.id for hit in retriever.search(query, mode="keyword", limit=5)]
    monkeypatch.setattr(retrieval_module, "_named_words", names_written_out)
    assert [hit.fact.id for hit in retriever.search(query, mode="keyword", limit=5)] == found


def test_graph_search_and_a_learned_model_never_look_for_names_in_the_query(store,
                                                                             monkeypatch):
    """Only keyword matching and word hashing drop stopwords, so only they need the names."""
    fact = store.add("The Who", "genre", "rock", valid_from="2025-01-01")

    class Flat:
        semantic = True
        name = "flat-fixture"

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    def never(query, facts):
        raise AssertionError("names were looked for")

    monkeypatch.setattr(retrieval_module, "_named_words", never)
    graph = HybridRetriever(store, HashingEmbedder()).search("The Who", mode="graph")
    assert [hit.fact.id for hit in graph] == [fact.id]
    learned = HybridRetriever(store, Flat()).search("The Who", mode="semantic")
    assert [hit.fact.id for hit in learned] == [fact.id]


@pytest.mark.parametrize("query", ["Alex habite à Lyon", "Alex thé", "Alex À"])
def test_keyword_stopwords_are_recognized_with_accents(store, query):
    """A query word that is a stopword once its accent is folded does not match every fact."""
    store.add("Alex", "home city", "Lyon", valid_from="2024-01-01",
              evidence="Alex moved to Lyon.")
    store.add("Bob", "employer", "Acme", valid_from="2024-01-01",
              evidence="Bob has a job at the lab.")
    retriever = HybridRetriever(store, HashingEmbedder())
    for mode in ("keyword", "hybrid"):
        assert [hit.fact.entity for hit in retriever.search(query, mode=mode)] == ["Alex"], mode


class CountingHashing(HashingEmbedder):
    """The offline lexical embedder, recording every batch it is asked to embed."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return super().embed(texts)


def test_a_store_larger_than_the_vector_cache_still_reuses_most_vectors(store):
    """Only the documents that did not fit in the cache are embedded again."""
    for name in ("alpha", "beta", "gamma"):
        store.add(name, "drink", "tea", valid_from="2025-01-01")
    embedder = CountingHashing()
    retriever = HybridRetriever(store, embedder)
    retriever.cache_limit = 2
    first = retriever.search("tea", mode="semantic")
    second = retriever.search("tea", mode="semantic")
    assert [len(batch) for batch in embedder.calls] == [4, 2]  # The query and one document.
    assert [hit.fact.id for hit in first] == [hit.fact.id for hit in second]


def test_the_vector_cache_drops_the_least_recently_used_entry():
    """A vector read since it was stored outlives one that was not."""
    entry = (2, None, array("d", [1.0, 0.0]), 1.0, frozenset({"tea"}))
    cache = VectorCache()
    cache.store({("m", "a"): entry, ("m", "b"): entry}, limit=2)
    assert cache.get(("m", "a")) is not None
    cache.store({("m", "c"): entry}, limit=2)
    assert len(cache) == 2
    assert cache.get(("m", "b")) is None
    assert cache.get(("m", "a")) is not None and cache.get(("m", "c")) is not None


class Marked:
    """A learned-model stand-in: a text containing a marker gets that marker's vector."""

    semantic = True

    def __init__(self, markers, name="marked-fixture"):
        self.name = name
        self.markers = markers
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [next((vector for marker, vector in self.markers.items() if marker in text),
                     [0.6, 0.8]) for text in texts]


@pytest.mark.parametrize("bad", [[math.nan, 0.0], [math.inf, 0.0], [1e200, 1e200], ["x", 0.0]],
                         ids=["nan", "inf", "norm-overflow", "not-a-number"])
def test_one_document_the_model_cannot_embed_does_not_break_search(store, bad):
    """A record whose vector is unusable is skipped; the other records are still found."""
    tea = store.add("Alex", "drink", "tea", valid_from="2025-01-01")
    store.add("Bob", "note", "overflow", valid_from="2025-01-01")
    embedder = Marked({"overflow": bad})
    retriever = HybridRetriever(store, embedder)
    for mode in ("semantic", "hybrid"):
        hits = retriever.search("Alex drink", mode=mode)
        assert [hit.fact.id for hit in hits] == [tea.id], mode
    # Not cached, so a later search tries the unusable record again.
    assert len(embedder.calls[1]) == 2


@pytest.mark.parametrize("vectors, message", [
    ([[math.nan, 0.0], [1.0, 0.0]], "finite"),
    ([["x", 0.0], [1.0, 0.0]], "finite"),
    ([[1.0, 0.0], [1.0, 0.0, 0.0]], "dimensions"),
    ([[1.0, 0.0]], "wrong number"),
], ids=["nan-query", "text-query", "dimensions", "count"])
def test_unusable_model_output_is_an_unavailable_model_not_invalid_input(store, vectors,
                                                                           message):
    """A bad query vector, vector count or size is the model's fault, not the request's."""
    store.add("person", "drink", "tea", valid_from="2025-01-01")

    class Provider:
        semantic = True
        name = "broken"

        def embed(self, texts):
            return vectors

    with pytest.raises(EmbeddingUnavailableError, match=message) as caught:
        HybridRetriever(store, Provider()).search("tea", mode="semantic")
    assert not isinstance(caught.value, ValueError)


class MissingInnerDependency:
    """An import hook that fails the way a mismatched dependency of the package does."""

    def find_spec(self, name, path=None, target=None):
        if name == "sentence_transformers":
            raise ModuleNotFoundError("No module named 'huggingface_hub'", name="huggingface_hub")
        return None


@pytest.mark.parametrize("how", ["inner-module", "missing-name"])
def test_a_broken_sentence_transformers_install_reports_its_cause(monkeypatch, how):
    """Only a missing package gets the install hint; any other import error is named."""
    if how == "inner-module":
        monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
        monkeypatch.setattr(sys, "meta_path", [MissingInnerDependency(), *sys.meta_path])
        cause = r"ModuleNotFoundError: No module named 'huggingface_hub'"
    else:
        monkeypatch.setitem(sys.modules, "sentence_transformers",
                            types.ModuleType("sentence_transformers"))
        cause = r"ImportError: cannot import name 'SentenceTransformer'"
    with pytest.raises(EmbeddingUnavailableError,
                       match=r"Could not import sentence-transformers \(" + cause) as caught:
        SentenceTransformerEmbedder("local-model").embed(["text"])
    assert "pip install" not in str(caught.value)
