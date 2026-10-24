"""Retrieval behavior that mutation testing showed no other test checked.

Covers vector-cache limits and layout, argument validation, embedding inputs
and scores, keyword fields, graph seeding and tie order, and citations.
"""

from array import array

import pytest

from temporal_memory_os.embeddings import HashingEmbedder
from temporal_memory_os.models import Fact
from temporal_memory_os.retrieval import _QueryWords, HybridRetriever, VectorCache
from temporal_memory_os.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "store") as memory:
        yield memory


# ==========================================================================
# Vector cache, argument validation, embedding inputs, scores and keywords
# ==========================================================================


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    monkeypatch.delenv("MEMORYOS_EMBEDDING_MODEL", raising=False)


class Learned:
    """Stands in for a learned model: a text gets the vector of the first marker it contains."""

    semantic = True

    def __init__(self, markers=None, default=(1.0, 0.0), name="learned-fixture"):
        self.name = name
        self.markers = dict(markers or {})
        self.default = default
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [list(next((vector for marker, vector in self.markers.items() if marker in text),
                          self.default)) for text in texts]


class CountingHashing(HashingEmbedder):
    """The offline lexical embedder, recording every batch it is asked to embed."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return super().embed(texts)


class NamelessPointingUp:
    """A learned model with no ``name`` attribute."""

    semantic = True

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class NamelessPointingRight:
    """A different learned model, also without a ``name`` attribute."""

    semantic = True

    def embed(self, texts):
        return [[0.0, 1.0] for _ in texts]


class Unlabeled:
    """A model that does not say whether it is semantic."""

    name = "unlabeled-fixture"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def entry(*values):
    return (len(values), None, array("d", values), 1.0, frozenset({"tea"}))


# VectorCache.store


def test_a_batch_exactly_the_size_of_the_cache_limit_is_cached():
    """A batch as large as the limit still fits in the cache."""
    cache = VectorCache()
    cache.store({("m", "a"): entry(1.0, 0.0), ("m", "b"): entry(0.0, 1.0)}, limit=2)
    assert len(cache) == 2
    assert cache.get(("m", "a")) is not None and cache.get(("m", "b")) is not None


def test_filling_the_cache_exactly_to_its_limit_keeps_earlier_entries():
    """Entries are only evicted when a batch would push the cache past its limit."""
    cache = VectorCache()
    cache.store({("m", "a"): entry(1.0, 0.0)}, limit=2)
    cache.store({("m", "b"): entry(0.0, 1.0)}, limit=2)
    assert len(cache) == 2
    assert cache.get(("m", "a")) is not None


# _entry: cached vector layout


def test_only_vectors_that_are_mostly_zeros_are_cached_sparsely(store):
    """Fewer than half non-zero values are stored sparsely; half or more are stored densely."""
    sparse = store.add("person", "drink", "tea", valid_from="2025-01-01", evidence="sparsemarker")
    dense = store.add("person", "snack", "toast", valid_from="2025-01-01", evidence="densemarker")
    embedder = Learned({
        "sparsemarker": (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "densemarker": (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }, default=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    retriever = HybridRetriever(store, embedder)
    assert len(retriever.search("person", mode="semantic")) == 2
    dimensions, indices, values, _, _ = retriever.vectors.get(
        (embedder.name, HybridRetriever._text(sparse)))
    assert dimensions == 10 and list(indices) == [0, 1, 2, 3] and list(values) == [1.0] * 4
    dimensions, indices, values, _, _ = retriever.vectors.get(
        (embedder.name, HybridRetriever._text(dense)))
    assert dimensions == 10 and indices is None and list(values) == [1.0] * 5 + [0.0] * 5


# HybridRetriever.search: argument validation


@pytest.mark.parametrize("arguments, name", [
    ({"query": 42}, "query"),
    ({"query": "tea", "limit": 0}, "limit"),
    ({"query": "tea", "mode": "fuzzy"}, "mode"),
    ({"query": "tea", "entity": "   "}, "entity"),
], ids=["query", "limit", "mode", "entity"])
def test_invalid_search_arguments_are_reported_by_name(store, arguments, name):
    """The error for a bad search argument (an HTTP 422 detail) names that argument."""
    with pytest.raises(ValueError) as caught:
        HybridRetriever(store, HashingEmbedder()).search(**arguments)
    assert name in str(caught.value)


@pytest.mark.parametrize("threshold", [1.5, -1.01, float("nan")])
def test_out_of_range_semantic_threshold_is_reported_by_name(store, threshold):
    """A threshold outside [-1, 1] is rejected with an error naming semantic_threshold."""
    with pytest.raises(ValueError) as caught:
        HybridRetriever(store, HashingEmbedder(), semantic_threshold=threshold)
    assert "semantic_threshold" in str(caught.value)


# HybridRetriever.search: queries without word tokens


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_a_query_without_words_never_reaches_a_lexical_embedder(store, mode):
    """Word hashing cannot match a symbol-only query, so it is not embedded at all."""
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    embedder = CountingHashing()
    retriever = HybridRetriever(store, embedder)
    assert retriever.search("☕ !!", mode=mode) == []
    assert embedder.calls == []
    assert len(retriever.vectors) == 0


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_a_model_that_does_not_declare_semantic_is_treated_as_learned(store, mode):
    """Without a ``semantic`` attribute a model is learned, so it can match a symbol-only query."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    [hit] = HybridRetriever(store, Unlabeled()).search("☕", mode=mode)
    assert hit.fact.id == fact.id
    assert hit.channels == {"semantic": 1.0}


# HybridRetriever.search: fusion and reported scores


def test_channel_scores_are_rounded_to_eight_decimal_places(store):
    """Raw channel scores are reported rounded to eight decimals, like the fused score."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01", evidence="slanted")
    embedder = Learned({"slanted": (1.0, 2.0)}, default=(1.0, 0.0))
    [hit] = HybridRetriever(store, embedder).search("caffeine", mode="semantic")
    assert hit.fact.id == fact.id
    assert hit.channels == {"semantic": 0.4472136}


# HybridRetriever._text: what document vectors are computed from


def test_every_tag_is_its_own_word_in_the_embedded_document(store):
    """With several tags, each tag can be matched by the embedding channel."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01",
                     tags=["morning", "ritual"])
    retriever = HybridRetriever(store, HashingEmbedder())
    for word in ("morning", "ritual"):
        assert [hit.fact.id for hit in retriever.search(word, mode="semantic")] == [fact.id]


def test_every_link_is_its_own_word_in_the_embedded_document(store):
    """With several links, each linked name can be matched by the embedding channel."""
    fact = store.add("person", "trip", "summer", valid_from="2025-01-01",
                     links=["Kyoto", "Osaka"])
    retriever = HybridRetriever(store, HashingEmbedder())
    for word in ("kyoto", "osaka"):
        assert [hit.fact.id for hit in retriever.search(word, mode="semantic")] == [fact.id]


# HybridRetriever._keyword


def test_keyword_search_matches_evidence_tags_links_and_topic(store):
    """Words from evidence, tags, links and topic are separate keyword-searchable words."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01",
                     evidence="bought at the market", tags=["espresso"], links=["Kyoto"],
                     topic="travel")
    store.add("other", "drink", "water", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    for word in ("market", "espresso", "kyoto", "travel"):
        assert [hit.fact.id for hit in retriever.search(word, mode="keyword")] == [fact.id], word


# HybridRetriever._embedding


def test_models_without_a_name_do_not_share_cached_vectors(store):
    """Two nameless models of different classes never reuse each other's document vectors."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    cache = VectorCache()
    [first] = HybridRetriever(store, NamelessPointingUp(), vector_cache=cache).search(
        "tea", mode="semantic")
    assert first.channels == {"semantic": 1.0}
    [second] = HybridRetriever(store, NamelessPointingRight(), vector_cache=cache).search(
        "tea", mode="semantic")
    assert second.fact.id == fact.id
    assert second.channels == {"semantic": 1.0}


def test_a_learned_model_reads_the_whole_question(store):
    """A semantic model is given the query exactly as asked, question words included."""
    store.add("Alice", "employer", "Acme", valid_from="2025-01-01")
    embedder = Learned()
    HybridRetriever(store, embedder).search("Where does Alice work?", mode="semantic")
    assert embedder.calls[0][0] == "Where does Alice work?"


def test_a_cached_vector_of_another_size_is_reported_as_a_dimension_mismatch(store):
    """A model whose vector size changes between calls is reported as inconsistent dimensions."""
    store.add("person", "drink", "tea", valid_from="2025-01-01")
    embedder = Learned(default=(1.0, 0.0))
    retriever = HybridRetriever(store, embedder)
    assert len(retriever.search("tea", mode="semantic")) == 1
    embedder.default = (1.0, 0.0, 0.0)
    with pytest.raises(ValueError) as caught:
        retriever.search("tea", mode="semantic")
    assert "dimension" in str(caught.value)


def test_documents_pointing_the_same_way_as_the_query_tie(store):
    """A cosine that rounding pushes above 1 is clamped to 1, so it ties with an exact 1."""
    one = store.add("person", "drink", "tea", valid_from="2025-01-01", evidence="unitmarker")
    three = store.add("person", "snack", "toast", valid_from="2025-01-01", evidence="triplemarker")
    # In floating point, cos([1,1,1], [1,1,1]) comes out just above 1 and
    # cos([1,1,1], [3,3,3]) exactly 1.
    embedder = Learned({"unitmarker": (1.0, 1.0, 1.0), "triplemarker": (3.0, 3.0, 3.0)},
                       default=(1.0, 1.0, 1.0))
    hits = HybridRetriever(store, embedder).search("caffeine", mode="semantic")
    assert [hit.fact.id for hit in hits] == sorted([one.id, three.id])
    assert hits[0].score == hits[1].score
    assert [hit.channels for hit in hits] == [{"semantic": 1.0}, {"semantic": 1.0}]


# ==========================================================================
# Graph seeding, graph tie order and citations
# ==========================================================================


def graph_scores(store, query):
    """Graph channel scores by fact ID for one query."""
    retriever = HybridRetriever(store, HashingEmbedder())
    return {hit.fact.id: hit.channels["graph"]
            for hit in retriever.search(query, mode="graph", limit=100)}


def note(fact_id, entity, links=()):
    """A minimal valid fact with a fixed ID, for calling the graph ranking directly."""
    return Fact(id=fact_id, entity=entity, attribute="note", value="x",
                valid_from="2025-01-01", links=tuple(links))


# Graph channel: which entities seed


def test_names_with_different_words_all_seed_when_only_one_is_written_out(store):
    """A seed whose words appear out of order is kept alongside a name written out verbatim."""
    alice = store.add("Alice", "employer", "Acme", valid_from="2025-01-01")
    acme = store.add("Acme Corp", "city", "Riverton", valid_from="2025-01-01")
    assert graph_scores(store, "Alice corp acme") == {alice.id: 1.0, acme.id: 1.0}


def test_names_sharing_words_all_seed_when_none_is_written_out(store):
    """When no punctuation variant is written out, every variant sharing the words seeds."""
    sharp = store.add("C#", "paradigm", "object oriented", valid_from="2025-01-01")
    plus = store.add("C++", "paradigm", "multi paradigm", valid_from="2025-01-01")
    assert graph_scores(store, "c programming") == {sharp.id: 1.0, plus.id: 1.0}


def test_name_written_out_beats_its_only_punctuation_variant(store):
    """With exactly two names sharing words, only the one written in the query seeds."""
    c = store.add("C", "paradigm", "procedural", valid_from="2025-01-01")
    store.add("C++", "paradigm", "multi paradigm", valid_from="2025-01-01")
    assert graph_scores(store, "C paradigm") == {c.id: 1.0}


# Graph channel: ranking order


def test_graph_ranking_breaks_score_ties_by_fact_id_not_snapshot_order():
    """The graph ranking lists seeds before neighbours, each tie ordered by fact ID."""
    facts = [
        note("d" * 32, "Acme"),
        note("b" * 32, "Acme"),
        note("c" * 32, "Beacon", links=["Acme"]),
        note("a" * 32, "Beacon", links=["Acme"]),
    ]
    assert HybridRetriever._graph("Acme", facts) == [
        ("b" * 32, 1.0), ("d" * 32, 1.0), ("a" * 32, 0.5), ("c" * 32, 0.5)]


# Citations


def test_citation_names_the_file_a_moved_record_now_lives_in(store):
    """A record moved to another Markdown file is cited at its actual path."""
    tea = store.add("person", "drink", "tea", valid_from="2025-01-01", topic="profile")
    original = store.path_for(tea.id)
    moved = store.memory_dir / "notes" / "morning-drink.md"
    moved.parent.mkdir()
    original.rename(moved)
    assert store.refresh() is True
    [hit] = HybridRetriever(store, HashingEmbedder()).search("tea")
    assert hit.fact.id == tea.id
    assert hit.citation == "memory/notes/morning-drink.md"


@pytest.mark.parametrize("exposed", ["root", "path_for"])
def test_source_with_only_one_location_attribute_gets_the_canonical_citation(store, exposed):
    """A fact source exposing a root or a path lookup, but not both, is cited canonically."""
    tea = store.add("person", "drink", "tea", valid_from="2025-01-01", topic="profile")

    class Partial:
        """Delegates snapshots to the store and exposes one location attribute."""

        def snapshot(self, **arguments):
            return store.snapshot(**arguments)

    source = Partial()
    setattr(source, exposed, getattr(store, exposed))
    [hit] = HybridRetriever(source, HashingEmbedder()).search("tea")
    assert hit.citation == "memory/profile/{}.md".format(tea.id)


def test_a_name_without_words_is_never_written_out():
    """An empty or blank name matches no position in the query."""
    words = _QueryWords("a b c")
    assert not words.write("") and not words.write("   ")
    assert words.write("b c") and not words.write("c d")


# ==========================================================================
# Graph seeding around quotes, brackets and possessives
# ==========================================================================


def graph_entities(store, query):
    """Entities of the facts the graph channel returns for ``query``."""
    retriever = HybridRetriever(store, HashingEmbedder())
    return {hit.fact.entity for hit in retriever.search(query, mode="graph", limit=20)}


@pytest.mark.parametrize("query", ["what is Plan S", "plan s", "Plan S.", "Plan S's goals"])
def test_a_one_letter_s_word_in_a_query_still_seeds_the_graph(store, query):
    """A query holding the word "s" on its own is searched, not rejected."""
    plan = store.add("Plan S", "kind", "open access initiative", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    assert [hit.fact.id for hit in retriever.search(query, mode="graph")] == [plan.id]
    assert retriever.search(query)[0].fact.id == plan.id


@pytest.mark.parametrize("query", ["ask the C++'ers", "C++es", "C++'d"])
def test_letters_after_a_symbol_name_mean_it_is_not_written_out(store, query):
    """A name followed by more letters is not written out, so names sharing its words all seed."""
    for name in ("C", "C++"):
        store.add(name, "kind", "language", valid_from="2025-01-01")
    assert graph_entities(store, query) == {"C", "C++"}


def test_a_symbol_name_written_with_a_possessive_still_wins(store):
    """A possessive "'s" after a name still counts as writing the name out."""
    for name in ("C", "C++"):
        store.add(name, "kind", "language", valid_from="2025-01-01")
    assert graph_entities(store, "C++'s templates") == {"C++"}
    assert graph_entities(store, "C++’s templates") == {"C++"}
    assert graph_entities(store, "C's pointers") == {"C"}


def test_a_name_that_starts_with_an_opening_mark_is_written_out_by_itself(store):
    """A name beginning with "¡" or "(" counts as written when the query writes it exactly."""
    store.add("¡Forward, Russia!", "kind", "band", valid_from="2025-01-01")
    store.add("Forward Russia", "kind", "campaign", valid_from="2025-01-01")
    store.add("(+)", "kind", "operator", valid_from="2025-01-01")
    # Both names are written out (the plain one with marks around its words).
    assert graph_entities(store, "tickets for ¡Forward, Russia! tonight") == {
        "¡Forward, Russia!", "Forward Russia"}
    assert graph_entities(store, "forward russia tonight") == {"Forward Russia"}
    assert graph_entities(store, "what does (+) mean") == {"(+)"}
