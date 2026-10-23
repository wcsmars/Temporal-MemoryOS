"""Exact scores, extraction rules, boundaries and clock handling.

Each test here pins a detail that mutation testing showed could change
without any other test noticing.
"""

import math
import time
import types
from datetime import datetime, timedelta, timezone

import pytest

import temporal_memory_os.store as store_module
from temporal_memory_os.embeddings import HashingEmbedder
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.models import Fact, normalize_time, validate_topic
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "store") as memory:
        yield memory


@pytest.fixture
def hong_kong_clock(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Hong_Kong")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


# Retrieval scores

def test_rank_fusion_score_is_the_documented_formula(store):
    fact = store.add("Acme", "industry", "robotics", valid_from="2025-01-01")
    [hit] = HybridRetriever(store, HashingEmbedder()).search("Acme", as_of="2025-02-01")
    assert hit.fact.id == fact.id
    # Rank 1 in keyword, embedding and graph: (1 + 1 + 0.7) / (60 + 1).
    assert hit.score == round(2.7 / 61, 8)
    assert hit.channels["graph"] == 1.0
    assert hit.channels["keyword"] > 0  # BM25 is reported as a positive relevance.
    assert 0 < hit.channels["lexical_embedding"] <= 1


def test_semantic_channel_ranks_by_cosine_and_applies_a_strict_threshold(store):
    vectors = {
        "q": [1.0, 0.0],
        "p90": [1.8, 2 * math.sqrt(0.19)],  # cosine 0.9 with norm 2: scores are normalized.
        "q50": [0.5, math.sqrt(0.75)],      # cosine 0.5
        "r25": [0.25, math.sqrt(0.9375)],   # cosine exactly 0.25: not above the threshold.
        "s90": [-0.9, math.sqrt(0.19)],     # negative similarity
    }
    for cosine in (0.8, 0.7, 0.6):  # More ranks make an accidental pass by ID order unlikely.
        vectors["p{}".format(int(cosine * 100))] = [cosine, math.sqrt(1 - cosine * cosine)]
    facts = {value: store.add(value, "code", value, valid_from="2025-01-01")
             for value in ("s90", "r25", "q50", "p60", "p70", "p80", "p90")}

    class Provider:
        semantic = True
        name = "controlled"

        def embed(self, texts):
            return [vectors["q"] if text == "q" else vectors[text.split()[0]] for text in texts]

    hits = HybridRetriever(store, Provider()).search("q", as_of="2025-02-01", mode="semantic",
                                                     limit=10)
    order = ("p90", "p80", "p70", "p60", "q50")  # Not r25 (equal to the threshold) or s90.
    assert [hit.fact.id for hit in hits] == [facts[name].id for name in order]
    assert [hit.channels["semantic"] for hit in hits] == pytest.approx([0.9, 0.8, 0.7, 0.6, 0.5])
    assert [hit.score for hit in hits] == [round(1 / (61 + rank), 8) for rank in range(5)]


def test_providers_without_a_semantic_flag_are_treated_as_semantic(store):
    store.add("person", "drink", "tea", valid_from="2025-01-01")

    class Unlabelled:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    [hit] = HybridRetriever(store, Unlabelled()).search("anything", mode="semantic")
    assert hit.channels == {"semantic": 1.0}


def test_lexical_embeddings_keep_weak_overlaps_but_drop_hash_collisions(store):
    words = " ".join("word{}".format(index) for index in range(40))
    weak = store.add("Alex", "drink", "tea", valid_from="2025-01-01", evidence=words)
    [hit] = HybridRetriever(store, HashingEmbedder()).search("tea", mode="semantic")
    assert hit.fact.id == weak.id
    assert 0 < hit.channels["lexical_embedding"] < 0.25  # Below the learned-model threshold.
    # "w253" lands on the same hashed slot and sign as "tea" in 32 dimensions, so
    # cosine similarity is positive although the texts share no word.
    store.add("w253", "w253", "w253", valid_from="2025-01-01")
    hits = HybridRetriever(store, HashingEmbedder(dimensions=32)).search("tea", mode="semantic")
    assert [hit.fact.id for hit in hits] == [weak.id]


def test_hashing_embedder_output_is_pinned():
    embedder = HashingEmbedder(dimensions=64)
    [vector] = embedder.embed(["decaf tea tea"])
    assert {index: round(value, 6) for index, value in enumerate(vector) if value} == {
        2: round(1 / math.sqrt(5), 6), 4: round(-2 / math.sqrt(5), 6)}
    assert embedder.embed([""]) == [[0.0] * 64]
    assert embedder.embed(["applications"]) == embedder.embed(["application"])
    for text in ("tea", "Acme application status", "完 😀"):
        assert math.isclose(math.sqrt(sum(v * v for v in embedder.embed([text])[0])), 1.0)
    HashingEmbedder(dimensions=32)
    with pytest.raises(ValueError):
        HashingEmbedder(dimensions=31)


def test_graph_follows_links_in_both_directions_for_one_hop(store):
    acme = [store.add("Acme", attribute, "x", valid_from="2025-01-01", links=["Delta"])
            for attribute in ("industry", "size", "founded")]
    beacon = [store.add("Beacon project", attribute, "Acme", valid_from="2025-01-01",
                        links=["Acme"]) for attribute in ("owner", "stage")]
    delta = store.add("Delta", "city", "Riverton", valid_from="2025-01-01")
    store.add("Echo", "city", "Lakeside", valid_from="2025-01-01", links=["Delta"])  # Two hops.
    hits = HybridRetriever(store, HashingEmbedder()).search("Acme", as_of="2025-02-01",
                                                             mode="graph", limit=10)
    expected = {**{f.id: 1.0 for f in acme}, **{f.id: 0.5 for f in beacon}, delta.id: 0.5}
    assert {hit.fact.id: hit.channels["graph"] for hit in hits} == expected
    # Seeds first, then neighbours; ties are broken by fact ID.
    assert [hit.fact.id for hit in hits] == sorted(expected, key=lambda i: (-expected[i], i))


def test_search_limits(store):
    for index in range(7):
        store.add("tea{}".format(index), "drink", "tea", valid_from="2025-01-01")
    retriever = HybridRetriever(store, HashingEmbedder())
    assert len(retriever.search("tea", as_of="2025-02-01")) == 5
    assert len(retriever.search("tea", as_of="2025-02-01", limit=100)) == 7
    for threshold in (-1.0, 1.0):
        HybridRetriever(store, HashingEmbedder(), semantic_threshold=threshold)


# Store details

def test_history_filters_by_attribute_and_knowledge_time(store):
    store.add("person", "drink", "tea", valid_from="2025-01-01", recorded_at="2025-01-01")
    store.add("person", "city", "Riverton", valid_from="2025-01-01", recorded_at="2025-01-01")
    store.add("person", "drink", "coffee", valid_from="2025-02-01", recorded_at="2025-03-01")
    assert [f.value for f in store.history("person", attribute="drink")] == ["tea", "coffee"]
    assert [f.value for f in store.history("person", attribute="drink",
                                           known_at="2025-02-15")] == ["tea"]


def test_same_microsecond_writes_stay_strictly_ordered(store, monkeypatch, hong_kong_clock):
    instant = "2025-06-01T12:00:00.000000Z"
    first = store.add("person", "drink", "tea", valid_from="2025-01-01", recorded_at=instant)
    real = store_module.normalize_time
    monkeypatch.setattr(store_module, "normalize_time",
                        lambda value=None: real(instant if value is None else value))
    second = store.add("person", "city", "Riverton", valid_from="2025-01-01")
    assert first.recorded_at == instant
    assert second.recorded_at == "2025-06-01T12:00:00.000001Z"


def test_a_generated_id_collision_is_refused(store, monkeypatch):
    existing = store.add("person", "drink", "tea", valid_from="2025-01-01")
    monkeypatch.setattr(store_module.uuid, "uuid4", lambda: types.SimpleNamespace(hex=existing.id))
    with pytest.raises(ValueError, match="Duplicate memory id"):
        store.add("person", "city", "Riverton", valid_from="2025-01-01")
    assert store.get(existing.id).value == "tea"


def test_index_rows_without_files_force_a_rebuild(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    store.path_for(fact.id).unlink()
    # An index that lists a record but no file signature at all (as a version 1
    # index would) must not be trusted just because the folder is empty.
    store.connection.execute("DELETE FROM sources")
    assert store.refresh() is True
    assert store.all_facts() == []


def test_schema_version_and_parsed_record_reuse(store):
    assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 2
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    assert store.get(fact.id) is store.get(fact.id)


def test_markdown_record_format_is_stable():
    fact = Fact(id="a" * 32, entity="person", attribute="drink", value="茶",
                valid_from="2025-01-01", recorded_at="2025-01-02", tags=["x"])
    assert MemoryStore._render(fact) == (
        "# person · drink\n\n"
        "The JSON block below is canonical. Edits are picked up the next time the store opens.\n\n"
        "<!-- memoryos:fact:v1 -->\n```json\n{\n"
        '  "id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",\n  "entity": "person",\n'
        '  "attribute": "drink",\n  "value": "茶",\n'
        '  "valid_from": "2025-01-01T00:00:00.000000Z",\n  "valid_to": null,\n'
        '  "recorded_at": "2025-01-02T00:00:00.000000Z",\n  "source": "manual",\n'
        '  "evidence": "",\n  "confidence": 1.0,\n  "tags": [\n    "x"\n  ],\n'
        '  "links": [],\n  "topic": "general",\n  "supersedes": null,\n  "deleted": false\n'
        "}\n```\n")


# Time

def test_default_time_is_utc_now_whatever_the_local_zone(hong_kong_clock):
    now = datetime.fromisoformat(normalize_time().replace("Z", "+00:00"))
    assert abs(now - datetime.now(timezone.utc)) < timedelta(seconds=5)


@pytest.mark.parametrize("text", ["20250101", "2025-01", "2025", " ", "2025/01/01"])
def test_only_dashed_iso_dates_are_accepted(text):
    with pytest.raises(ValueError):
        normalize_time(text)


def test_topic_length_boundary():
    longest = "/".join(["a" * 128, "b" * 128, "c" * 128, "d" * 125])
    assert len(longest) == 512 and validate_topic(longest) == longest
    with pytest.raises(ValueError):
        validate_topic(longest + "d")


# Extraction rules

@pytest.mark.parametrize("line, expected", [
    ("Application Acme is pending.", ("Acme application", "status", "pending", 0.85)),
    ("my application TO Beacon is Interview Invited!",
     ("Beacon application", "status", "Interview Invited", 0.85)),
    ("My Favourite Colour is Blue.", ("user", "favourite colour", "Blue", 0.85)),
    ("i PREFER green tea", ("user", "preference", "green tea", 0.75)),
    ("I like decaf coffee!", ("user", "preference", "decaf coffee", 0.75)),
    ("Alex | home city | Riverton", ("Alex", "home city", "Riverton", 1.0)),
])
def test_each_extraction_rule(store, line, expected):
    [proposal] = ProposalQueue(store).extract(line, source="synthetic:rules",
                                              observed_at="2025-01-01")
    assert (proposal.entity, proposal.attribute, proposal.value, proposal.confidence) == expected
    assert proposal.evidence == line and proposal.kind == "add"


@pytest.mark.parametrize("line", ["# My name is Alex", "# a | b | c", "Unsupported prose.", "   "])
def test_comments_and_prose_are_skipped_without_stopping(store, line):
    proposals = ProposalQueue(store).extract("x\n{}\nmy name is Alex".format(line),
                                             source="synthetic:rules")
    assert [(p.entity, p.attribute, p.value) for p in proposals] == [("user", "name", "Alex")]


@pytest.mark.parametrize("line", ["a | b", "a | b | c | 2025-01-01 | e", "a |  | c"])
def test_malformed_structured_lines_are_errors(store, line):
    with pytest.raises(ValueError, match="Structured facts"):
        ProposalQueue(store).extract(line, source="synthetic:rules")


def test_extraction_size_boundary(store):
    queue = ProposalQueue(store)
    assert queue.extract("#\n" * 500_000, source="synthetic:size") == []  # Exactly 1,000,000.
    with pytest.raises(ValueError, match="1 MB"):
        queue.extract("#\n" * 500_000 + "#", source="synthetic:size")


def test_classification_only_considers_the_same_entity_and_attribute(store):
    store.add("Beacon application", "status", "pending", valid_from="2025-01-01")
    tombstoned = store.add("Acme application", "owner", "Alex", valid_from="2025-01-01")
    store.retract(tombstoned.id)
    early = store.add("Acme application", "status", "draft", valid_from="2024-12-01")
    store.retract(early.id)
    queue = ProposalQueue(store)
    [proposal] = queue.extract("Acme application | status | pending | 2025-01-01",
                               source="synthetic:classify")
    # Another entity's status, another attribute's retraction on the same day,
    # and this attribute's earlier retraction are all irrelevant: this is new.
    assert (proposal.kind, proposal.target_id, proposal.prior_value) == ("add", None, None)


def test_update_proposal_goes_stale_when_a_newer_predecessor_appears(store):
    store.add("Acme application", "status", "pending", valid_from="2025-01-01")
    queue = ProposalQueue(store)
    [proposal] = queue.extract("Acme application | status | invited | 2025-01-10",
                               source="synthetic:stale")
    assert proposal.kind == "update" and proposal.prior_value == "pending"
    store.add("Acme application", "status", "on hold", valid_from="2025-01-05")
    with pytest.raises(ValueError, match="Memory changed"):
        queue.approve(proposal.id)


def test_proposals_are_listed_in_creation_order(store):
    queue = ProposalQueue(store)
    lines = ["Entity{} | status | v{}".format(index, index) for index in range(6)]
    created = queue.extract("\n".join(lines), source="synthetic:order", observed_at="2025-01-01")
    listed = queue.list()
    assert [p.id for p in listed] == [p.id for p in created]
    assert [p.entity for p in listed] == ["Entity{}".format(index) for index in range(6)]
    assert all(datetime.fromisoformat(p.created_at) for p in listed)


def test_a_batch_lists_in_line_order_even_within_one_clock_tick(store, monkeypatch):
    import temporal_memory_os.extraction as extraction
    monkeypatch.setattr(extraction, "_now", lambda: "2025-01-01T00:00:00.000000+00:00")
    queue = ProposalQueue(store)
    lines = ["Entity{} | status | v{}".format(index, index) for index in range(5)]
    created = queue.extract("\n".join(lines), source="synthetic:tick", observed_at="2025-01-01")
    assert [p.created_at[24:26] for p in created] == ["00", "01", "02", "03", "04"]
    assert [p.entity for p in queue.list()] == ["Entity{}".format(index) for index in range(5)]


def test_confidence_bounds_are_inclusive():
    assert Fact(**{"id": "a" * 32, "entity": "e", "attribute": "a", "value": "v",
                   "valid_from": "2025-01-01", "recorded_at": "2025-01-01",
                   "confidence": 0}).confidence == 0.0


def test_approval_keeps_every_reviewed_field(store):
    queue = ProposalQueue(store)
    [proposal] = queue.extract("My application at Acme is pending.", source="synthetic:review",
                               observed_at="2025-01-01", topic="career/applications")
    fact = queue.approve(proposal.id)
    assert (fact.topic, fact.confidence, fact.source, fact.evidence) == (
        "career/applications", 0.85, "synthetic:review", "My application at Acme is pending.")
    [correction] = queue.extract("2025-01-01: My application at Acme is withdrawn.",
                                 source="synthetic:review-2")
    assert correction.kind == "correct"
    corrected = queue.approve(correction.id)
    assert (corrected.value, corrected.source, corrected.evidence, corrected.supersedes) == (
        "withdrawn", "synthetic:review-2", "2025-01-01: My application at Acme is withdrawn.",
        fact.id)


def test_classification_uses_the_proposal_date_and_its_own_entity(store):
    store.add("Acme application", "status", "done", valid_from="2025-03-01")
    other = store.add("Cedar application", "status", "draft", valid_from="2025-01-01")
    store.retract(other.id)
    store.add("Beacon application", "status", "open", valid_from="2025-01-01")
    queue = ProposalQueue(store)
    [proposal] = queue.extract("Acme application | status | pending | 2025-01-01",
                               source="synthetic:date")
    assert proposal.kind == "add"  # The March state is in the future of 1 January.
    assert queue.approve(proposal.id).value == "pending"


def test_creation_times_only_step_forward_when_the_clock_has_not():
    from temporal_memory_os.extraction import _after
    assert _after("2000-01-01T00:00:00.000000+00:00") > "2026"
    assert _after("2999-01-01T00:00:00.000000+00:00") == "2999-01-01T00:00:00.000001+00:00"


def test_one_unusable_document_does_not_end_the_embedding_scan(store):
    # Snapshot order is alphabetical: the unusable records come first.
    store.add("Aardvark", "diet", "ants", valid_from="2025-01-01")
    zero = store.add("Badger", "diet", "worms", valid_from="2025-01-01")
    tea = store.add("Zebra", "drink", "tea", valid_from="2025-01-01")
    lexical = HybridRetriever(store, HashingEmbedder()).search("tea", mode="semantic")
    assert [hit.fact.id for hit in lexical] == [tea.id]

    class Provider:
        semantic = True
        name = "zero-vector"

        def embed(self, texts):
            return [[0.0, 0.0] if text.startswith("Badger") else [1.0, 0.0] for text in texts]

    hits = HybridRetriever(store, Provider()).search("q", mode="semantic", limit=10)
    assert zero.id not in {hit.fact.id for hit in hits} and len(hits) == 2


def test_reextraction_reuses_duplicates_and_still_proposes_new_lines(store):
    queue = ProposalQueue(store)
    [first] = queue.extract("Alex | city | Riverton", source="synthetic:repeat",
                            observed_at="2025-01-01")
    again = queue.extract("Alex | city | Riverton\nAlex | drink | tea", source="synthetic:repeat",
                          observed_at="2025-01-01")
    assert [p.id for p in again][0] == first.id
    assert [p.value for p in again] == ["Riverton", "tea"]
