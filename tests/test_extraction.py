import json

import pytest
from helpers import approved_city

from temporal_memory_os import MemoryStore
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.store import MemoryConflictError


def test_extract_requires_approval_and_preserves_evidence(tmp_path):
    with MemoryStore(tmp_path) as store:
        queue = ProposalQueue(store)
        text = "2026-09-01: My application at Acme is pending.\nThis sentence is unsupported."
        proposals = queue.extract(text, source="conversation:1")
        assert len(proposals) == 1
        assert store.all_facts() == []
        proposal = proposals[0]
        assert proposal.entity == "Acme application"
        assert proposal.value == "pending"
        fact = queue.approve(proposal.id)
        assert fact.evidence == text.splitlines()[0]
        assert fact.source == "conversation:1"
        assert store.snapshot(as_of="2026-09-10")[0].value == "pending"
        assert queue.approve(proposal.id).id == fact.id


def test_correction_preview_and_knowledge_history(tmp_path):
    with MemoryStore(tmp_path) as store:
        old = store.add("user", "city", "Paris", valid_from="2026-01-01", recorded_at="2026-01-02")
        queue = ProposalQueue(store)
        proposal = queue.extract("user | city | Lyon | 2026-01-01", source="correction")[0]
        assert proposal.kind == "correct"
        assert proposal.prior_value == "Paris"
        corrected = queue.approve(proposal.id)
        assert corrected.supersedes == old.id
        assert store.snapshot(as_of="2026-02-01", known_at="2026-02-01")[0].value == "Paris"
        assert store.snapshot(as_of="2026-02-01")[0].value == "Lyon"


def test_stale_approval_rejected_and_can_reextract(tmp_path):
    with MemoryStore(tmp_path) as store:
        old = store.add("user", "city", "Paris", valid_from="2026-01-01")
        queue = ProposalQueue(store)
        text = "user | city | Lyon | 2026-01-01"
        proposed = queue.extract(text, source="chat")[0]
        store.correct(old.id, "London")
        with pytest.raises(ValueError, match="Memory changed"):
            queue.approve(proposed.id)
        queue.reject(proposed.id)
        fresh = queue.extract(text, source="chat")[0]
        assert fresh.id != proposed.id
        assert queue.approve(fresh.id).value == "Lyon"


def test_deduplicate_and_rejection(tmp_path):
    with MemoryStore(tmp_path) as store:
        queue = ProposalQueue(store)
        first = queue.extract("My city is Paris.", source="chat", observed_at="2026-01-01")[0]
        assert queue.extract("My city is Paris.", source="chat", observed_at="2026-01-01")[0].id == first.id
        queue.reject(first.id)
        with pytest.raises(ValueError, match="pending"):
            queue.approve(first.id)
        assert store.all_facts() == []


def test_duplicate_fact_approval_does_not_add_revision(tmp_path):
    with MemoryStore(tmp_path) as store:
        fact = store.add("user", "city", "Paris", valid_from="2026-01-01")
        queue = ProposalQueue(store)
        proposal = queue.extract("user | city | Paris | 2026-02-01", source="chat")[0]
        assert proposal.kind == "duplicate"
        assert queue.approve(proposal.id).id == fact.id
        assert len(store.all_facts()) == 1


def test_unsafe_inputs_do_not_write_any_proposals(tmp_path):
    with MemoryStore(tmp_path) as store:
        queue = ProposalQueue(store)
        with pytest.raises(ValueError):
            queue.extract("user | city | Paris", source="chat", topic="../outside")
        assert queue.list() == []
        # A malformed structured line is skipped like other unsupported prose.
        [proposal] = queue.extract("user | city | Paris\nbroken | line", source="chat")
        assert proposal.value == "Paris"
        with pytest.raises(ValueError):
            queue.get("../../outside")


def test_multi_update_conversation_can_be_approved_in_order(tmp_path):
    with MemoryStore(tmp_path) as store:
        queue = ProposalQueue(store)
        proposals = queue.extract(
            "Acme | status | pending | 2026-09-01\n"
            "Acme | status | invited | 2026-09-15\n"
            "Acme | status | completed | 2026-10-01", source="conversation:timeline")
        for proposal in proposals:
            queue.approve(proposal.id)
        assert [fact.value for fact in store.history("Acme")] == ["pending", "invited", "completed"]


def test_manual_edit_invalidates_preview_even_when_fact_id_is_unchanged(tmp_path):
    with MemoryStore(tmp_path) as store:
        old = store.add("user", "city", "Paris", valid_from="2026-01-01")
        queue = ProposalQueue(store)
        proposal = queue.extract("user | city | Lyon | 2026-01-01", source="chat")[0]
        path = store.path_for(old.id)
        path.write_text(path.read_text().replace('"value": "Paris"', '"value": "Berlin"'))
        with pytest.raises(ValueError, match="Memory changed"):
            queue.approve(proposal.id)
        assert store.get(old.id).value == "Berlin"


def test_restore_retracted_assertion_at_same_effective_time(tmp_path):
    with MemoryStore(tmp_path) as store:
        old = store.add("user", "city", "Paris", valid_from="2026-01-01")
        tombstone = store.retract(old.id)
        queue = ProposalQueue(store)
        proposal = queue.extract("user | city | Lyon | 2026-01-01", source="chat")[0]
        assert proposal.kind == "correct"
        assert queue.approve(proposal.id).supersedes == tombstone.id
        assert store.snapshot(as_of="2026-02-01")[0].value == "Lyon"


# Rules pinned in 0.2.2


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path) as memory:
        yield memory


@pytest.fixture
def queue(store):
    return ProposalQueue(store)


def test_undated_restatement_folds_only_when_observed_at_is_not_given(queue, ticking_clock):
    [first] = queue.extract("My city is Paris.", source="chat")
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id and not first.dated
    # An explicit observation time is a statement made then, not the pending one.
    [observed] = queue.extract("My city is Paris.", source="chat", observed_at="2026-06-01")
    assert observed.id != first.id and observed.valid_from == "2026-06-01T00:00:00.000000Z"
    [repeat] = queue.extract("My city is Paris.", source="chat", observed_at="2026-06-01")
    assert repeat.id == observed.id
    assert len(queue.list()) == 2


def test_undated_restatement_after_a_newer_value_is_a_new_proposal(store, queue, ticking_clock):
    [paris] = queue.extract("My city is Paris.", source="chat")
    [london] = queue.extract("My city is London.", source="chat")
    approved = queue.approve(london.id)
    # Paris said again after London: new information, not the old pending line.
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id != paris.id
    assert (again.kind, again.prior_value, again.target_id) == ("update", "London", approved.id)
    assert again.valid_from > london.valid_from > paris.valid_from
    queue.approve(again.id)
    assert [f.value for f in store.snapshot()] == ["Paris"]


def test_update_goes_stale_when_its_target_is_edited_in_place(store, queue):
    london = approved_city(queue)
    [update] = queue.extract("user | city | Paris | 2026-03-01", source="chat")
    assert (update.kind, update.target_id, update.prior_value) == ("update", london.id, "London")
    path = store.path_for(london.id)
    path.write_text(path.read_text(encoding="utf-8").replace('"London"', '"Berlin"'),
                    encoding="utf-8")
    with pytest.raises(MemoryConflictError, match="Memory changed since this proposal"):
        queue.approve(update.id)
    assert [f.value for f in store.history("user", "city")] == ["Berlin"]


@pytest.mark.parametrize("change", [
    # The reviewed target corrected, even by the same source: it starts no later.
    lambda store, target: store.correct(target.id, "Rome", source="chat"),
    # A newer predecessor from another source.
    lambda store, target: store.add("user", "city", "Oslo", valid_from="2026-02-01",
                                    source="other"),
    # Same source, but recorded before the proposal was made (an import).
    lambda store, target: store.add("user", "city", "Oslo", valid_from="2026-02-01",
                                    source="chat", recorded_at="2020-01-01"),
], ids=["same-source-correction", "other-source", "recorded-before-the-proposal"])
def test_update_goes_stale_unless_a_newer_same_source_predecessor_appeared(store, queue,
                                                                          change):
    london = approved_city(queue)
    [update] = queue.extract("user | city | Paris | 2026-03-01", source="chat")
    change(store, london)
    before = store.all_facts()
    with pytest.raises(MemoryConflictError, match="Memory changed since this proposal"):
        queue.approve(update.id)
    assert store.all_facts() == before and queue.get(update.id).status == "pending"


def test_a_newer_same_source_predecessor_recorded_later_is_expected(store, queue):
    london = approved_city(queue)
    [update] = queue.extract("user | city | Paris | 2026-03-01", source="chat")
    oslo = store.add("user", "city", "Oslo", valid_from="2026-02-01", source="chat")
    assert oslo.recorded_at > queue.get(update.id).created_at.replace("+00:00", "Z")
    assert queue.approve(update.id).value == "Paris"
    assert [f.value for f in store.history("user", "city")] == ["London", "Oslo", "Paris"]
    assert london.id in {f.id for f in store.all_facts()}


def test_a_sibling_from_the_same_text_gets_a_specific_message(store, queue):
    tea, jazz = queue.extract("I prefer tea.\nI like jazz.", source="chat",
                              observed_at="2026-10-01")
    queue.approve(tea.id)
    with pytest.raises(MemoryConflictError, match=(
            "^Another proposal from the same source already set preference for this time; each "
            "attribute holds one value at a time")):
        queue.approve(jazz.id)
    assert [f.value for f in store.all_facts()] == ["tea"]


def test_recovery_does_not_claim_a_fact_written_for_another_topic(store, queue, monkeypatch):
    [profile] = queue.extract("user | city | Paris | 2026-01-01", source="chat",
                              topic="profile")
    [travel] = queue.extract("user | city | Paris | 2026-01-01", source="chat", topic="travel")
    assert profile.id != travel.id

    def crash(self, proposal):
        raise RuntimeError("process stopped after the fact was written")

    # Approving the travel proposal stops between writing its fact and saving it.
    with monkeypatch.context() as patch:
        patch.setattr(ProposalQueue, "_save", crash)
        with pytest.raises(RuntimeError):
            queue.approve(travel.id)
    [written] = store.all_facts()
    assert written.topic == "travel" and queue.get(travel.id).status == "pending"
    with pytest.raises(MemoryConflictError):
        queue.approve(profile.id)  # The travel fact is not the profile proposal's fact.
    assert queue.get(profile.id).status == "pending"
    assert queue.approve(travel.id).id == written.id  # The interrupted approval completes.
    store.retract(written.id)
    with pytest.raises(MemoryConflictError, match="Memory changed"):
        queue.approve(profile.id)
    assert store.snapshot(as_of="2026-02-01") == []


def test_recovery_does_not_claim_a_fact_with_another_confidence(store, queue):
    [proposal] = queue.extract("My city is Paris.", source="chat", observed_at="2026-01-01")
    assert proposal.confidence == 0.85
    manual = store.add("user", "city", "Paris", valid_from="2026-01-01", source="chat",
                       evidence="My city is Paris.", confidence=1.0)
    with pytest.raises(MemoryConflictError):
        queue.approve(proposal.id)
    assert queue.get(proposal.id).result_fact_id is None
    assert [f.id for f in store.all_facts()] == [manual.id]


def test_recovery_does_not_claim_a_fact_another_proposal_already_owns(store, queue):
    [proposal] = queue.extract("user | city | Paris | 2026-01-01", source="chat")
    written = queue.approve(proposal.id)
    # A copy of the reviewed file under a new ID, back to pending.
    data = json.loads((queue.directory / (proposal.id + ".json")).read_text(encoding="utf-8"))
    copy = "c" * 32
    (queue.directory / (copy + ".json")).write_text(json.dumps(
        {**data, "id": copy, "status": "pending", "result_fact_id": None}), encoding="utf-8")
    with pytest.raises(MemoryConflictError):
        queue.approve(copy)
    assert queue.get(copy).status == "pending"
    assert [f.id for f in store.all_facts()] == [written.id]


def parsed(queue, line):
    return [(p.entity, p.attribute, p.value, p.valid_from[:10]) for p in queue.extract(
        line + "\nAlex | pet | cat", source="chat", observed_at="2026-06-01")][:-1]


@pytest.mark.parametrize("line, expected", [
    ("2025-01-01: # a | b | c", []),
    ("2025-01-01: # My city is Paris", []),
    ("2025-01-01:   #comment", []),
    ("2025-01-01: Alex | city | Paris | 2025-02-01", []),
    ("2025-01-01: Alex | city | Paris | 2025-01-01T00:00Z",
     [("Alex", "city", "Paris", "2025-01-01")]),
    ("Application This Is Corp is pending.",
     [("This Is Corp application", "status", "pending", "2026-06-01")]),
    ("My name is Alex and my job is X", [("user", "name", "Alex and my job is X", "2026-06-01")]),
    ("My city is ...", []),
    ("My city is !!!", []),
    ("I prefer ---.", []),
    ("Application Acme is ?", []),
    ("Alex | mood | :", [("Alex", "mood", ":", "2026-06-01")]),
    ("Alex | mood | ...", [("Alex", "mood", "...", "2026-06-01")]),
], ids=["dated-comment-with-pipes", "dated-comment", "dated-comment-no-space",
        "conflicting-dates", "agreeing-dates", "name-with-is", "attribute-not-greedy",
        "dots-only", "bangs-only", "dashes-only-preference", "question-mark-status",
        "structured-colon", "structured-dots"])
def test_line_rules(queue, line, expected):
    assert parsed(queue, line) == expected
