import json
import os

import pytest
from helpers import approved_city

from temporal_memory_os import MemoryStore
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.store import MemoryConflictError, StoreIntegrityError

BOM = "\ufeff"


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


def test_duplicate_fact_approval_adds_a_record_only_for_a_later_date(tmp_path):
    """Restating the value in effect writes a record only when the line gives a later date."""
    with MemoryStore(tmp_path) as store:
        fact = store.add("user", "city", "Paris", valid_from="2026-01-01")
        queue = ProposalQueue(store)
        same, later = queue.extract("user | city | Paris | 2026-01-01\n"
                                    "user | city | Paris | 2026-02-01", source="chat")
        assert (same.kind, later.kind) == ("duplicate", "duplicate")
        assert queue.approve(same.id).id == fact.id  # The record itself, restated.
        restated = queue.approve(later.id)
        assert (restated.value, restated.valid_from) == ("Paris", "2026-02-01T00:00:00.000000Z")
        assert [f.id for f in store.all_facts()] == [fact.id, restated.id]


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


def test_undated_restatement_after_a_different_pending_value_is_a_new_proposal(
        store, queue, ticking_clock):
    """A different value still waiting for review also makes a restatement new information."""
    [paris] = queue.extract("My city is Paris.", source="chat")
    [london] = queue.extract("My city is London.", source="chat")
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id != paris.id and again.valid_from > london.valid_from
    # The same line once more, with nothing said in between, is the pending one.
    [repeat] = queue.extract("My city is Paris.", source="chat")
    assert repeat.id == again.id
    for proposal in queue.list():
        queue.approve(proposal.id)
    assert [f.value for f in store.snapshot()] == ["Paris"]
    assert [f.value for f in store.history("user", "city")] == ["Paris", "London", "Paris"]


def test_undated_restatement_still_folds_after_a_rejected_or_same_value_proposal(
        store, queue, ticking_clock):
    """Only a different value proposed since, and not rejected, moves past a pending line."""
    [paris] = queue.extract("My city is Paris.", source="chat")
    [london] = queue.extract("My city is London.", source="chat")
    queue.reject(london.id)
    [elsewhere] = queue.extract("My city is Paris.", source="other")
    assert elsewhere.id != paris.id
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == paris.id


@pytest.mark.parametrize("earlier", [
    ["My city is Paris.", "My city is Paris.\nMy city is London."],
    ["My city is Paris.", "My city is London."],
], ids=["the-log-so-far", "one-message-at-a-time"])
def test_extracting_an_unchanged_log_again_returns_its_pending_proposals(
        store, queue, ticking_clock, earlier):
    """A different value the same text states on a later line is not news for an earlier line."""
    for text in earlier:
        queue.extract(text, source="chat")
    pending = [p.id for p in queue.list()]
    log = "My city is Paris.\nMy city is London."
    assert [p.id for p in queue.extract(log, source="chat")] == pending
    assert [p.id for p in queue.extract(log, source="chat")] == pending
    for proposal in queue.list():
        queue.approve(proposal.id)
    assert [f.value for f in store.history("user", "city")] == ["Paris", "London"]


def test_a_log_that_returns_to_an_earlier_value_proposes_it_again_once(store, queue,
                                                                        ticking_clock):
    """A value a log returns to is proposed again, and extracting that log again adds nothing."""
    queue.extract("My city is Paris.", source="chat")
    queue.extract("My city is Paris.\nMy city is London.", source="chat")
    log = "My city is Paris.\nMy city is London.\nMy city is Paris."
    first = queue.extract(log, source="chat")
    assert [p.id for p in queue.extract(log, source="chat")] == [p.id for p in first]
    assert len(queue.list()) == 3
    # The earlier lines return their own, earlier proposals: nothing shares its time.
    assert queue.list()[-1].warnings == []
    for proposal in queue.list():
        queue.approve(proposal.id)
    assert [f.value for f in store.history("user", "city")] == ["Paris", "London", "Paris"]


def test_a_dated_restatement_is_not_overridden_by_an_earlier_change_approved_later(store, queue):
    """An approved line repeating the value in effect keeps its date against a later backfill."""
    january = approved_city(queue, "user | city | London | 2026-01-01")
    [march] = queue.extract("user | city | London | 2026-03-01", source="chat")
    assert (march.kind, march.target_id) == ("duplicate", january.id)
    restated = queue.approve(march.id)
    assert restated.id != january.id and restated.valid_from == "2026-03-01T00:00:00.000000Z"
    approved_city(queue, "user | city | Oslo | 2026-02-01")  # Said later, about February.
    assert [f.value for f in store.snapshot(as_of="2026-03-15")] == ["London"]
    assert [f.value for f in store.history("user", "city")] == ["London", "Oslo", "London"]


def test_a_restatement_at_an_observation_time_is_not_overridden_by_a_later_backfill(store,
                                                                                    queue):
    """An undated line timed by an explicit observation time is recorded like a dated line."""
    [january] = queue.extract("My city is London.", source="chat", observed_at="2026-01-01")
    queue.approve(january.id)
    [march] = queue.extract("My city is London.", source="chat", observed_at="2026-03-01")
    assert (march.kind, march.dated) == ("duplicate", False)
    restated = queue.approve(march.id)
    assert restated.valid_from == "2026-03-01T00:00:00.000000Z"
    [february] = queue.extract("My city is Oslo.", source="chat", observed_at="2026-02-01")
    queue.approve(february.id)  # Said later, about February.
    assert [f.value for f in store.snapshot(as_of="2026-03-15")] == ["London"]
    assert [f.value for f in store.history("user", "city")] == ["London", "Oslo", "London"]


@pytest.mark.parametrize("line, observed_at, observed", [
    ("My city is Paris.", "2026-06-01", True),
    ("My city is Paris.", None, False),
    ("My city is Paris.", "  ", False),
    ("2026-01-01: My city is Paris.", "2026-06-01", False),
    ("Alex | city | Paris | 2026-01-01", "2026-06-01", False),
], ids=["undated-with-a-time", "undated-now", "undated-blank-time", "date-prefix",
        "date-field"])
def test_only_an_undated_line_with_an_explicit_observation_time_is_observed(queue, line,
                                                                            observed_at,
                                                                            observed):
    """The observed flag marks a line that took an explicit observation time, not "now"."""
    [proposal] = queue.extract(line, source="chat", observed_at=observed_at)
    assert proposal.observed is observed
    assert queue.get(proposal.id).observed is observed


def test_an_undated_or_same_date_restatement_writes_nothing(store, queue, ticking_clock):
    """An undated restatement extracted "now", or one for the record's own date, adds no record."""
    [first] = queue.extract("My city is Paris.", source="chat")
    paris = queue.approve(first.id)
    [undated] = queue.extract("My city is Paris.", source="chat")
    assert (undated.kind, undated.dated) == ("duplicate", False)
    assert queue.approve(undated.id).id == paris.id
    dated = approved_city(queue, "user | drink | tea | 2026-01-01")
    [same_date] = queue.extract("2026-01-01: user | drink | tea", source="chat")
    assert same_date.kind == "duplicate"
    assert queue.approve(same_date.id).id == dated.id
    assert [f.id for f in store.all_facts()] == [paris.id, dated.id]


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


def test_a_restatement_from_another_source_keeps_later_proposals_valid(store, queue):
    """A newer record restating the prior value a reviewer saw does not make the preview stale."""
    approved_city(queue, "user | city | London | 2026-01-01")
    [march] = queue.extract("user | city | London | 2026-03-01", source="email")
    [april] = queue.extract("user | city | Oslo | 2026-04-01", source="chat")
    [may] = queue.extract("user | city | Oslo | 2026-05-01", source="notes")
    [june] = queue.extract("user | city | London | 2026-06-01", source="notes")
    assert (march.kind, april.kind, april.prior_value) == ("duplicate", "update", "London")
    assert (may.prior_value, june.kind) == ("London", "duplicate")
    queue.approve(march.id)
    assert queue.approve(april.id).value == "Oslo"
    # Oslo, not London, now precedes May and June: those previews no longer hold.
    for stale in (may, june):
        with pytest.raises(MemoryConflictError, match="Memory changed since this proposal"):
            queue.approve(stale.id)
    assert [f.value for f in store.history("user", "city")] == ["London", "London", "Oslo"]


def test_a_restated_prior_still_allows_a_later_duplicate_from_another_source(store, queue):
    """A later line confirming the value a newer restatement holds is recorded at its date."""
    approved_city(queue, "user | city | London | 2026-01-01")
    [march] = queue.extract("user | city | London | 2026-03-01", source="email")
    [may] = queue.extract("user | city | London | 2026-05-01", source="chat")
    assert (march.kind, may.kind) == ("duplicate", "duplicate")
    queue.approve(march.id)
    restated = queue.approve(may.id)
    assert restated.valid_from == "2026-05-01T00:00:00.000000Z"
    assert len(store.history("user", "city")) == 3


def test_a_sibling_from_the_same_text_gets_a_specific_message(store, queue):
    tea, jazz = queue.extract("I prefer tea.\nI like jazz.", source="chat",
                              observed_at="2026-10-01")
    queue.approve(tea.id)
    with pytest.raises(MemoryConflictError, match=(
            "^Another proposal from the same source already set preference for this time; each "
            "attribute holds one value at a time")) as caught:
        queue.approve(jazz.id)
    assert [f.value for f in store.all_facts()] == ["tea"]
    # Extracting again before rejecting returns the same pending proposal, so the
    # message says to reject first.
    assert "reject this one first, then extract it again" in str(caught.value)
    queue.reject(jazz.id)
    [change] = queue.extract("I like jazz.", source="chat", observed_at="2026-10-01")
    assert (change.kind, change.prior_value) == ("correct", "tea")
    assert queue.approve(change.id).value == "jazz"


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


def approve_interrupted(queue, proposal_id, monkeypatch):
    """Approve, stopping after the fact is written but before the proposal is saved."""
    def crash(self, proposal):
        raise RuntimeError("process stopped after the fact was written")

    with monkeypatch.context() as patch:
        patch.setattr(ProposalQueue, "_save", crash)
        with pytest.raises(RuntimeError):
            queue.approve(proposal_id)


def test_recovery_recognizes_a_return_to_an_earlier_value_approved_in_order(
        store, queue, monkeypatch):
    """A duplicate applied in order as a return to an earlier value is recovered after a stop."""
    approved_city(queue, "user | city | London | 2026-01-01")
    oslo, march = queue.extract("user | city | Oslo | 2026-02-01\n"
                                "user | city | London | 2026-03-01", source="chat")
    assert (oslo.kind, march.kind) == ("update", "duplicate")
    queue.approve(oslo.id)
    approve_interrupted(queue, march.id, monkeypatch)
    [written] = [f for f in store.all_facts() if f.valid_from.startswith("2026-03-01")]
    assert queue.get(march.id).status == "pending"
    assert queue.approve(march.id).id == written.id
    assert queue.get(march.id).result_fact_id == written.id
    assert [f.value for f in store.history("user", "city")] == ["London", "Oslo", "London"]


def test_recovery_recognizes_a_dated_restatement(store, queue, monkeypatch):
    """A dated restatement interrupted after writing its record is recovered by approving again."""
    january = approved_city(queue, "user | city | London | 2026-01-01")
    [march] = queue.extract("user | city | London | 2026-03-01", source="chat")
    approve_interrupted(queue, march.id, monkeypatch)
    [written] = [f for f in store.all_facts() if f.id != january.id]
    assert queue.approve(march.id).id == written.id
    assert len(store.all_facts()) == 2


@pytest.mark.parametrize("text, confirming", [
    # Undated and extracted "now", so approving it writes nothing and points at the fact.
    ("user | city | London | 2025-01-01\nMy city is London.", None),
    ("user | city | London | 2025-01-01", "2025-01-01: user | city | London"),
], ids=["later-undated-line-of-the-same-text", "same-instant-on-another-line"])
def test_a_duplicate_approval_does_not_claim_the_fact_it_confirms(store, queue, monkeypatch,
                                                                 ticking_clock, text,
                                                                 confirming):
    """An interrupted approval completes even after another line was approved as its duplicate."""
    first, *rest = queue.extract(text, source="chat")
    approve_interrupted(queue, first.id, monkeypatch)
    [written] = store.all_facts()
    if confirming is not None:
        rest = queue.extract(confirming, source="chat")
        assert rest[0].kind == "duplicate" and rest[0].target_id == written.id
    [other] = rest
    assert queue.approve(other.id).id == written.id  # Approved as a duplicate of that fact.
    assert queue.approve(first.id).id == written.id
    assert queue.get(first.id).status == "approved"
    assert store.all_facts() == [written]


@pytest.mark.parametrize("topic, filed", [("work", "Work"), ("work/new", "Work/new")])
def test_approving_a_topic_spelled_unlike_its_existing_folder_files_the_record_there(
        store, queue, monkeypatch, topic, filed):
    """On a case-insensitive disk, an approved record goes into the existing folder as spelled."""
    store.add("Alex", "city", "Riverton", valid_from="2025-01-01", topic="Work")
    if not (store.memory_dir / "work").is_dir():
        pytest.skip("needs a case-insensitive file system")
    employer, pet = queue.extract("Alex | employer | Acme\nAlex | pet | cat", source="chat",
                                  observed_at="2025-01-01", topic=topic)
    assert (employer.kind, employer.topic) == ("add", topic)
    fact = queue.approve(employer.id)
    assert fact.topic == filed
    assert store.path_for(fact.id).parent == store.memory_dir.joinpath(*filed.split("/"))
    assert store.refresh() is False
    # An approval interrupted after filing its record there completes when approved again.
    approve_interrupted(queue, pet.id, monkeypatch)
    [written] = [f for f in store.all_facts() if f.attribute == "pet"]
    assert written.topic == filed
    assert queue.approve(pet.id).id == written.id
    assert os.listdir(store.memory_dir) == ["Work"]


def test_rejecting_a_proposal_whose_interrupted_approval_wrote_its_fact_is_refused(
        store, queue, monkeypatch):
    """Rejecting would leave the written fact in memory, so it is refused and stays pending."""
    [first] = queue.extract("user | city | Paris | 2026-01-01", source="chat")
    approve_interrupted(queue, first.id, monkeypatch)
    [written] = store.all_facts()
    with pytest.raises(MemoryConflictError, match="approve it again to finish"):
        queue.reject(first.id)
    assert queue.get(first.id).status == "pending"
    assert queue.approve(first.id).id == written.id


def test_a_repeated_line_is_one_proposal_without_a_same_time_warning(queue):
    """The same line twice is one statement: one proposal, returned once, with no warning."""
    proposals = queue.extract("My city is Paris.\nMy city is Paris.", source="chat",
                              observed_at="2026-10-01")
    assert [(p.value, p.warnings) for p in proposals] == [("Paris", [])]
    assert len(queue.list()) == 1
    [again] = queue.extract("My city is Paris.\nMy city is Paris.", source="chat",
                            observed_at="2026-10-01")
    assert again.id == proposals[0].id


def test_a_byte_order_mark_at_the_start_of_a_later_line_is_ignored(store, queue):
    """Files joined with cat keep each one's byte-order mark; it is not part of a name."""
    store.add("user", "city", "London", valid_from="2026-01-01")
    text = "# notes\n" + BOM + "user | city | Paris | 2026-02-01\n" + BOM + "My drink is tea."
    proposals = queue.extract(text, source="notes", observed_at="2026-03-01")
    assert [(p.entity, p.attribute, p.kind) for p in proposals] == [
        ("user", "city", "update"), ("user", "drink", "add")]
    assert all(BOM not in p.evidence for p in proposals)


@pytest.mark.parametrize("link", ["folder", "lock", "file"])
def test_symbolic_links_in_the_proposal_queue_are_store_problems(store, tmp_path_factory, link):
    """A linked proposal folder, lock or file needs repair on disk, not a different request."""
    outside = tmp_path_factory.mktemp("outside")
    if link == "folder":
        (store.root / "proposals").symlink_to(outside, target_is_directory=True)
        with pytest.raises(StoreIntegrityError, match="Proposal directory"):
            ProposalQueue(store)
        return
    queue = ProposalQueue(store)
    if link == "lock":
        (queue.directory / ".lock").symlink_to(outside / "lock")
        with pytest.raises(StoreIntegrityError, match="Proposal lock"):
            queue.extract("Alex | city | Riverton", source="chat")
        return
    [proposal] = queue.extract("Alex | city | Riverton", source="chat")
    path = queue.directory / (proposal.id + ".json")
    (outside / "proposal.json").write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside / "proposal.json")
    name = "proposals/{}.json".format(proposal.id)
    for action in (queue.get, queue.approve, queue.reject):
        with pytest.raises(StoreIntegrityError, match=name):
            action(proposal.id)


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="root ignores file permissions")
def test_unreadable_proposal_files_are_named_without_the_memory_path(store, queue):
    """A proposal file that cannot be read is reported relative to the memory directory."""
    unreadable, folder, kept = queue.extract("Alex | city | Riverton\nAlex | drink | tea\n"
                                             "Alex | pet | cat", source="chat",
                                             observed_at="2026-01-01")
    path = queue.directory / (unreadable.id + ".json")
    path.chmod(0)
    try:
        (queue.directory / (folder.id + ".json")).unlink()
        (queue.directory / (folder.id + ".json")).mkdir()
        assert [p.id for p in queue.list()] == [kept.id]
        assert sorted(queue.problems) == sorted([
            "proposals/{}.json cannot be read (Permission denied); fix or delete it".format(
                unreadable.id),
            "proposals/{}.json cannot be read (Is a directory); fix or delete it".format(
                folder.id)])
        with pytest.raises(StoreIntegrityError) as caught:
            queue.approve(unreadable.id)
        assert str(caught.value).startswith("proposals/{}.json ".format(unreadable.id))
        assert str(store.root) not in str(caught.value)
    finally:
        path.chmod(0o600)
    assert queue.approve(unreadable.id).value == "Riverton"


def test_a_file_the_listing_cannot_inspect_is_named_without_the_memory_path(queue, monkeypatch):
    """An operating-system error while listing names the file, not its absolute path."""
    damaged, kept = queue.extract("Alex | city | Riverton\nAlex | drink | tea", source="chat",
                                  observed_at="2026-01-01")
    real_get = ProposalQueue.get

    def get(self, proposal_id):
        if proposal_id == damaged.id:
            raise PermissionError(13, "Permission denied", str(self.directory / proposal_id))
        return real_get(self, proposal_id)

    monkeypatch.setattr(ProposalQueue, "get", get)
    assert [p.id for p in queue.list()] == [kept.id]
    assert queue.problems == [
        "proposals/{}.json cannot be read (Permission denied)".format(damaged.id)]


def test_http_problems_and_link_refusals_do_not_disclose_the_memory_path(tmp_path):
    """The API answers 503 for a linked proposal file and names files only by relative path."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from temporal_memory_os.api import create_app

    with TestClient(create_app(tmp_path), base_url="http://localhost") as client:
        linked, kept = client.post("/api/proposals/extract", json={
            "text": "Alex | city | Riverton\nAlex | drink | tea", "source": "synthetic:chat",
            "observed_at": "2026-01-01"}).json()["proposals"]
        path = tmp_path / "proposals" / (linked["id"] + ".json")
        outside = tmp_path / "outside.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        approve = client.post("/api/proposals/{}/approve".format(linked["id"]))
        assert approve.status_code == 503, approve.text
        assert approve.json()["detail"].endswith("proposals/{}.json".format(linked["id"]))
        listed = client.get("/api/proposals").json()
        assert [p["id"] for p in listed["proposals"]] == [kept["id"]]
        assert [problem.endswith("proposals/{}.json".format(linked["id"]))
                for problem in listed["problems"]] == [True]
        assert str(tmp_path) not in approve.text + json.dumps(listed)


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
    # A sentence rule reads one sentence per line; the pipe form takes any value.
    ("My city is Paris. I moved here last year.", []),
    ("2025-01-01: My application at Acme is pending. Fingers crossed!", []),
    ("I prefer tea. Coffee keeps me up.", []),
    ("My application at Acme. Honestly it is pending.", []),
    ("My city is Paris? Not sure", []),
    # A period after an initial or a two-letter abbreviation does not end a sentence.
    ("My city is St. Louis", [("user", "city", "St. Louis", "2026-06-01")]),
    ("My doctor is Dr. Smith.", [("user", "doctor", "Dr. Smith", "2026-06-01")]),
    ("My employer is J. P. Morgan", [("user", "employer", "J. P. Morgan", "2026-06-01")]),
    ("I prefer Mr. Rogers. He is calm.", []),
    ("Alex | city | St. Louis", [("Alex", "city", "St. Louis", "2026-06-01")]),
    ("My city is St.Louis.", [("user", "city", "St.Louis", "2026-06-01")]),
    ("My employer is IBM. I started in May.", []),
], ids=["dated-comment-with-pipes", "dated-comment", "dated-comment-no-space",
        "conflicting-dates", "agreeing-dates", "name-with-is", "attribute-not-greedy",
        "dots-only", "bangs-only", "dashes-only-preference", "question-mark-status",
        "structured-colon", "structured-dots", "two-sentences", "two-sentences-dated",
        "two-sentences-preference", "two-sentences-in-the-name", "question-then-more",
        "abbreviation-in-prose", "title-in-prose", "initials-in-prose",
        "title-then-a-second-sentence", "abbreviation-structured", "no-space-after-the-dot",
        "acronym-then-a-second-sentence"])
def test_line_rules(queue, line, expected):
    assert parsed(queue, line) == expected


def test_a_log_extracted_again_after_another_text_keeps_its_earlier_lines(store, queue,
                                                                         ticking_clock):
    """Another text's value does not make a growing log's earlier lines new statements."""
    queue.extract("My city is London.", source="chat")
    queue.extract("My city is Oslo.", source="chat")  # Another conversation file.
    london, paris = queue.extract("My city is London.\nMy city is Paris.", source="chat")
    assert (london.warnings, paris.warnings) == ([], [])
    assert len(queue.list()) == 3
    for proposal in queue.list():
        queue.approve(proposal.id)
    assert [fact.value for fact in store.snapshot()] == ["Paris"]


@pytest.mark.parametrize("second_source", ["email", "chat"])
@pytest.mark.parametrize("timed", ["dated", "observed"])
def test_two_restatements_for_the_same_date_can_both_be_approved(store, queue, second_source,
                                                                 timed):
    """A second restatement of the current value for the same date changes nothing, so it
    is approved by the record the first one wrote."""
    approved_city(queue, "user | city | London | 2026-01-01")
    if timed == "dated":
        [first] = queue.extract("user | city | London | 2026-03-01", source="chat")
        [second] = queue.extract("2026-03-01: My city is London.", source=second_source)
    else:
        [first] = queue.extract("My city is London.", source="chat", observed_at="2026-03-01")
        [second] = queue.extract("user | city | London", source=second_source,
                                 observed_at="2026-03-01")
    assert (first.kind, second.kind) == ("duplicate", "duplicate")
    written = queue.approve(first.id)
    assert queue.approve(second.id).id == written.id
    assert [fact.valid_from[:10] for fact in store.history("user", "city")] == [
        "2026-01-01", "2026-03-01"]


def test_the_same_line_under_another_spelling_of_its_topic_can_still_be_rejected(store,
                                                                                queue):
    """A second proposal whose topic names the same folder in another letter case is not
    mistaken for an interrupted approval."""
    probe = store.memory_dir / "CaseProbe"
    probe.mkdir()
    insensitive = (store.memory_dir / "caseprobe").exists()
    probe.rmdir()
    if not insensitive:
        pytest.skip("needs a case-insensitive file system")
    line = "user | city | Paris | 2026-01-01"
    [first] = queue.extract(line, source="chat", topic="Work")
    [second] = queue.extract(line, source="chat", topic="work")
    assert first.id != second.id
    written = queue.approve(first.id)
    rejected = queue.reject(second.id)
    assert rejected.status == "rejected"
    assert [fact.id for fact in store.all_facts()] == [written.id]


# Rules pinned in 0.2.3


@pytest.mark.parametrize("line, observed_at", [
    ("user | city | Paris | 2026-02-01", None),
    ("My city is Paris.", "2026-02-01"),
], ids=["dated", "observed"])
def test_a_restatement_inside_an_explicit_interval_keeps_its_end(store, queue, line,
                                                                 observed_at):
    """A restatement recorded at its own date ends where the record it confirms ends, so
    approving it does not extend the value past an explicit end."""
    bounded = store.add("user", "city", "Paris", valid_from="2026-01-01", valid_to="2026-03-01")
    [restatement] = queue.extract(line, source="chat", observed_at=observed_at)
    assert (restatement.kind, restatement.target_id) == ("duplicate", bounded.id)
    restated = queue.approve(restatement.id)
    assert restated.id != bounded.id
    assert (restated.valid_from, restated.valid_to) == ("2026-02-01T00:00:00.000000Z",
                                                        "2026-03-01T00:00:00.000000Z")
    assert [f.value for f in store.snapshot(as_of="2026-02-15")] == ["Paris"]
    assert store.snapshot(as_of="2026-04-01") == []


def test_a_restatement_does_not_make_an_end_set_by_the_next_value_explicit(store, queue):
    """A record ended only by the next value has no end of its own to carry over."""
    paris = store.add("user", "city", "Paris", valid_from="2026-01-01")
    store.add("user", "city", "London", valid_from="2026-03-01")
    [restatement] = queue.extract("user | city | Paris | 2026-02-01", source="chat")
    assert (restatement.kind, restatement.target_id) == ("duplicate", paris.id)
    assert queue.approve(restatement.id).valid_to is None
    assert [f.value for f in store.history("user", "city")] == ["Paris", "Paris", "London"]


@pytest.mark.parametrize("revise", ["retract", "correct"])
def test_a_proposal_whose_interrupted_fact_was_revised_since_can_be_rejected(
        store, queue, monkeypatch, revise):
    """Once the fact an interrupted approval wrote is retracted or corrected, rejecting the
    proposal leaves nothing of it in effect, so the reject goes through."""
    [first] = queue.extract("user | city | Paris | 2026-01-01", source="chat")
    approve_interrupted(queue, first.id, monkeypatch)
    [written] = store.all_facts()
    if revise == "retract":
        store.retract(written.id)
    else:
        store.correct(written.id, "Lyon")
    assert queue.reject(first.id).status == "rejected"
    assert queue.get(first.id).status == "rejected"
    expected = [] if revise == "retract" else ["Lyon"]
    assert [f.value for f in store.snapshot(as_of="2026-02-01")] == expected


def test_approving_after_the_interrupted_fact_was_retracted_does_not_write_it_again(
        store, queue, monkeypatch):
    """Approving instead still finishes the interrupted approval; the retraction stands."""
    [first] = queue.extract("user | city | Paris | 2026-01-01", source="chat")
    approve_interrupted(queue, first.id, monkeypatch)
    [written] = store.all_facts()
    store.retract(written.id)
    assert queue.approve(first.id).id == written.id
    assert len(store.all_facts()) == 2
    assert store.snapshot(as_of="2026-02-01") == []


@pytest.mark.parametrize("rejected_first", ["twin", "writer"])
def test_one_of_two_spellings_of_a_topic_can_be_rejected_while_an_approval_is_interrupted(
        store, queue, monkeypatch, rejected_first):
    """Two pending proposals state the same line under two letter cases of one topic folder.
    While either can still claim the fact the interrupted approval wrote, the other can be
    rejected; the last one cannot, and approving it completes the approval."""
    probe = store.memory_dir / "CaseProbe"
    probe.mkdir()
    insensitive = (store.memory_dir / "caseprobe").exists()
    probe.rmdir()
    if not insensitive:
        pytest.skip("needs a case-insensitive file system")
    line = "user | city | Paris | 2026-01-01"
    [writer] = queue.extract(line, source="chat", topic="Work")
    [twin] = queue.extract(line, source="chat", topic="work")
    approve_interrupted(queue, writer.id, monkeypatch)
    [written] = store.all_facts()
    first, last = (twin, writer) if rejected_first == "twin" else (writer, twin)
    assert queue.reject(first.id).status == "rejected"
    with pytest.raises(MemoryConflictError, match="approve it again to finish"):
        queue.reject(last.id)
    assert queue.approve(last.id).id == written.id
    assert [fact.id for fact in store.all_facts()] == [written.id]


def test_a_repeated_line_after_another_value_gives_one_new_proposal(store, queue,
                                                                     ticking_clock):
    """A line said twice after another value is one new statement; the older pending
    proposal for it is not returned with it."""
    [paris] = queue.extract("My city is Paris.", source="chat")
    queue.extract("My city is London.", source="chat")
    [again] = queue.extract("My city is Paris.\nMy city is Paris.", source="chat")
    assert again.id != paris.id and again.warnings == []
    assert len(queue.list()) == 3
    [repeat] = queue.extract("My city is Paris.\nMy city is Paris.", source="chat")
    assert repeat.id == again.id


@pytest.mark.parametrize("line, expected", [
    ('My motto is "Carpe diem." I live by it', []),
    ("My city is Paris.) I moved here", []),
    ("My city is Paris!\u201d I moved", []),
    ("My city is Paris?] Not sure", []),
    ('My motto is "Carpe diem."', [("user", "motto", '"Carpe diem."', "2026-06-01")]),
    ("My president is \u00c9. Macron", [("user", "president", "\u00c9. Macron", "2026-06-01")]),
    ("My doctor is \u00d6. Smith.", [("user", "doctor", "\u00d6. Smith", "2026-06-01")]),
    ("My saint is \u015aw. Jan", [("user", "saint", "\u015aw. Jan", "2026-06-01")]),
    ("My employer is J. P. Morgan", [("user", "employer", "J. P. Morgan", "2026-06-01")]),
    ("My city is St. Louis", [("user", "city", "St. Louis", "2026-06-01")]),
    ("My doctor is Dr. Smith.", [("user", "doctor", "Dr. Smith", "2026-06-01")]),
    ("My city is \u6771. I moved", []),
    ("My president is E\u0301. Macron", [("user", "president", "E\u0301. Macron", "2026-06-01")]),
    ("My motto is \u2039Carpe diem.\u203a Yes", []),
    ("My city is Paris.\uff09 I moved", []),
], ids=["quote-then-a-second-sentence", "bracket-then-a-second-sentence",
        "bang-and-curly-quote-then-more", "question-and-bracket-then-more", "quote-at-the-end",
        "accented-initial", "umlaut-initial", "accented-abbreviation", "initials", "abbreviation",
        "title", "uncased-letter-then-a-second-sentence", "decomposed-accent-initial",
        "angle-quote-then-more", "full-width-bracket-then-more"])
def test_sentence_ends_before_closing_marks_and_unicode_initials(queue, line, expected):
    """A sentence end followed by a closing quote or bracket still ends the sentence, and a
    period after a capital initial in any script does not."""
    assert parsed(queue, line) == expected


@pytest.mark.parametrize("text", [BOM + "\n" + BOM, BOM + "  \r\n " + BOM + "\t\n\n"])
def test_text_made_only_of_byte_order_marks_on_several_lines_is_empty(queue, text):
    """Each line's leading byte-order mark is ignored, so these lines are all empty."""
    with pytest.raises(ValueError, match="Conversation text must not be empty"):
        queue.extract(text, source="chat")
    assert queue.list() == []


def test_oversized_text_is_refused_before_its_lines_are_read(queue):
    """The size limit comes first, so text that would also read as empty gets the size error."""
    with pytest.raises(ValueError, match="1 MB extraction limit"):
        queue.extract((BOM + "\n") * 600_000, source="chat")


def test_reject_is_still_refused_when_a_twin_under_another_topic_cannot_claim_the_fact(
        store, queue, monkeypatch):
    """The same line under an unrelated topic is no claimant for the written fact."""
    line = "user | city | Paris | 2026-01-01"
    [writer] = queue.extract(line, source="chat", topic="general")
    [twin] = queue.extract(line, source="chat", topic="work")
    assert twin.id != writer.id
    approve_interrupted(queue, writer.id, monkeypatch)
    [written] = store.all_facts()
    with pytest.raises(MemoryConflictError, match="approve it again to finish"):
        queue.reject(writer.id)
    assert queue.get(writer.id).status == "pending"
    assert queue.approve(writer.id).id == written.id


def test_a_line_for_another_attribute_does_not_hide_a_restatement(queue):
    """Only another value for the same attribute makes a line one said before that value."""
    [first] = queue.extract("My city is Paris", source="chat")
    queue.extract("My city is London", source="chat")
    again = queue.extract("My city is Paris\nMy color is blue", source="chat")
    [paris] = [proposal for proposal in again if proposal.attribute == "city"]
    assert paris.id != first.id
    assert len(queue.list(status="all")) == 4
