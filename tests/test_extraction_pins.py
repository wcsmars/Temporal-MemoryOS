"""Proposal-queue behavior that mutation testing showed no other test checked.

Covers creation order, folder setup, the proposal file format and encoding,
change previews, statement parsing, input checks, re-extraction of undated
lines, crash recovery, the approval rules for expected predecessors, and
stray files in the proposal folder.
"""

import dataclasses
import hashlib
import io
import json
import shutil
from datetime import datetime, timedelta

import pytest
from helpers import approved_city, Python310Datetime

import temporal_memory_os.extraction as extraction
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.models import normalize_time
from temporal_memory_os.store import MemoryConflictError, MemoryStore


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "store") as memory:
        yield memory


@pytest.fixture
def queue(store):
    return ProposalQueue(store)


# ==========================================================================
# Creation order, folder setup, proposal files and change previews
# ==========================================================================


DUPLICATE_WARNING = "The same value is already effective on this date."
CHANGE_WARNING = "Changes an existing value; review the evidence and effective date."


def _latin1_default(encoding, stacklevel=2):
    """The default text encoding of a POSIX system with a Latin-1 locale."""
    return "latin-1" if encoding is None else encoding


# Creation times


def test_creation_times_step_forward_within_one_clock_tick_on_python_310(store, monkeypatch):
    """A batch extracted within one clock tick gets increasing creation times with 3.10's parser."""
    monkeypatch.setattr(extraction, "_now", lambda: "2025-01-01T00:00:00.000000+00:00")
    monkeypatch.setattr(extraction, "datetime", Python310Datetime)
    queue = ProposalQueue(store)
    created = queue.extract("Alpha | status | one\nBeta | status | two\nGamma | status | three",
                            source="synthetic:tick", observed_at="2025-01-01")
    assert [p.created_at for p in created] == [
        "2025-01-01T00:00:00.000000+00:00",
        "2025-01-01T00:00:00.000001+00:00",
        "2025-01-01T00:00:00.000002+00:00",
    ]
    assert [p.entity for p in queue.list()] == ["Alpha", "Beta", "Gamma"]


# Queue setup


def test_queue_recreates_its_folder_when_the_workspace_folder_was_removed(tmp_path):
    """Opening a queue creates the proposal folder together with any missing parent folders."""
    root = tmp_path / "workspace"
    with MemoryStore(root) as store:
        shutil.rmtree(root)
        queue = ProposalQueue(store)
        assert queue.directory.is_dir() and (root / "proposals").is_dir()
        assert queue.list(status="all") == [] and queue.problems == []


def test_a_new_queue_reports_no_problems_before_listing(store):
    """A queue that has not listed anything yet reports an empty list of problems."""
    queue = ProposalQueue(store)
    assert queue.problems == []


# Proposal files


def test_proposal_files_are_two_space_indented_json_ending_with_a_newline(queue):
    """A proposal file is UTF-8 JSON, two-space indented with one key per line, plus a newline."""
    [proposal] = queue.extract("Alex | city | Zürich", source="synthetic:chat",
                               observed_at="2025-01-01")
    text = (queue.directory / (proposal.id + ".json")).read_text(encoding="utf-8")
    assert text == json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2) + "\n"
    assert '\n  "value": "Zürich",\n' in text


def test_proposal_files_are_utf8_whatever_the_default_text_encoding(queue, monkeypatch):
    """Proposal files are written and read as UTF-8 even when the platform default is not."""
    monkeypatch.setattr(io, "text_encoding", _latin1_default)
    [tokyo, zurich] = queue.extract("Alex | city | 東京\nSam | city | Zürich",
                                    source="synthetic:chat", observed_at="2025-01-01")
    raw = (queue.directory / (tokyo.id + ".json")).read_bytes()
    assert '"value": "東京"'.encode("utf-8") in raw
    assert queue.get(tokyo.id).value == "東京"
    assert queue.get(zurich.id).value == "Zürich"
    assert [p.value for p in queue.list()] == ["東京", "Zürich"] and queue.problems == []


# Change previews


def test_target_fingerprint_is_the_sha256_of_the_records_sorted_key_json(store, queue):
    """A proposal fingerprints its target as SHA-256 of the record's sorted-key JSON."""
    old = store.add("user", "city", "Paris", valid_from="2026-01-01")
    [proposal] = queue.extract("user | city | Lyon | 2026-01-01", source="chat")
    canonical = json.dumps(store.get(old.id).to_dict(), sort_keys=True).encode("utf-8")
    assert proposal.target_fingerprint == hashlib.sha256(canonical).hexdigest()
    assert queue.approve(proposal.id).supersedes == old.id


def test_a_duplicate_proposal_warns_that_the_value_is_already_effective(store, queue):
    """Restating the effective value is previewed as a duplicate with its own warning."""
    fact = store.add("user", "city", "Paris", valid_from="2026-01-01")
    [proposal] = queue.extract("user | city | Paris | 2026-02-01", source="chat")
    assert (proposal.kind, proposal.target_id, proposal.prior_value) == (
        "duplicate", fact.id, "Paris")
    assert proposal.warnings == [DUPLICATE_WARNING]


@pytest.mark.parametrize("line, kind", [
    ("user | city | Lyon | 2026-01-01", "correct"),
    ("user | city | Lyon | 2026-03-01", "update"),
], ids=["same-date-correction", "later-update"])
def test_a_changed_value_warns_the_reviewer_to_check_evidence_and_date(store, queue, line, kind):
    """A proposal that changes a live value (not a restore) carries the change warning."""
    fact = store.add("user", "city", "Paris", valid_from="2026-01-01")
    [proposal] = queue.extract(line, source="chat")
    assert (proposal.kind, proposal.target_id, proposal.prior_value) == (kind, fact.id, "Paris")
    assert proposal.warnings == [CHANGE_WARNING]


# Line parsing


@pytest.mark.parametrize("line", [
    "Alex | city | Paris | 2025-02-01",
    "2025-02-01: Alex | city | Paris | 2025-02-01",
], ids=["date-field", "agreeing-prefix-and-field"])
def test_a_line_with_a_date_field_is_a_dated_statement(queue, line):
    """A fourth-field date makes the line dated, so it does not take the observation time."""
    [proposal] = queue.extract(line, source="chat", observed_at="2026-06-01")
    assert proposal.valid_from == "2025-02-01T00:00:00.000000Z"
    assert proposal.dated is True
    assert queue.get(proposal.id).dated is True


# ==========================================================================
# Statement parsing, input checks, undated re-extraction and crash recovery
# ==========================================================================


def proposal_files(queue):
    return sorted(path.name for path in queue.directory.glob("*.json"))


def approve_interrupted(queue, proposal_id, monkeypatch):
    """Approve, stopping after the fact is written but before the proposal is saved."""
    def crash(self, proposal):
        raise RuntimeError("process stopped after the fact was written")

    with monkeypatch.context() as patch:
        patch.setattr(ProposalQueue, "_save", crash)
        with pytest.raises(RuntimeError):
            queue.approve(proposal_id)


# Statement parsing


def test_structured_line_with_an_empty_fourth_field_is_undated(queue):
    """A trailing empty date field means no date: the observation time is used."""
    [proposal] = queue.extract("Alex | city | Riverton |", source="chat",
                               observed_at="2026-06-01")
    assert (proposal.entity, proposal.attribute, proposal.value) == ("Alex", "city", "Riverton")
    assert proposal.valid_from == "2026-06-01T00:00:00.000000Z"
    assert proposal.dated is False


# Input checks and line handling in extract()


def test_text_with_only_unsupported_content_is_not_empty(queue):
    """Text that holds content no rule understands yields no proposals, not an error."""
    assert queue.extract("X", source="chat") == []
    assert proposal_files(queue) == []


@pytest.mark.parametrize("source", ["", "   ", None], ids=["empty", "blank", "none"])
def test_a_missing_or_blank_source_is_rejected_before_anything_is_written(queue, source):
    """Extraction without a provenance source raises ValueError and writes nothing."""
    with pytest.raises(ValueError):
        queue.extract("Alex | city | Riverton", source=source)
    assert proposal_files(queue) == []


def test_a_line_that_fails_record_validation_does_not_stop_later_lines(queue):
    """A parsed line that is not a valid record is skipped; the lines after it still count."""
    proposals = queue.extract("My   is Paris.\nAlex | city | Riverton\nAlex | drink | tea",
                              source="chat", observed_at="2026-06-01")
    assert [(p.entity, p.attribute, p.value) for p in proposals] == [
        ("Alex", "city", "Riverton"), ("Alex", "drink", "tea")]


# Re-extracting undated lines


def test_undated_restatement_of_an_approved_line_is_a_new_proposal(store, queue, ticking_clock):
    """Only a pending proposal absorbs an undated restatement; an approved one does not."""
    [first] = queue.extract("My city is Paris.", source="chat")
    fact = queue.approve(first.id)
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id != first.id
    assert (again.status, again.kind, again.target_id) == ("pending", "duplicate", fact.id)
    assert queue.get(first.id).status == "approved"


@pytest.mark.parametrize("entity, attribute", [("Alex", "city"), ("user", "drink")],
                         ids=["other-entity-same-attribute", "same-entity-other-attribute"])
def test_undated_restatement_ignores_later_values_of_other_histories(store, queue, ticking_clock,
                                                                    entity, attribute):
    """Only a later value of the same entity and attribute makes a restatement new."""
    [first] = queue.extract("My city is Paris.", source="chat")
    store.add(entity, attribute, "Oslo", valid_from="2026-06-01", source="other")
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id
    assert len(queue.list(status="all")) == 1


def test_undated_restatement_folds_into_a_proposal_whose_approval_was_interrupted(
        store, queue, ticking_clock, monkeypatch):
    """A fact at the pending line's own time is not a later value, so the line still folds."""
    [first] = queue.extract("My city is Paris.", source="chat")
    approve_interrupted(queue, first.id, monkeypatch)
    [written] = store.all_facts()
    assert written.valid_from == first.valid_from and queue.get(first.id).status == "pending"
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id
    assert queue.approve(first.id).id == written.id
    assert len(queue.list(status="all")) == 1


# Recognizing the fact an interrupted approval wrote


def test_approving_a_duplicate_does_not_claim_a_matching_fact_added_after_it(store, queue):
    """A duplicate proposal never writes a fact, so it never adopts one written later."""
    older = store.add("Alex", "city", "Riverton", valid_from="2025-01-01", source="import")
    line = "Alex | city | Riverton | 2025-02-01"
    [proposal] = queue.extract(line, source="chat")
    assert (proposal.kind, proposal.target_id) == ("duplicate", older.id)
    store.add("Alex", "city", "Riverton", valid_from="2025-02-01", source="chat", evidence=line)
    with pytest.raises(MemoryConflictError):
        queue.approve(proposal.id)
    reloaded = queue.get(proposal.id)
    assert reloaded.status == "pending" and reloaded.result_fact_id is None


def test_recovery_recognizes_a_fact_recorded_in_the_proposals_clock_tick(store, queue):
    """A fact recorded at exactly the proposal's creation time counts as its interrupted write."""
    line = "Alex | city | Riverton | 2025-01-01"
    [proposal] = queue.extract(line, source="chat")
    written = store.add("Alex", "city", "Riverton", valid_from="2025-01-01", source="chat",
                        evidence=line, confidence=1.0, recorded_at=proposal.created_at)
    assert written.recorded_at == normalize_time(proposal.created_at)
    assert queue.approve(proposal.id).id == written.id
    assert queue.get(proposal.id).result_fact_id == written.id
    assert [f.id for f in store.all_facts()] == [written.id]


# ==========================================================================
# Approval: expected predecessors and sibling conflicts
# ==========================================================================


def shifted(timestamp, microseconds):
    """``timestamp`` moved by a whole number of microseconds, in the proposal's format."""
    instant = datetime.fromisoformat(timestamp) + timedelta(microseconds=microseconds)
    return instant.isoformat(timespec="microseconds")


# ProposalQueue._check_still_valid: a newer same-source predecessor


def test_same_source_predecessor_recorded_at_the_proposal_instant_is_expected(store, queue):
    """A newer same-source predecessor recorded exactly when the proposal was made is expected."""
    london = approved_city(queue)
    [update] = queue.extract("user | city | Paris | 2026-03-01", source="chat")
    assert (update.kind, update.target_id) == ("update", london.id)
    # Same clock tick as the proposal: recorded no earlier than the proposal was made.
    store.add("user", "city", "Oslo", valid_from="2026-02-01", source="chat",
              recorded_at=update.created_at)
    assert queue.approve(update.id).value == "Paris"
    assert queue.get(update.id).status == "approved"
    assert [f.value for f in store.history("user", "city")] == ["London", "Oslo", "Paris"]


def test_same_source_predecessor_recorded_just_before_the_proposal_is_stale(store, queue):
    """A same-source predecessor recorded one microsecond before the proposal makes it stale."""
    approved_city(queue)
    [update] = queue.extract("user | city | Paris | 2026-03-01", source="chat")
    store.add("user", "city", "Oslo", valid_from="2026-02-01", source="chat",
              recorded_at=shifted(update.created_at, -1))
    before = store.all_facts()
    with pytest.raises(MemoryConflictError):
        queue.approve(update.id)
    assert store.all_facts() == before
    assert queue.get(update.id).status == "pending"


def test_add_with_predecessor_recorded_at_the_proposal_instant_is_expected(store, queue):
    """An add proposal stays valid when a same-source earlier value appears at its creation instant."""
    [later] = queue.extract("user | city | Paris | 2026-03-01", source="chat")
    assert (later.kind, later.target_id) == ("add", None)
    store.add("user", "city", "Oslo", valid_from="2026-02-01", source="chat",
              recorded_at=later.created_at)
    assert queue.approve(later.id).value == "Paris"
    assert [f.value for f in store.history("user", "city")] == ["Oslo", "Paris"]


# ProposalQueue._check_still_valid: siblings from the same text


def test_a_same_value_sibling_from_the_same_text_gets_the_sibling_conflict(store, queue):
    """A second line repeating an approved sibling's value is refused with a conflict naming the attribute."""
    tea, again = queue.extract("I prefer tea.\nI like tea.", source="chat",
                               observed_at="2026-10-01")
    assert (tea.kind, again.kind) == ("add", "add")
    queue.approve(tea.id)
    with pytest.raises(MemoryConflictError, match="preference"):
        queue.approve(again.id)
    assert [f.value for f in store.all_facts()] == ["tea"]
    assert queue.get(again.id).status == "pending"


# ==========================================================================
# Listing skips stray files that sort before proposals
# ==========================================================================


def _two_pending(queue):
    return queue.extract("Alpha | status | one | 2026-01-01\nBeta | status | two | 2026-01-02",
                         source="synthetic:stray")


@pytest.mark.parametrize("stray", ["0.json", "00000000.json", "-notes.json", "0 copy.json"])
def test_listing_skips_stray_json_that_sorts_before_proposal_files(queue, stray):
    """A non-proposal .json file named to sort first does not hide the proposals after it."""
    created = _two_pending(queue)
    (queue.directory / stray).write_text("{}", encoding="utf-8")
    assert sorted(p.name for p in queue.directory.glob("*.json"))[0] == stray
    listed = queue.list()
    assert [p.id for p in listed] == [p.id for p in created]
    assert queue.problems == []


def test_listing_all_statuses_skips_a_leading_stray_file(queue):
    """Listing every status still returns approved and rejected proposals behind a stray file."""
    first, second = _two_pending(queue)
    queue.approve(first.id)
    queue.reject(second.id)
    (queue.directory / "0-backup.json").write_text("not json", encoding="utf-8")
    listed = queue.list(status="all")
    assert [(p.id, p.status) for p in listed] == [(first.id, "approved"),
                                                   (second.id, "rejected")]
    assert queue.problems == []


def test_reopened_queue_lists_proposals_behind_a_leading_stray_file(store, tmp_path):
    """A queue opened again over the same folder sees every proposal despite a stray file."""
    created = _two_pending(ProposalQueue(store))
    (store.root / "proposals" / "0.json").write_text("[]", encoding="utf-8")
    with MemoryStore(tmp_path / "store") as reopened:
        listed = ProposalQueue(reopened).list(status="pending")
    assert [p.id for p in listed] == [p.id for p in created]


# ==========================================================================
# Re-extraction against the extraction clock; approving again by hand
# ==========================================================================


def set_clock(monkeypatch, instant):
    """Make the extraction's "now" a fixed instant (an ISO string with a UTC offset)."""
    monkeypatch.setattr(extraction, "_now", lambda: instant)


def rewrite_proposal(queue, proposal_id, **changes):
    """Edit fields of a saved proposal file by hand, keeping the queue's file format."""
    path = queue.directory / (proposal_id + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


EARLY = "2020-01-01T00:00:00.000000+00:00"


FIRST = "2026-01-01T00:00:00.000000+00:00"


SECOND = "2026-01-03T00:00:00.000000+00:00"


def test_undated_line_takes_the_extraction_clock_time(queue, monkeypatch):
    """An undated line's effective time is the extraction clock's reading, normalized to UTC."""
    set_clock(monkeypatch, EARLY)
    [proposal] = queue.extract("My city is Paris.", source="chat")
    assert proposal.dated is False
    assert proposal.valid_from == "2020-01-01T00:00:00.000000Z"
    assert proposal.created_at == EARLY


def test_undated_line_with_a_blank_observation_time_takes_the_extraction_clock_time(
        queue, monkeypatch):
    """A blank observation time means now, read from the extraction clock."""
    set_clock(monkeypatch, EARLY)
    [proposal] = queue.extract("Alex | city | Riverton", source="chat", observed_at="  ")
    assert (proposal.valid_from, proposal.dated) == ("2020-01-01T00:00:00.000000Z", False)


def test_value_taking_effect_exactly_at_the_new_extraction_time_makes_a_restatement_new(
        store, queue, monkeypatch):
    """A value effective exactly at the second extraction's "now" has taken effect, so the
    undated restatement is a new proposal rather than the pending one."""
    set_clock(monkeypatch, FIRST)
    [first] = queue.extract("My city is Paris.", source="chat")
    assert first.valid_from == "2026-01-01T00:00:00.000000Z"
    rome = store.add("user", "city", "Rome", valid_from="2026-01-03T00:00:00Z", source="import")
    set_clock(monkeypatch, SECOND)
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id != first.id
    assert again.valid_from == "2026-01-03T00:00:00.000000Z"
    assert (again.kind, again.target_id, again.prior_value) == ("correct", rome.id, "Rome")
    assert len(queue.list()) == 2


def test_value_taking_effect_just_after_the_new_extraction_time_does_not_make_it_new(
        store, queue, monkeypatch):
    """A value planned one microsecond after the second extraction's "now" has not taken
    effect yet, so the undated restatement folds into the pending proposal."""
    set_clock(monkeypatch, FIRST)
    [first] = queue.extract("My city is Paris.", source="chat")
    store.add("user", "city", "Rome", valid_from="2026-01-03T00:00:00.000001Z", source="import")
    set_clock(monkeypatch, SECOND)
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id
    assert len(queue.list()) == 1


@pytest.mark.parametrize("entity, attribute", [("Alex", "city"), ("user", "drink")],
                         ids=["other-entity-same-attribute", "same-entity-other-attribute"])
def test_restatement_ignores_values_of_other_histories_that_took_effect_in_between(
        store, queue, monkeypatch, entity, attribute):
    """A value of another entity or attribute that took effect between two extractions does
    not make an undated restatement new information."""
    set_clock(monkeypatch, FIRST)
    [first] = queue.extract("My city is Paris.", source="chat")
    store.add(entity, attribute, "Oslo", valid_from="2026-01-02", source="other")
    set_clock(monkeypatch, SECOND)
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id
    assert len(queue.list(status="all")) == 1


def test_dated_line_does_not_fold_into_a_pending_proposal_for_another_date(queue):
    """A dated line folds only into a pending proposal at its own date, even after the
    pending proposal's effective date was edited by hand."""
    line = "user | city | Paris | 2026-01-01"
    [first] = queue.extract(line, source="chat")
    rewrite_proposal(queue, first.id, valid_from="2026-01-05T00:00:00.000000Z")
    assert queue.get(first.id).valid_from == "2026-01-05T00:00:00.000000Z"
    [again] = queue.extract(line, source="chat")
    assert again.id != first.id
    assert (again.valid_from, again.dated) == ("2026-01-01T00:00:00.000000Z", True)
    assert len(queue.list()) == 2


def test_proposal_set_back_to_pending_by_hand_is_approved_again_with_its_own_fact(
        store, queue, monkeypatch):
    """A proposal file set back to pending by hand, still naming the fact it produced, is
    approved again with that fact: only another approved proposal's claim excludes a fact."""
    set_clock(monkeypatch, EARLY)  # Made well before the approval records its fact.
    written = approved_city(queue, "user | city | Paris | 2026-01-01")
    [proposal] = queue.list(status="approved")
    rewrite_proposal(queue, proposal.id, status="pending")
    reopened = queue.get(proposal.id)
    assert (reopened.status, reopened.result_fact_id) == ("pending", written.id)
    assert queue.approve(proposal.id) == written
    assert queue.get(proposal.id).status == "approved"
    assert queue.get(proposal.id).result_fact_id == written.id
    assert store.all_facts() == [written]


# ==========================================================================
# Approval with a newer predecessor
# ==========================================================================


@pytest.mark.parametrize("other_entity", ["office", "User"])
def test_newer_predecessor_is_expected_when_another_entity_shares_the_attribute(
        store, queue, other_entity):
    """Another entity's value for the same attribute does not make an in-order approval stale."""
    store.add(other_entity, "city", "Madrid", valid_from="2026-01-01", source="other")
    london = approved_city(queue)
    [update] = queue.extract("user | city | Paris | 2026-03-01", source="chat")
    assert (update.kind, update.target_id) == ("update", london.id)
    store.add("user", "city", "Oslo", valid_from="2026-02-01", source="chat")
    approved = queue.approve(update.id)
    assert (approved.entity, approved.value) == ("user", "Paris")
    assert queue.get(update.id).status == "approved"
    assert [f.value for f in store.history("user", "city")] == ["London", "Oslo", "Paris"]
    assert [f.value for f in store.history(other_entity, "city")] == ["Madrid"]


def test_reviewed_record_counts_only_when_strictly_older_than_the_predecessor(store, queue):
    """A predecessor with the reviewed record's own effective time is not a newer one."""
    london = approved_city(queue)
    [update] = queue.extract("user | city | Paris | 2026-03-01", source="chat")
    same_time = dataclasses.replace(london, id="d" * 32, value="Rome")
    assert queue._reviewed_unchanged(update, same_time) is False
    newer = store.add("user", "city", "Oslo", valid_from="2026-02-01", source="chat")
    assert queue._reviewed_unchanged(update, newer) is True
