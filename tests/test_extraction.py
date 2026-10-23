import pytest

from temporal_memory_os import MemoryStore
from temporal_memory_os.extraction import ProposalQueue


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
        with pytest.raises(ValueError):
            queue.extract("user | city | Paris\nbroken | line", source="chat")
        assert queue.list() == []
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
