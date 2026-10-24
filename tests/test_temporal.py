"""Behavioral tests for valid time, knowledge time, and editable source records."""

import json
import re

import pytest

from temporal_memory_os.store import MemoryConflictError, MemoryStore


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path) as memory:
        yield memory


def values(store, at, known_at=None):
    return [fact.value for fact in store.snapshot(as_of=at, known_at=known_at)]


def add(store, value, effective, recorded=None, **kwargs):
    return store.add(
        entity="Example application",
        attribute="status",
        value=value,
        valid_from=effective,
        recorded_at=recorded or effective,
        source="synthetic://test/conversation",
        **kwargs,
    )


def test_history_uses_effective_time_and_half_open_boundaries(store):
    pending = add(store, "pending", "2025-09-01")
    invited = add(store, "interview invited", "2025-09-15")
    add(store, "interview completed", "2025-10-01")

    assert values(store, "2025-08-31") == []
    assert values(store, "2025-09-10") == ["pending"]
    assert values(store, "2025-09-15") == ["interview invited"]
    assert values(store, "2025-10-01") == ["interview completed"]
    history = store.history("Example application", "status")
    assert [fact.value for fact in history] == [
        "pending", "interview invited", "interview completed"
    ]
    assert history[0].valid_to == invited.valid_from
    assert store.get(pending.id).valid_to is None  # Derived intervals do not rewrite evidence.


def test_out_of_order_ingestion_builds_chronological_intervals(store):
    add(store, "completed", "2025-10-01", "2025-10-05")
    add(store, "pending", "2025-09-01", "2025-10-06")
    add(store, "invited", "2025-09-15", "2025-10-07")

    assert values(store, "2025-09-10") == ["pending"]
    assert values(store, "2025-09-20") == ["invited"]
    assert values(store, "2025-10-02") == ["completed"]


def test_correction_changes_present_belief_without_rewriting_prior_knowledge(store):
    original = add(store, "pending", "2025-09-01", "2025-09-02")
    correction = store.correct(
        original.id, "withdrawn", source="synthetic://correction",
        evidence="The earlier status was entered incorrectly.",
        recorded_at="2025-09-20",
    )

    assert values(store, "2025-09-10", "2025-09-15") == ["pending"]
    assert values(store, "2025-09-10", "2025-09-25") == ["withdrawn"]
    assert correction.supersedes == original.id
    assert correction.valid_from == original.valid_from
    assert store.get(original.id).value == "pending"
    assert len(store.all_facts()) == 2


def test_recorded_at_excludes_information_not_yet_known(store):
    add(store, "pending", "2025-09-01", "2025-09-05")
    assert values(store, "2025-09-03", "2025-09-04") == []
    assert values(store, "2025-09-03", "2025-09-06") == ["pending"]


def test_late_known_update_does_not_close_interval_prematurely(store):
    add(store, "pending", "2025-09-01", "2025-09-01")
    add(store, "invited", "2025-09-15", "2025-09-20")
    assert values(store, "2025-09-18", "2025-09-19") == ["pending"]
    assert values(store, "2025-09-18", "2025-09-21") == ["invited"]


def test_explicit_end_preserves_a_gap(store):
    add(store, "pending", "2025-09-01", valid_to="2025-09-10")
    add(store, "invited", "2025-09-20")
    assert values(store, "2025-09-09") == ["pending"]
    assert values(store, "2025-09-10") == []
    assert values(store, "2025-09-15") == []
    assert values(store, "2025-09-20") == ["invited"]


def test_retraction_preserves_gap_instead_of_reviving_previous_value(store):
    add(store, "pending", "2025-09-01")
    invited = add(store, "invited", "2025-09-15")
    add(store, "completed", "2025-10-01")
    tombstone = store.retract(
        invited.id, source="synthetic://retraction", recorded_at="2025-10-10"
    )
    assert tombstone.deleted is True
    assert values(store, "2025-09-20", "2025-10-05") == ["invited"]
    assert values(store, "2025-09-10", "2025-10-11") == ["pending"]
    assert values(store, "2025-09-20", "2025-10-11") == []
    assert values(store, "2025-10-02", "2025-10-11") == ["completed"]


def test_stale_corrections_and_duplicate_effective_updates_are_rejected(store):
    original = add(store, "pending", "2025-09-01")
    store.correct(original.id, "withdrawn", recorded_at="2025-09-20")
    # Each message names the action to take on the latest revision instead.
    with pytest.raises(MemoryConflictError, match="correct the latest revision instead"):
        store.correct(original.id, "invited", recorded_at="2025-09-21")
    with pytest.raises(MemoryConflictError, match="retract the latest revision instead"):
        store.retract(original.id, recorded_at="2025-09-21")
    with pytest.raises(MemoryConflictError, match="use correct"):
        add(store, "another value", "2025-09-01", "2025-09-22")


def test_timezone_offsets_are_compared_as_instants(store):
    add(store, "pending", "2025-09-01T08:00:00+08:00")
    assert values(store, "2025-09-01T00:00:00Z") == ["pending"]
    assert values(store, "2025-08-31T23:59:59Z") == []


@pytest.mark.parametrize("topic", ["../escape", "/absolute", "career/../../escape"])
def test_topics_cannot_escape_memory_root(store, topic):
    with pytest.raises(ValueError):
        add(store, "pending", "2025-09-01", topic=topic)


def test_invalid_interval_is_rejected(store):
    with pytest.raises(ValueError):
        add(store, "pending", "2025-09-01", valid_to="2025-08-01")


def test_manual_markdown_edit_is_reindexed_and_deleted_file_disappears(store, tmp_path):
    fact = add(store, "pending", "2025-09-01", topic="career/applications")
    path = tmp_path / "memory" / "career" / "applications" / (fact.id + ".md")
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    assert match, "A fact's canonical JSON block must be editable in Markdown."
    payload = json.loads(match.group(1))
    payload["value"] = "manually corrected"
    path.write_text(
        text[:match.start(1)] + json.dumps(payload, indent=2) + text[match.end(1):],
        encoding="utf-8",
    )
    store.reindex()
    assert store.get(fact.id).value == "manually corrected"
    path.unlink()
    store.reindex()
    assert store.snapshot(as_of="2025-09-02") == []


def test_invalid_markdown_rebuild_keeps_previous_index(store, tmp_path):
    fact = add(store, "pending", "2025-09-01")
    path = tmp_path / "memory" / "general" / (fact.id + ".md")
    path.write_text('<!-- memoryos:fact:v1 -->\n```json\n{broken json}\n```\n', encoding="utf-8")
    with pytest.raises(ValueError):
        store.reindex()
    assert store.get(fact.id).value == "pending"


def test_provenance_and_relations_survive_reopening(tmp_path):
    with MemoryStore(tmp_path) as store:
        fact = add(
            store, "pending", "2025-09-01", topic="career/applications",
            evidence="The candidate says the application is pending.",
            tags=["career", "synthetic"], links=["Example employer"], confidence=0.8,
        )
    with MemoryStore(tmp_path) as store:
        restored = store.get(fact.id)
        assert restored.source == "synthetic://test/conversation"
        assert restored.evidence == fact.evidence
        assert restored.tags == ("career", "synthetic")
        assert restored.links == ("Example employer",)
        assert restored.confidence == 0.8
