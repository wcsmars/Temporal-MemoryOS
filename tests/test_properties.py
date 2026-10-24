"""Property-based tests: random histories checked against an independent model.

The store resolves time by building intervals and then filtering them. The
reference model below answers each point query directly from the documented
rules instead, so an error shared by both would have to be made twice in
different ways:

* only records learned by ``known_at`` exist;
* a record replaced by a visible correction does not count;
* the latest remaining assertion starting at or before ``as_of`` decides the
  answer, unless it is a retraction or its explicit end has passed.

The model also predicts which writes must be rejected, and with which error.
Every history starts with a seeded batch of writes, so it never stays empty.
After every step, full snapshots at a few fixed time pairs are compared with
the model; random steps compare other snapshots, histories and searches; and
the final state is compared with the model at every query time pair.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine, initialize, invariant, precondition, rule,
)

from temporal_memory_os.embeddings import HashingEmbedder
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.models import Fact, normalize_time
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.store import MemoryConflictError, MemoryStore

# Few distinct days force collisions between effective and recorded times.
DAYS = ["2025-01-0{}".format(day) for day in range(1, 9)]
QUERY_TIMES = ["2024-12-31", *DAYS, "2025-01-03T12:00:00Z", "2025-01-10"]
# Offsets in days. Zero makes an invalid end or, for a revision, a same-day record;
# -1 dates a correction before its target. Both must be handled.
SPANS = st.one_of(st.none(), st.integers(min_value=0, max_value=4))
DELAYS = st.integers(min_value=-1, max_value=4)


def shifted(day: str, offset: int) -> str:
    return DAYS[max(0, min(len(DAYS) - 1, DAYS.index(day) + offset))]
ENTITIES = ["Acme application", "Alex"]
ATTRIBUTES = ["status", "home city"]
TOPICS = ["general", "career/applications", "profile/a.b-c"]
SEARCH_WORDS = ["acme", "application", "alex", "status", "home", "city", "pending", "tea"]

# Free text with Markdown fences, metadata markers, quotes, Unicode and newlines.
TEXT = st.text(
    alphabet=st.sampled_from(list("abcxyz ABC|`#<>!-:{}\"'\\\n\t") + ["é", "完", "😀"]),
    min_size=1, max_size=24,
).filter(lambda text: text.strip() != "")
VALUES = st.one_of(st.sampled_from(["pending", "tea", "Riverton", "<!-- memoryos:fact:v1 -->",
                                    "```", "```json"]), TEXT)
# Arguments of one add, shared by the add rule and the seeded first batch.
ADD_ARGUMENTS = dict(
    entity=st.sampled_from(ENTITIES), attribute=st.sampled_from(ATTRIBUTES), value=VALUES,
    start=st.sampled_from(DAYS[:6]), span=SPANS, recorded=st.sampled_from(DAYS),
    topic=st.sampled_from(TOPICS), evidence=TEXT,
    links=st.lists(st.sampled_from(ENTITIES), max_size=2),
)
# Seeded adds never have an empty span, so the first one always succeeds. The
# add rule still draws empty spans and other rejected writes.
SEED_ADD = st.fixed_dictionaries(
    {**ADD_ARGUMENTS, "span": st.one_of(st.none(), st.integers(min_value=1, max_value=4))})
# (as_of, known_at) pairs compared with the model after every step: before any
# assertion starts, mid-history with partial and with full knowledge, and the end.
CHECKED_EVERY_STEP = [("2024-12-31", "2025-01-08"), ("2025-01-04", "2025-01-03"),
                      ("2025-01-03T12:00:00Z", "2025-01-08"), ("2025-01-10", "2025-01-08")]


@dataclass(frozen=True)
class Expected:
    """What a query should return for one entity/attribute pair."""

    id: str
    value: str
    valid_to: Optional[str]


class Model:
    """Reference semantics, written without reference to the store's code."""

    def __init__(self) -> None:
        self.records: Dict[str, Fact] = {}

    def superseded(self, known: Optional[str] = None) -> set:
        return {fact.supersedes for fact in self.records.values()
                if fact.supersedes and (known is None or fact.recorded_at <= known)}

    def add_error(self, entity: str, attribute: str, valid_from: str,
                  valid_to: Optional[str]) -> Optional[type]:
        if valid_to is not None and valid_to <= valid_from:
            return ValueError
        replaced = self.superseded()
        if any((fact.entity, fact.attribute, fact.valid_from) == (entity, attribute, valid_from)
               and fact.id not in replaced for fact in self.records.values()):
            return MemoryConflictError
        return None

    def revision_error(self, target: Fact, recorded_at: str, retract: bool) -> Optional[type]:
        if target.id in self.superseded():
            return MemoryConflictError
        if retract and target.deleted:
            return MemoryConflictError
        if recorded_at < target.recorded_at:
            return ValueError
        return None

    def answer(self, entity: str, attribute: str, as_of: str, known: str) -> Optional[Expected]:
        replaced = self.superseded(known)
        live = [fact for fact in self.records.values()
                if (fact.entity, fact.attribute) == (entity, attribute)
                and fact.recorded_at <= known and fact.id not in replaced]
        started = [fact for fact in live if fact.valid_from <= as_of]
        if not started:
            return None
        latest = max(fact.valid_from for fact in started)
        deciding = [fact for fact in started if fact.valid_from == latest]
        assert len(deciding) == 1, "two live assertions share an effective start"
        fact = deciding[0]
        if fact.deleted or (fact.valid_to is not None and as_of >= fact.valid_to):
            return None
        later = [other.valid_from for other in live if other.valid_from > fact.valid_from]
        ends = [end for end in (fact.valid_to, min(later, default=None)) if end is not None]
        return Expected(fact.id, fact.value, min(ends) if ends else None)


    def timeline(self, entity: str, attribute: str,
                 known: str) -> List[Tuple[str, str, Optional[str], bool]]:
        """The effective history known at ``known``, retractions included."""
        replaced = self.superseded(known)
        live = sorted((fact for fact in self.records.values()
                       if (fact.entity, fact.attribute) == (entity, attribute)
                       and fact.recorded_at <= known and fact.id not in replaced),
                      key=lambda fact: fact.valid_from)
        result = []
        for index, fact in enumerate(live):
            ends = [end for end in (fact.valid_to,
                                    live[index + 1].valid_from if index + 1 < len(live) else None)
                    if end is not None]
            result.append((fact.id, fact.value, min(ends) if ends else None, fact.deleted))
        return result


class TemporalMemoryMachine(RuleBasedStateMachine):
    """Random histories of adds, corrections, retractions and searches."""

    def __init__(self) -> None:
        super().__init__()
        self.root = Path(tempfile.mkdtemp(prefix="memoryos-property-"))
        self.store = MemoryStore(self.root)
        self.retriever = HybridRetriever(self.store, embedder=HashingEmbedder())
        self.model = Model()
        self.created: List[str] = []  # Creation order keeps replays deterministic.

    def teardown(self) -> None:
        try:
            # Hypothesis calls teardown from a ``finally`` block. When a step has
            # already failed, comparing again would only chain a second error onto
            # the first, which Hypothesis reports and shrinks as another failure.
            if sys.exc_info()[1] is not None:
                return
            event("records at end: {}".format(
                "none" if not self.model.records else "1-4" if len(self.model.records) < 5
                else "5 or more"))
            # The final state answers every query time pair as the model does.
            expected = self._all_answers(self.store)
            assert expected == self._model_answers()
            # Markdown is the source of truth: a fresh process and a forced
            # rebuild must both reproduce every answer.
            self.store.reindex()
            assert self._all_answers(self.store) == expected
            with MemoryStore(self.root) as reopened:
                assert reopened.all_facts() == self.store.all_facts()
                assert self._all_answers(reopened) == expected
        finally:
            self.store.close()
            shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def _all_answers(store: MemoryStore) -> List[Tuple[str, str, List[Tuple[str, str, str]]]]:
        return [(as_of, known, sorted((fact.id, fact.value, str(fact.valid_to))
                                      for fact in store.snapshot(as_of=as_of, known_at=known)))
                for as_of in QUERY_TIMES for known in DAYS]

    def _model_answers(self) -> List[Tuple[str, str, List[Tuple[str, str, str]]]]:
        """The model's version of :meth:`_all_answers`."""
        return [(as_of, known, sorted((fact.id, fact.value, str(fact.valid_to))
                                      for fact in self._expected_snapshot(as_of, known).values()))
                for as_of in QUERY_TIMES for known in DAYS]

    def _expected_snapshot(self, as_of: str, known: str) -> Dict[Tuple[str, str], Fact]:
        """Full records the model expects in a snapshot, with their derived ends."""
        moment, cutoff = normalize_time(as_of), normalize_time(known)
        expected = {}
        for entity in ENTITIES:
            for attribute in ATTRIBUTES:
                answer = self.model.answer(entity, attribute, moment, cutoff)
                if answer is not None:
                    record = self.model.records[answer.id]
                    assert record.value == answer.value
                    expected[(entity, attribute)] = replace(record, valid_to=answer.valid_to)
        return expected

    def _check_snapshot(self, as_of: str, known: str) -> int:
        snapshot = self.store.snapshot(as_of=as_of, known_at=known)
        actual = {(fact.entity, fact.attribute): fact for fact in snapshot}
        assert len(actual) == len(snapshot), \
            "a snapshot holds at most one value per entity/attribute"
        expected = self._expected_snapshot(as_of, known)
        assert actual == expected
        return len(expected)

    def _attempt(self, expected_error: Optional[type], write) -> Optional[Fact]:
        event("write rejected: " + (expected_error.__name__ if expected_error else "no"))
        if expected_error is None:
            return write()
        with pytest.raises(ValueError) as caught:
            write()
        assert type(caught.value) is expected_error, caught.value
        return None

    # Swarm testing disables a random subset of rules in each history, often
    # including ``add``. Initialize rules always run, so every history starts
    # with at least one record.
    @initialize(batch=st.lists(SEED_ADD, min_size=1, max_size=6))
    def seed(self, batch: List[dict]) -> None:
        for arguments in batch:
            self._add(**arguments)
        event("seeded records: {}".format(len(self.model.records)))

    @rule(**ADD_ARGUMENTS)
    def add(self, entity: str, attribute: str, value: str, start: str, span: Optional[int],
            recorded: str, topic: str, evidence: str, links: List[str]) -> None:
        self._add(entity, attribute, value, start, span, recorded, topic, evidence, links)

    def _add(self, entity: str, attribute: str, value: str, start: str, span: Optional[int],
             recorded: str, topic: str, evidence: str, links: List[str]) -> None:
        end = None if span is None else shifted(start, span)
        valid_from = normalize_time(start)
        valid_to = None if end is None else normalize_time(end)
        error = self.model.add_error(entity, attribute, valid_from, valid_to)
        fact = self._attempt(error, lambda: self.store.add(
            entity, attribute, value, valid_from=start, valid_to=end, recorded_at=recorded,
            source="synthetic:property", evidence=evidence, topic=topic,
            links=[link for link in links if link != entity]))
        if fact is not None:
            assert (fact.entity, fact.attribute, fact.value, fact.valid_from, fact.valid_to,
                    fact.recorded_at, fact.supersedes, fact.deleted) == (
                entity, attribute, value, valid_from, valid_to, normalize_time(recorded), None,
                False)
            self.model.records[fact.id] = fact
            self.created.append(fact.id)

    @precondition(lambda self: self.model.records)
    @rule(data=st.data(), value=VALUES, delay=DELAYS, retract=st.booleans(), evidence=TEXT)
    def revise(self, data: st.DataObject, value: str, delay: int, retract: bool,
               evidence: str) -> None:
        target = self.model.records[self.created[data.draw(
            st.integers(min_value=0, max_value=len(self.created) - 1))]]
        # A target's own day when the delay is zero; any earlier day is invalid.
        recorded = shifted(target.recorded_at[:10], delay)
        error = self.model.revision_error(target, normalize_time(recorded), retract)
        if retract:
            fact = self._attempt(error, lambda: self.store.retract(
                target.id, source="synthetic:retraction", recorded_at=recorded))
        else:
            fact = self._attempt(error, lambda: self.store.correct(
                target.id, value, source="synthetic:correction", evidence=evidence,
                recorded_at=recorded))
        if fact is not None:
            expected = replace(
                target, id=fact.id, value="" if retract else value,
                recorded_at=normalize_time(recorded), supersedes=target.id, deleted=retract,
                source="synthetic:retraction" if retract else "synthetic:correction",
                evidence="Retracted assertion" if retract else evidence)
            assert fact == expected
            self.model.records[fact.id] = fact
            self.created.append(fact.id)

    @rule(as_of=st.sampled_from(QUERY_TIMES), known=st.sampled_from(DAYS))
    def snapshot_matches_model(self, as_of: str, known: str) -> None:
        answers = self._check_snapshot(as_of, known)
        event("snapshot answers: {}".format(min(answers, 3)))

    @rule(entity=st.sampled_from(ENTITIES), known=st.sampled_from(DAYS),
          only=st.one_of(st.none(), st.sampled_from(ATTRIBUTES)))
    def history_matches_model(self, entity: str, known: str, only: Optional[str]) -> None:
        history = self.store.history(entity, attribute=only, known_at=known)
        cutoff = normalize_time(known)
        assert {fact.attribute for fact in history} <= ({only} if only else set(ATTRIBUTES))
        for attribute in ATTRIBUTES if only is None else [only]:
            timeline = [fact for fact in history if fact.attribute == attribute]
            assert [(fact.id, fact.value, fact.valid_to, fact.deleted) for fact in timeline] == \
                self.model.timeline(entity, attribute, cutoff)
            starts = [fact.valid_from for fact in timeline]
            assert starts == sorted(starts) and len(set(starts)) == len(starts)
            for fact, following in zip(timeline, timeline[1:]):
                assert fact.valid_to is not None and fact.valid_to <= following.valid_from

    @rule(entity=st.sampled_from(ENTITIES), attribute=st.sampled_from(ATTRIBUTES))
    def audit_trail_lists_every_revision_after_the_one_it_replaces(self, entity: str,
                                                                    attribute: str) -> None:
        trail = self.store.audit_trail(entity, attribute)
        assert sorted(fact.id for fact in trail) == sorted(
            fact.id for fact in self.model.records.values()
            if (fact.entity, fact.attribute) == (entity, attribute))
        assert all(fact == self.model.records[fact.id] for fact in trail)
        recorded = [fact.recorded_at for fact in trail]
        assert recorded == sorted(recorded)
        # Same-day revisions share a timestamp with their target; order must still hold.
        position = {fact.id: index for index, fact in enumerate(trail)}
        assert all(position[fact.supersedes] < position[fact.id]
                   for fact in trail if fact.supersedes is not None)

    @rule(words=st.lists(st.sampled_from(SEARCH_WORDS), min_size=1, max_size=3),
          as_of=st.sampled_from(QUERY_TIMES), known=st.sampled_from(DAYS),
          mode=st.sampled_from(HybridRetriever.available_modes),
          entity=st.one_of(st.none(), st.sampled_from(ENTITIES)), limit=st.integers(1, 5))
    def search_only_returns_time_valid_facts(self, words: List[str], as_of: str, known: str,
                                             mode: str, entity: Optional[str],
                                             limit: int) -> None:
        eligible = {fact.id: fact for fact in self.store.snapshot(as_of=as_of, known_at=known)}
        results = self.retriever.search(" ".join(words), as_of=as_of, known_at=known,
                                        limit=limit, mode=mode, entity=entity)
        assert len(results) <= limit
        assert [hit.score for hit in results] == sorted((hit.score for hit in results),
                                                        reverse=True)
        assert len({hit.fact.id for hit in results}) == len(results)
        event("search results: {}".format("some" if results else "none"))
        for hit in results:
            assert hit.fact == eligible[hit.fact.id]
            assert entity is None or hit.fact.entity == entity
            assert hit.citation == self.store.path_for(hit.fact.id).relative_to(
                self.store.root).as_posix()

    @invariant()
    def fixed_snapshots_match_model(self) -> None:
        for as_of, known in CHECKED_EVERY_STEP:
            self._check_snapshot(as_of, known)

    @invariant()
    def every_record_is_one_markdown_file(self) -> None:
        files = sorted(path.stem for path in (self.root / "memory").rglob("*.md"))
        assert files == sorted(self.model.records)


# Example counts come from the active profile (see conftest.py).
TemporalMemoryMachine.TestCase.settings = settings(settings(), stateful_step_count=40)
test_temporal_memory_matches_reference_model = TemporalMemoryMachine.TestCase


OFFSETS = st.integers(min_value=-14 * 60, max_value=14 * 60).map(
    lambda minutes: timezone(timedelta(minutes=minutes)))
INSTANTS = st.builds(
    lambda naive, zone: naive.replace(tzinfo=zone),
    st.datetimes(min_value=datetime(1900, 1, 2), max_value=datetime(2200, 12, 30)), OFFSETS)


@given(INSTANTS, INSTANTS)
def test_normalized_times_sort_like_the_instants_they_name(first, second):
    # SQLite and the store compare these strings directly.
    left, right = normalize_time(first), normalize_time(second)
    assert (left < right) == (first < second)
    assert (left == right) == (first == second)


@given(INSTANTS)
def test_time_normalization_is_idempotent_and_lossless_to_the_microsecond(instant):
    text = normalize_time(instant)
    assert normalize_time(text) == text
    assert datetime.fromisoformat(text.replace("Z", "+00:00")) == instant
    assert text.endswith("Z") and len(text) == len("2025-01-01T00:00:00.000000Z")


@given(st.dates(min_value=datetime(1900, 1, 1).date(), max_value=datetime(2200, 1, 1).date()))
def test_a_date_means_midnight_utc(day):
    assert normalize_time(day.isoformat()) == normalize_time(day) == "{}T00:00:00.000000Z".format(
        day.isoformat())


FIELD = st.text(min_size=1, max_size=40).filter(lambda text: text.strip() != "")


@given(entity=FIELD, attribute=FIELD, value=FIELD, evidence=st.text(max_size=80),
       source=FIELD, tags=st.lists(FIELD, max_size=3), links=st.lists(FIELD, max_size=3),
       deleted=st.booleans())
def test_markdown_records_round_trip_any_text(entity, attribute, value, evidence, source,
                                               tags, links, deleted):
    fact = Fact(id="a" * 32, entity=entity, attribute=attribute,
                value="" if deleted else value, valid_from="2025-01-01",
                recorded_at="2025-01-02", source=source, evidence=evidence, tags=tags,
                links=links, deleted=deleted)
    raw = MemoryStore._render(fact).encode("utf-8")
    assert MemoryStore._parse(Path("record.md"), raw) == fact


LINE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc", "Zl", "Zp"), blacklist_characters="|"),
    min_size=1, max_size=30,
).filter(lambda text: text.strip() != "")


@settings(max_examples=40, suppress_health_check=[
    *settings().suppress_health_check, HealthCheck.function_scoped_fixture])
@given(entity=LINE_TEXT, attribute=LINE_TEXT, value=LINE_TEXT,
       day=st.sampled_from(DAYS))
def test_structured_lines_become_exactly_one_reviewable_proposal(tmp_path_factory, entity,
                                                                 attribute, value, day):
    assume(not entity.strip().startswith("#"))
    with MemoryStore(tmp_path_factory.mktemp("extract")) as store:
        queue = ProposalQueue(store)
        line = "{} | {} | {} | {}".format(entity, attribute, value, day)
        proposals = queue.extract(line, source="synthetic:property")
        assert [(p.entity, p.attribute, p.value, p.valid_from, p.kind) for p in proposals] == [
            (entity.strip(), attribute.strip(), value.strip(), normalize_time(day), "add")]
        assert store.all_facts() == []  # Extraction never writes memory.
        fact = queue.approve(proposals[0].id)
        assert store.snapshot(as_of=day, known_at="2100-01-01") == [fact]
