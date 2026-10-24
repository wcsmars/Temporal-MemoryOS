"""Conservative offline extraction into an explicit human review queue.

This is a rule-based extractor, not an LLM. Lines it does not understand are
skipped, including lines with ``|`` that are not a well-formed structured fact
and lines whose date prefix is not a valid date. For arbitrary assertions,
use: entity | attribute | value | optional ISO date.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple
from uuid import uuid4

from .models import Fact, _text, normalize_time, validate_topic
from .store import MemoryConflictError, MemoryStore, _flush_directory

_PROPOSAL_ID = re.compile(r"[a-f0-9]{32}")
_STATUSES = ("pending", "approved", "rejected")
_KINDS = ("add", "update", "correct", "duplicate")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _after(previous: str) -> str:
    """The current time, but strictly later than ``previous``.

    Proposals are listed by creation time, so a batch extracted within one
    clock tick must still list in the order its lines appeared.
    """
    now = _now()
    if previous and now <= previous:
        later = datetime.fromisoformat(normalize_time(previous).replace("Z", "+00:00"))
        return (later + timedelta(microseconds=1)).isoformat(timespec="microseconds")
    return now


def _fingerprint(fact: Fact) -> str:
    """A hash of a stored record, so an edit that keeps its ID is still noticed."""
    return hashlib.sha256(json.dumps(fact.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()


def _well_typed(proposal: "Proposal") -> bool:
    text = ("id", "entity", "attribute", "value", "valid_from", "source", "evidence", "topic",
            "created_at", "status", "kind")
    optional = ("target_id", "target_fingerprint", "prior_value", "result_fact_id")
    if not all(isinstance(getattr(proposal, name), str) for name in text):
        return False
    if not all(getattr(proposal, name) is None or isinstance(getattr(proposal, name), str)
               for name in optional):
        return False
    if (isinstance(proposal.confidence, bool) or not isinstance(proposal.confidence, (int, float))
            or not isinstance(proposal.dated, bool) or not isinstance(proposal.warnings, list)
            or not all(isinstance(item, str) for item in proposal.warnings)):
        return False
    try:
        # The package's own grammar, so validity does not depend on the Python version.
        normalize_time(proposal.created_at)
        normalize_time(proposal.valid_from)
    except ValueError:
        return False
    if proposal.kind in ("update", "correct", "duplicate") and not proposal.target_id:
        return False
    if proposal.status == "approved" and not proposal.result_fact_id:
        return False
    return True


def _required(value: Optional[str], name: str) -> str:
    if not value:
        raise ValueError("Proposal file is missing {}; reject it and extract again".format(name))
    return value


@dataclass
class Proposal:
    id: str
    entity: str
    attribute: str
    value: str
    valid_from: str
    source: str
    evidence: str
    topic: str
    created_at: str
    status: str = "pending"
    kind: str = "add"
    confidence: float = 0.8
    target_id: Optional[str] = None
    target_fingerprint: Optional[str] = None
    prior_value: Optional[str] = None
    result_fact_id: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    # False when the line had no date and took the observation time; such a
    # line is the same statement whenever it is extracted again.
    dated: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ProposalQueue:
    """Persist proposals separately; only approve() changes the fact store."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.directory = Path(store.root) / "proposals"
        if self.directory.is_symlink():
            raise ValueError("Proposal directory must not be a symbolic link")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.problems: List[str] = []  # Damaged files skipped by the last list().

    @contextmanager
    def _lock(self) -> Iterator[None]:
        # Same POSIX lock discipline as the store; approvals serialize across API workers.
        import fcntl
        lock_path = self.directory / ".lock"
        if lock_path.is_symlink():
            raise ValueError("Proposal lock must not be a symbolic link")
        with lock_path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _path(self, proposal_id: str) -> Path:
        if not isinstance(proposal_id, str) or not _PROPOSAL_ID.fullmatch(proposal_id):
            raise ValueError("Invalid proposal ID")
        path = self.directory / (proposal_id + ".json")
        if path.is_symlink():
            raise ValueError("Proposal files must not be symbolic links")
        return path

    def _save(self, proposal: Proposal) -> None:
        destination = self._path(proposal.id)
        fd, temporary = tempfile.mkstemp(dir=self.directory, prefix=".proposal-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(proposal.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _flush_directory(self.directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get(self, proposal_id: str) -> Proposal:
        path = self._path(proposal_id)
        if not path.exists():
            raise KeyError(f"Proposal not found: {proposal_id}")
        name = "proposals/" + path.name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            # Very deep nesting raises RecursionError.
            raise ValueError("{} is not valid JSON; fix or delete it".format(name)) from exc
        try:
            # JSON can escape a lone surrogate ("\ud800"), which no output can encode.
            json.dumps(data, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("{} contains text that is not valid Unicode; fix or delete "
                             "it".format(name)) from exc
        known = {item.name for item in fields(Proposal)}
        if not isinstance(data, dict) or not known.issuperset(data):
            raise ValueError("{} is not a proposal file; fix or delete it".format(name))
        try:
            proposal = Proposal(**data)
        except TypeError as exc:
            raise ValueError("{} is missing proposal fields; fix or delete it".format(name)) from exc
        if (proposal.id != proposal_id or proposal.status not in _STATUSES
                or proposal.kind not in _KINDS or not _well_typed(proposal)):
            raise ValueError("{} has an inconsistent id, status, kind or field type; fix or "
                             "delete it".format(name))
        # A hand-edited time may use any accepted form; it is compared with stored ones.
        proposal.valid_from = normalize_time(proposal.valid_from)
        return proposal

    # ``list`` is a method name here, so annotations in this class use ``List``.
    def list(self, status: Optional[str] = "pending") -> List[Proposal]:
        """Proposals in creation order. Files not named like proposals are ignored.

        A damaged proposal file is left out and described in ``self.problems``,
        so one bad file does not block reviewing or extracting the others.
        """
        if status not in (None, "all", "pending", "approved", "rejected"):
            raise ValueError("status must be pending, approved, rejected, or all")
        proposals, self.problems = [], []
        for path in sorted(self.directory.glob("*.json")):
            if not _PROPOSAL_ID.fullmatch(path.stem):
                continue
            try:
                proposals.append(self.get(path.stem))
            except (ValueError, OSError) as exc:
                self.problems.append(str(exc))
        return sorted([p for p in proposals if status in (None, "all") or p.status == status],
                      key=lambda p: (normalize_time(p.created_at), p.id))

    def _classify(self, entity: str, attribute: str, value: str, valid_from: str) -> Dict[str, Any]:
        facts = [f for f in self.store.snapshot(as_of=valid_from, entity=entity)
                 if f.attribute == attribute]
        restoring = False
        if not facts:
            # A tombstone still owns its effective timestamp. Restoring it is
            # an auditable correction, not a conflicting second assertion.
            facts = [f for f in self.store.history(entity, attribute)
                     if f.deleted and f.valid_from == valid_from]
            if not facts:
                return {"kind": "add"}
            restoring = True
        prior = facts[0]
        fingerprint = _fingerprint(self.store.get(prior.id))
        if prior.value == value:
            return {"kind": "duplicate", "target_id": prior.id, "prior_value": prior.value,
                    "target_fingerprint": fingerprint,
                    "warnings": ["The same value is already effective on this date."]}
        correction = prior.valid_from == valid_from
        warning = ("Restores an assertion that was retracted at this effective time."
                   if restoring else
                   "Changes an existing value; review the evidence and effective date.")
        return {"kind": "correct" if correction else "update", "target_id": prior.id,
                "target_fingerprint": fingerprint,
                "prior_value": None if restoring else prior.value,
                "warnings": [warning]}

    @staticmethod
    def _parse_line(raw: str, observed: str
                    ) -> Optional[Tuple[str, str, str, str, float, bool]]:
        """Return (entity, attribute, value, effective, confidence, dated) or None."""
        effective, dated, statement = observed, False, raw
        # Any script's digits are recognized as a date prefix here, so a prefix that is
        # not a valid ISO date (normalize_time accepts ASCII digits only) skips the line
        # instead of leaking into the entity name.
        date_match = re.match(r"^(\d{4}-\d{2}-\d{2}):\s*(.*)$", raw)
        if date_match:
            try:
                effective, dated = normalize_time(date_match[1]), True
            except ValueError:
                return None  # Not a valid date, so not a dated statement we support.
            statement = date_match[2].strip()
            if statement.startswith("#"):
                return None  # A dated comment.
        parsed = ProposalQueue._parse_statement(statement)
        if parsed is None:
            return None
        entity, attribute, value, confidence, field_date = parsed
        if field_date:
            try:
                field_effective = normalize_time(field_date)
            except ValueError:
                return None
            if dated and field_effective != effective:
                return None  # Two different dates on one line: ambiguous, so skipped.
            effective, dated = field_effective, True
        return entity, attribute, value, effective, confidence, dated

    @staticmethod
    def _parse_statement(statement: str) -> Optional[Tuple[str, str, str, float, str]]:
        """Return (entity, attribute, value, confidence, date field or "") or None."""
        if "|" in statement:
            parts = [part.strip() for part in statement.split("|")]
            if len(parts) not in (3, 4) or not all(parts[:3]):
                return None  # Prose or a table that happens to contain "|".
            return parts[0], parts[1], parts[2], 1.0, parts[3] if len(parts) == 4 else ""
        # The name is matched greedily up to the last " is ", so names such as
        # "This Is Corp" stay whole.
        # Prose patterns. A value made only of punctuation is not a statement.
        match = re.fullmatch(r"(?:Application (.+)|My application (?:at|to) (.+)) is (.+?)[.!]?",
                             statement, re.I)
        if match and re.search(r"\w", match[3]):
            return ((match[1] or match[2]).strip() + " application", "status",
                    match[3].strip(), 0.85, "")
        match = re.fullmatch(r"My ([\w -]{1,60}?) is (.+?)[.!]?", statement, re.I)
        if match and re.search(r"\w", match[2]):
            return "user", match[1].strip().lower(), match[2].strip(), 0.85, ""
        match = re.fullmatch(r"I (?:prefer|like) (.+?)[.!]?", statement, re.I)
        if match and re.search(r"\w", match[1]):
            return "user", "preference", match[1].strip(), 0.75, ""
        return None

    def extract(self, text: str, source: str, observed_at: Optional[str] = None,
                topic: str = "general") -> List[Proposal]:
        if not isinstance(text, str) or not text.lstrip("\ufeff").strip():
            raise ValueError("Conversation text must not be empty")
        if len(text) > 1_000_000:
            raise ValueError("Conversation exceeds the 1 MB extraction limit")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("A provenance source is required")
        # Rejected up front: every line's record would fail validation on them.
        _text(text, "Conversation text")
        _text(source, "source")
        validate_topic(topic)
        if isinstance(observed_at, str) and not observed_at.strip():
            observed_at = None  # An empty time means "now", as when it is omitted.
        observed = normalize_time(_now() if observed_at is None else observed_at)
        candidates = []
        for line in text.lstrip("\ufeff").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parsed = self._parse_line(raw, observed)
            if parsed is None:
                continue
            entity, attribute, value, effective, confidence, dated = parsed
            # Validate the candidate using the exact same rules as stored facts.
            try:
                Fact.from_dict(dict(id=uuid4().hex, entity=entity, attribute=attribute,
                                    value=value, valid_from=effective, valid_to=None,
                                    recorded_at=_now(), source=source, evidence=raw,
                                    confidence=confidence, tags=[], links=[], topic=topic,
                                    supersedes=None, deleted=False))
            except ValueError:
                continue
            candidates.append((entity, attribute, value, effective, raw, confidence, dated))
        # Each attribute holds one value at a time: two lines giving the same
        # attribute for the same instant cannot both be approved.
        instants: Dict[Tuple[str, str, str], int] = {}
        for entity, attribute, _, effective, *_ in candidates:
            instants[(entity, attribute, effective)] = instants.get(
                (entity, attribute, effective), 0) + 1
        with self._lock():
            self.store.refresh()
            existing = self.list(status="all")
            results = []
            created = ""
            for entity, attribute, value, effective, raw, confidence, dated in candidates:
                duplicate = next((p for p in existing if p.status != "rejected" and
                                  (p.entity, p.attribute, p.value, p.source, p.evidence, p.topic)
                                  == (entity, attribute, value, source, raw, topic)
                                  and (p.valid_from == effective
                                       or self._same_undated(p, dated, observed_at, observed))),
                                 None)
                if duplicate:
                    results.append(duplicate)
                    continue
                classified = self._classify(entity, attribute, value, effective)
                if instants[(entity, attribute, effective)] > 1:
                    classified.setdefault("warnings", []).append(
                        "Another line in this text gives {} for the same time; only one can be "
                        "approved.".format(attribute))
                proposal = Proposal(id=uuid4().hex, entity=entity, attribute=attribute, value=value,
                                    valid_from=effective, source=source, evidence=raw, topic=topic,
                                    created_at=(created := _after(created)),
                                    confidence=confidence, dated=dated, **classified)
                self._save(proposal)
                existing.append(proposal)
                results.append(proposal)
            return results

    def _same_undated(self, pending: Proposal, dated: bool, observed_at: Optional[str],
                      now: str) -> bool:
        """An undated line extracted again "now" is the statement already pending,
        unless memory has moved past it since (then it is new information). A value
        that takes effect only after ``now`` has not moved past it yet."""
        # The line's own flag decides: the evidence is the same line, and proposal
        # files written before the flag existed load as dated.
        if dated or pending.status != "pending" or observed_at is not None:
            return False
        later = [fact for fact in self.store.history(pending.entity, pending.attribute)
                 if pending.valid_from < fact.valid_from <= now]
        return not later

    def _produced(self, proposal: Proposal) -> Optional[Fact]:
        """The fact an interrupted approval already wrote, if any.

        Fact writes and proposal status are separate files; a crash between
        them leaves a pending proposal whose fact exists. Approving again
        recognizes that fact instead of reporting a conflict.
        """
        if proposal.kind == "duplicate":
            return None
        created = normalize_time(proposal.created_at)
        claimed = {other.result_fact_id for other in self.list(status="approved")}
        for fact in self.store.audit_trail(proposal.entity, proposal.attribute):
            if ((fact.value, fact.valid_from, fact.source, fact.evidence, fact.confidence)
                    == (proposal.value, proposal.valid_from, proposal.source, proposal.evidence,
                        proposal.confidence)
                    and not fact.deleted and fact.recorded_at >= created
                    and fact.id not in claimed
                    and (fact.supersedes == proposal.target_id if proposal.kind == "correct"
                         else fact.supersedes is None and fact.topic == proposal.topic)):
                return fact
        return None

    def approve(self, proposal_id: str) -> Fact:
        with self._lock():
            proposal = self.get(proposal_id)
            if proposal.status == "approved":
                return self.store.get(_required(proposal.result_fact_id, "result_fact_id"))
            if proposal.status != "pending":
                raise MemoryConflictError("Only pending proposals can be approved")
            self.store.refresh()
            produced = self._produced(proposal)
            if produced is not None:
                proposal.status, proposal.result_fact_id = "approved", produced.id
                self._save(proposal)
                return produced
            current = self._classify(proposal.entity, proposal.attribute, proposal.value,
                                     proposal.valid_from)
            kind, target_id = proposal.kind, proposal.target_id
            if (current.get("kind"), current.get("target_id"), current.get("target_fingerprint")
                    ) != (proposal.kind, proposal.target_id, proposal.target_fingerprint):
                # Only an expected predecessor changed the preview: apply what it needs now.
                kind = self._check_still_valid(proposal, current)
                target_id = current.get("target_id")
            if kind == "duplicate":
                result = self.store.get(_required(target_id, "target_id"))
            elif kind == "correct":
                # The corrected record keeps its topic folder; the confidence is
                # that of the new statement.
                result = self.store.correct(_required(proposal.target_id, "target_id"),
                                            proposal.value, source=proposal.source,
                                            evidence=proposal.evidence,
                                            confidence=proposal.confidence)
            else:
                result = self.store.add(entity=proposal.entity, attribute=proposal.attribute,
                                        value=proposal.value, valid_from=proposal.valid_from,
                                        source=proposal.source, evidence=proposal.evidence,
                                        confidence=proposal.confidence, topic=proposal.topic)
            proposal.status, proposal.result_fact_id = "approved", result.id
            self._save(proposal)
            return result

    def _check_still_valid(self, proposal: Proposal, current: Dict[str, Any]) -> str:
        """Return how to apply the proposal now, or raise if it is stale.

        Approving a dated conversation in order appends assertions, so a record
        approved from the same source after this proposal was made, effective
        before it, is an expected predecessor: the proposal then adds its value
        ("update"), or confirms the value that record already holds
        ("duplicate"). Anything else (including an edit of the record the
        reviewer saw, even with the same ID) needs a new review.
        """
        stale = MemoryConflictError("Memory changed since this proposal; reject it and extract "
                                    "again")
        target, kind = current.get("target_id"), current.get("kind")
        try:
            if (proposal.kind in ("add", "update", "duplicate") and kind in ("update", "duplicate")
                    and target != proposal.target_id):
                predecessor = self.store.get(str(target))
                if (predecessor.source == proposal.source
                        and predecessor.recorded_at >= normalize_time(proposal.created_at)
                        and predecessor.valid_from < proposal.valid_from
                        and (proposal.target_id is None
                             or self._reviewed_unchanged(proposal, predecessor))):
                    return str(kind)
            if (proposal.kind == "add" and current.get("kind") in ("correct", "duplicate")
                    and target is not None
                    and self.store.get(str(target)).source == proposal.source):
                raise MemoryConflictError(
                    "Another proposal from the same source already set {} for this time; each "
                    "attribute holds one value at a time, so reject this one, or extract it "
                    "again to review it as a correction".format(proposal.attribute))
        except KeyError:
            raise stale from None  # A record the reviewer saw is gone.
        raise stale

    def _reviewed_unchanged(self, proposal: Proposal, predecessor: Fact) -> bool:
        """The record the reviewer saw is unedited, still in effect at its own
        time, and older than the newer predecessor."""
        reviewed = self.store.get(str(proposal.target_id))
        in_effect = [fact.id for fact in self.store.snapshot(as_of=reviewed.valid_from,
                                                             entity=proposal.entity)
                     if fact.attribute == proposal.attribute]
        return (predecessor.valid_from > reviewed.valid_from and in_effect == [reviewed.id]
                and _fingerprint(reviewed) == proposal.target_fingerprint)

    def reject(self, proposal_id: str) -> Proposal:
        with self._lock():
            proposal = self.get(proposal_id)
            if proposal.status == "approved":
                raise MemoryConflictError(
                    "Approved proposals cannot be rejected; retract the fact instead")
            proposal.status = "rejected"
            self._save(proposal)
            return proposal
