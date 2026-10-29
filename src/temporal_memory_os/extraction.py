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
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import unicodedata
from uuid import uuid4

from .models import Fact, _text, normalize_time, validate_topic
from .store import MemoryConflictError, MemoryStore, StoreIntegrityError, _flush_directory

_PROPOSAL_ID = re.compile(r"[a-f0-9]{32}")
_STATUSES = ("pending", "approved", "rejected")
_KINDS = ("add", "update", "correct", "duplicate")
# A possible sentence end: the word before it, a terminator, any closing quotes or
# brackets, then a space. A match starts only at the start of that word, so a long
# word is scanned once.
_SENTENCE_END = re.compile(
    r"(?<!\w)(\w*)([.!?])[\"'\u201d\u2019\u00bb\u203a)\]}\u300d\u300f\u3011\uff09\uff3d\uff5d]*\s")


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
            or not isinstance(proposal.dated, bool) or not isinstance(proposal.observed, bool)
            or not isinstance(proposal.warnings, list)
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


def _line(proposal: "Proposal") -> Tuple[str, ...]:
    """The line a proposal was extracted from, and where it was extracted to."""
    return (proposal.entity, proposal.attribute, proposal.value, proposal.source,
            proposal.evidence, proposal.topic)


def _statement(proposal: "Proposal") -> Tuple[Any, ...]:
    """What approving ``proposal`` asserts. A correction keeps its target's
    topic folder, so the target stands in for the topic."""
    return (proposal.entity, proposal.attribute, proposal.value, proposal.valid_from,
            proposal.source, proposal.evidence, proposal.confidence,
            proposal.target_id if proposal.kind == "correct" else proposal.topic)


def _content(line: str) -> str:
    # Files joined with cat keep each one's byte-order mark, at a line start.
    return line.strip().lstrip("\ufeff").strip()


def _more_than_one_sentence(statement: str) -> bool:
    """A sentence ends before the line does. A period after an initial ("J.") or a
    two-letter abbreviation ("Dr.", "St.") does not end a sentence; capital and small
    letters are those of any script, so an accented initial counts too, however
    its accent is encoded."""
    for match in _SENTENCE_END.finditer(unicodedata.normalize("NFC", statement)):
        word = match[1]
        if match[2] == "." and (len(word) == 1 and word.isupper()
                                or len(word) == 2 and word[0].isupper() and word[1].islower()):
            continue
        return True
    return False


def _reason(error: OSError) -> str:
    # str(error) would show the absolute path of the memory directory.
    return error.strerror or type(error).__name__


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
    # True when an undated line took an explicit observation time (observed_at)
    # rather than "now". Files written before this field load as False.
    observed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class ProposalQueue:
    """Persist proposals separately; only approve() changes the fact store."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.directory = Path(store.root) / "proposals"
        if self.directory.is_symlink():
            raise StoreIntegrityError("Proposal directory must not be a symbolic link")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.problems: List[str] = []  # Damaged files skipped by the last list().

    @contextmanager
    def _lock(self) -> Iterator[None]:
        # Same POSIX lock discipline as the store; approvals serialize across API workers.
        import fcntl
        lock_path = self.directory / ".lock"
        if lock_path.is_symlink():
            raise StoreIntegrityError("Proposal lock must not be a symbolic link")
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
            raise StoreIntegrityError("Proposal files must not be symbolic links: proposals/"
                                      + path.name)
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
        except OSError as exc:  # Unreadable, or a folder with a proposal's name.
            raise StoreIntegrityError("{} cannot be read ({}); fix or delete it".format(
                name, _reason(exc))) from exc
        known = {item.name for item in fields(Proposal)}
        if not isinstance(data, dict) or not known.issuperset(data):
            raise ValueError("{} is not a proposal file; fix or delete it".format(name))
        try:
            # JSON can escape a lone surrogate ("\ud800"), which no output can encode.
            json.dumps(data, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("{} contains text that is not valid Unicode; fix or delete "
                             "it".format(name)) from exc
        except RecursionError as exc:
            # Where the stack allows deeper loading than writing (Linux, Python 3.14),
            # a deeply nested field loads but cannot be written back.
            raise ValueError("{} is not a proposal file; fix or delete it".format(name)) from exc
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
            except ValueError as exc:  # These name the file relative to the memory directory.
                self.problems.append(str(exc))
            except OSError as exc:
                self.problems.append("proposals/{} cannot be read ({})".format(
                    path.name, _reason(exc)))
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
        # A sentence rule matches the whole line, so a second sentence on it would end
        # up in the value ("Paris. I moved here", "Paris.) I moved"): such a line is
        # skipped. A period after an initial or a two-letter abbreviation ("J. P. Morgan",
        # "Dr. Smith", "St. Louis") does not end a sentence; other values with a period
        # followed by a space ("Mrs. Jones") need the pipe form.
        if _more_than_one_sentence(statement):
            return None
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
        if isinstance(text, str) and len(text) > 1_000_000:  # Before splitting it.
            raise ValueError("Conversation exceeds the 1 MB extraction limit")
        # Empty by the same reading as the lines below, so a byte-order mark on each
        # of several lines is still empty text.
        if not isinstance(text, str) or not any(_content(line) for line in text.splitlines()):
            raise ValueError("Conversation text must not be empty")
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
        for line in text.splitlines():
            raw = _content(line)
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
        with self._lock():
            self.store.refresh()
            # Indexed, so a long text is not compared with the whole queue line by line.
            statements: Dict[Tuple[str, ...], List[Proposal]] = {}
            histories: Dict[Tuple[str, str], List[Proposal]] = {}

            def remember(proposal: Proposal) -> None:
                statements.setdefault(_line(proposal), []).append(proposal)
                histories.setdefault((proposal.entity, proposal.attribute), []).append(proposal)

            for proposal in self.list(status="all"):
                remember(proposal)
            # Where this text last states each line, so a re-extracted log keeps its order.
            last: Dict[Tuple[str, ...], int] = {
                (entity, attribute, value, source, raw, topic): index
                for index, (entity, attribute, value, _, raw, *_) in enumerate(candidates)}
            # A line followed in this text by another value for its attribute was said
            # before that value; only a line no other value follows can be a new
            # statement. The same line again later is not another value.
            followed: List[bool] = []
            values_after: Dict[Tuple[str, str], set[str]] = {}
            for entity, attribute, value, *_ in reversed(candidates):
                values = values_after.setdefault((entity, attribute), set())
                followed.append(len(values) > (value in values))  # Another value follows.
                values.add(value)
            followed.reverse()
            results: List[Proposal] = []
            returned: set[str] = set()
            new: List[Proposal] = []
            # Each attribute holds one value at a time: two lines giving the same
            # attribute for the same instant cannot both be approved. A repeated line
            # is the same statement, not a second one, and a line that returns a
            # pending proposal counts at that proposal's time.
            instants: Dict[Tuple[str, str, str], set[str]] = {}
            created = ""
            for index, (entity, attribute, value, effective, raw, confidence,
                        dated) in enumerate(candidates):
                same = statements.get((entity, attribute, value, source, raw, topic), [])
                duplicate = next((p for p in same if p.status != "rejected"
                                  and (p.valid_from == effective
                                       or self._same_undated(
                                           p, dated, observed_at, observed,
                                           histories[(entity, attribute)],
                                           lambda other: (followed[index]
                                                          or last.get(_line(other), -1) > index)))),
                                 None)
                if duplicate:
                    instants.setdefault((entity, attribute, duplicate.valid_from), set()).add(raw)
                    if duplicate.id not in returned:  # A repeated line is returned once.
                        returned.add(duplicate.id)
                        results.append(duplicate)
                    continue
                instants.setdefault((entity, attribute, effective), set()).add(raw)
                proposal = Proposal(id=uuid4().hex, entity=entity, attribute=attribute, value=value,
                                    valid_from=effective, source=source, evidence=raw, topic=topic,
                                    created_at=(created := _after(created)),
                                    confidence=confidence, dated=dated,
                                    observed=not dated and observed_at is not None,
                                    **self._classify(entity, attribute, value, effective))
                remember(proposal)
                new.append(proposal)
                returned.add(proposal.id)
                results.append(proposal)
            for proposal in new:
                if len(instants[(proposal.entity, proposal.attribute, proposal.valid_from)]) > 1:
                    proposal.warnings.append(
                        "Another line in this text gives {} for the same time; only one can be "
                        "approved.".format(proposal.attribute))
                self._save(proposal)
            return results

    def _same_undated(self, pending: Proposal, dated: bool, observed_at: Optional[str],
                      now: str, queued: List[Proposal],
                      stated_later: Callable[[Proposal], bool]) -> bool:
        """An undated line extracted again "now" is the statement already pending,
        unless memory has moved past it since (then it is new information): a
        value took effect after it, or a different value was proposed after it.
        A value that takes effect only after ``now`` has not moved past it yet.

        ``queued`` holds the proposals for the pending one's entity and attribute.
        ``stated_later`` tells whether the text being extracted states a
        proposal's line again after this line: that value was said after this
        line, so re-extracting an unchanged log is not new information.
        """
        # The line's own flag decides: the evidence is the same line, and proposal
        # files written before the flag existed load as dated.
        if dated or pending.status != "pending" or observed_at is not None:
            return False
        later = [fact for fact in self.store.history(pending.entity, pending.attribute)
                 if pending.valid_from < fact.valid_from <= now]
        # Not yet approved, but approving the queue in order would otherwise end on
        # that different value although this statement was made after it.
        said = [other for other in queued
                if other.status != "rejected" and other.value != pending.value
                and pending.valid_from < other.valid_from <= now and not stated_later(other)]
        return not later and not said

    def _produced(self, proposal: Proposal) -> Optional[Fact]:
        """The fact an interrupted approval already wrote, if any.

        Fact writes and proposal status are separate files; a crash between
        them leaves a pending proposal whose fact exists. Approving again
        recognizes that fact instead of reporting a conflict. A duplicate can
        have written one too: a dated restatement, or a return to an earlier
        value approved in order.
        """
        created = normalize_time(proposal.created_at)
        # A new record takes the spelling of an existing topic folder (see approve).
        topics = {proposal.topic, self.store.existing_topic(proposal.topic)}
        matching = [fact for fact in self.store.audit_trail(proposal.entity, proposal.attribute)
                    if (fact.value, fact.valid_from, fact.source, fact.evidence, fact.confidence)
                    == (proposal.value, proposal.valid_from, proposal.source, proposal.evidence,
                        proposal.confidence)
                    and not fact.deleted and fact.recorded_at >= created
                    and (fact.supersedes == proposal.target_id if proposal.kind == "correct"
                         else fact.supersedes is None and fact.topic in topics)]
        if not matching:
            return None  # The usual case: no need to read every approved proposal.
        # Only an approved proposal stating the same thing can have written a matching
        # fact. One approved as a duplicate of it merely points at it.
        def stated(other: Proposal) -> Tuple[Any, ...]:
            # Topics compare by the folder they name, as the matching facts do.
            *rest, where = _statement(other)
            return (*rest, where if other.kind == "correct" else self.store.existing_topic(where))

        own = stated(proposal)
        claimed = {other.result_fact_id for other in self.list(status="approved")
                   if stated(other) == own}
        return next((fact for fact in matching if fact.id not in claimed), None)

    def approve(self, proposal_id: str) -> Fact:
        with self._lock():
            proposal = self.get(proposal_id)
            if proposal.status == "approved":
                return self.store.get(_required(proposal.result_fact_id, "result_fact_id"))
            if proposal.status != "pending":
                raise MemoryConflictError("Only pending proposals can be approved")
            # Persist outside edits even if the approval later proves stale.
            self.store.refresh()
            # Hold the store's writer transaction from preview validation through
            # the write: a direct add/correction does not take the queue lock.
            with self.store._transaction():
                self.store._refresh_locked()
                result = self._approve_pending(proposal)
            # The fact must commit first. An interrupted status save is recoverable
            # via _produced; an approved status pointing at a rolled-back fact is not.
            proposal.status, proposal.result_fact_id = "approved", result.id
            self._save(proposal)
            return result

    def _approve_pending(self, proposal: Proposal) -> Fact:
        """Validate and write under both the queue lock and store transaction."""
        produced = self._produced(proposal)
        if produced is not None:
            return produced
        current = self._classify(proposal.entity, proposal.attribute, proposal.value,
                                 proposal.valid_from)
        kind, target_id = proposal.kind, proposal.target_id
        if (current.get("kind"), current.get("target_id"), current.get("target_fingerprint")
                ) != (proposal.kind, proposal.target_id, proposal.target_fingerprint):
            # Only an expected predecessor changed the preview: apply what it needs now.
            kind = self._check_still_valid(proposal, current)
            target_id = current.get("target_id")
        confirmed = (self.store.get(_required(target_id, "target_id"))
                     if kind == "duplicate" else None)
        if confirmed is not None and ((not proposal.dated and not proposal.observed)
                                      or confirmed.valid_from == proposal.valid_from):
            # Nothing new: an undated line extracted "now" restates what holds now,
            # and a line for the confirmed record's own date restates that record.
            result = confirmed
        elif kind == "correct":
            # The corrected record keeps its topic folder; the confidence is
            # that of the new statement.
            result = self.store.correct(_required(proposal.target_id, "target_id"),
                                        proposal.value, source=proposal.source,
                                        evidence=proposal.evidence,
                                        confidence=proposal.confidence)
        else:
            # A line restating the value in effect at its own date (or observation
            # time) is recorded too, so a change approved later for an earlier date
            # cannot override it. It keeps the confirmed record's own end (as stored,
            # not as bounded by the next value), so it never extends the value past
            # an explicit end. On a case-insensitive disk, a topic spelled unlike
            # its existing folder is filed there under the folder's spelling.
            result = self.store.add(entity=proposal.entity, attribute=proposal.attribute,
                                    value=proposal.value, valid_from=proposal.valid_from,
                                    valid_to=None if confirmed is None else confirmed.valid_to,
                                    source=proposal.source, evidence=proposal.evidence,
                                    confidence=proposal.confidence,
                                    topic=self.store.existing_topic(proposal.topic))
        return result

    def _check_still_valid(self, proposal: Proposal, current: Dict[str, Any]) -> str:
        """Return how to apply the proposal now, or raise if it is stale.

        Approving a dated conversation in order appends assertions, so a record
        approved from the same source after this proposal was made, effective
        before it, is an expected predecessor: the proposal then adds its value
        ("update"), or confirms the value that record already holds
        ("duplicate", recorded at its own date). A newer record from any source
        that restates the prior value the reviewer saw, such as an approved dated
        restatement, changes nothing the preview showed either. Anything else
        (including an edit of the record the reviewer saw, even with the same ID)
        needs a new review.
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
                if (kind == proposal.kind and proposal.target_id is not None
                        and predecessor.value == proposal.prior_value
                        and predecessor.valid_from <= proposal.valid_from
                        and self._reviewed_unchanged(proposal, predecessor)):
                    return str(kind)
            if (proposal.kind == "add" and current.get("kind") in ("correct", "duplicate")
                    and target is not None
                    and self.store.get(str(target)).source == proposal.source):
                raise MemoryConflictError(
                    "Another proposal from the same source already set {} for this time; each "
                    "attribute holds one value at a time, so reject this one; to replace that "
                    "value with it instead, reject this one first, then extract it again to "
                    "review the change".format(proposal.attribute))
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

    def _orphaned_by_reject(self, proposal: Proposal) -> bool:
        """Rejecting ``proposal`` would leave a fact in memory with no approval behind it.

        That is the case when an interrupted approval wrote the fact it states, that
        fact has not been retracted or corrected since (a revision carries its own
        provenance), and no other pending proposal can claim it. On a
        case-insensitive disk, the same line under another letter case of its topic
        names the same folder, so that proposal claims the fact just as well.
        """
        produced = self._produced(proposal)
        if produced is None:
            return False
        if any(fact.supersedes == produced.id
               for fact in self.store.audit_trail(proposal.entity, proposal.attribute)):
            return False
        # Topics aside, a proposal that can claim the fact states the same thing.
        stated = _statement(proposal)[:-1]
        for other in self.list(status="pending"):
            if other.id != proposal.id and _statement(other)[:-1] == stated:
                claim = self._produced(other)
                if claim is not None and claim.id == produced.id:
                    return False
        return True

    def reject(self, proposal_id: str) -> Proposal:
        with self._lock():
            proposal = self.get(proposal_id)
            if proposal.status == "approved":
                raise MemoryConflictError(
                    "Approved proposals cannot be rejected; retract the fact instead")
            if proposal.status == "pending":
                self.store.refresh()
                if self._orphaned_by_reject(proposal):
                    raise MemoryConflictError(
                        "An interrupted approval already wrote this proposal's fact; approve it "
                        "again to finish, then retract the fact if it is wrong")
            proposal.status = "rejected"
            self._save(proposal)
            return proposal
