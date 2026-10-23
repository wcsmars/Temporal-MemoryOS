"""Conservative offline extraction into an explicit human review queue.

This is a rule-based extractor, not an LLM. Unsupported sentences are ignored.
For arbitrary assertions, use: entity | attribute | value | optional ISO date.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

from .models import Fact, normalize_time
from .store import MemoryStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _after(previous: str) -> str:
    """The current time, but strictly later than ``previous``.

    Proposals are listed by creation time, so a batch extracted within one
    clock tick must still list in the order its lines appeared.
    """
    now = _now()
    if previous and now <= previous:
        return (datetime.fromisoformat(previous) + timedelta(microseconds=1)).isoformat(
            timespec="microseconds")
    return now


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
        if not re.fullmatch(r"[a-f0-9]{32}", proposal_id):
            raise ValueError("Invalid proposal ID")
        path = self.directory / (proposal_id + ".json")
        if path.is_symlink():
            raise ValueError("Proposal files must not be symbolic links")
        return path

    def _save(self, proposal: Proposal) -> None:
        destination = self._path(proposal.id)
        fd, temporary = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(proposal.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get(self, proposal_id: str) -> Proposal:
        path = self._path(proposal_id)
        if not path.exists():
            raise KeyError(f"Proposal not found: {proposal_id}")
        return Proposal(**json.loads(path.read_text(encoding="utf-8")))

    # ``list`` is a method name here, so annotations in this class use ``List``.
    def list(self, status: Optional[str] = "pending") -> List[Proposal]:
        if status not in (None, "all", "pending", "approved", "rejected"):
            raise ValueError("status must be pending, approved, rejected, or all")
        proposals = [self.get(p.stem) for p in sorted(self.directory.glob("*.json"))]
        return sorted([p for p in proposals if status in (None, "all") or p.status == status],
                      key=lambda p: (p.created_at, p.id))

    def _classify(self, entity: str, attribute: str, value: str, valid_from: str) -> Dict[str, Any]:
        facts = [f for f in self.store.snapshot(as_of=valid_from, entity=entity)
                 if f.attribute == attribute]
        if not facts:
            # A tombstone still owns its effective timestamp. Restoring it is
            # an auditable correction, not a conflicting second assertion.
            facts = [f for f in self.store.history(entity, attribute)
                     if f.deleted and f.valid_from == valid_from]
            if not facts:
                return {"kind": "add"}
        prior = facts[0]
        fingerprint = hashlib.sha256(json.dumps(self.store.get(prior.id).to_dict(),
                                               sort_keys=True).encode("utf-8")).hexdigest()
        if prior.value == value:
            return {"kind": "duplicate", "target_id": prior.id, "prior_value": prior.value,
                    "target_fingerprint": fingerprint,
                    "warnings": ["The same value is already effective on this date."]}
        correction = prior.valid_from == valid_from
        return {"kind": "correct" if correction else "update", "target_id": prior.id,
                "target_fingerprint": fingerprint,
                "prior_value": prior.value,
                "warnings": ["Changes an existing value; review the evidence and effective date."]}

    def extract(self, text: str, source: str, observed_at: Optional[str] = None,
                topic: str = "general") -> List[Proposal]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Conversation text must not be empty")
        if len(text) > 1_000_000:
            raise ValueError("Conversation exceeds the 1 MB extraction limit")
        if not source.strip():
            raise ValueError("A provenance source is required")
        observed = normalize_time(observed_at or _now())
        candidates = []
        for line in text.splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            effective = observed
            date_match = re.match(r"^(\d{4}-\d{2}-\d{2}):\s*(.*)$", raw)
            statement = raw
            if date_match:
                effective, statement = normalize_time(date_match[1]), date_match[2]
            if "|" in statement:
                parts = [part.strip() for part in statement.split("|")]
                if len(parts) not in (3, 4) or not all(parts[:3]):
                    raise ValueError("Structured facts need entity | attribute | value | optional date")
                entity, attribute, value = parts[:3]
                if len(parts) == 4:
                    effective = normalize_time(parts[3])
                confidence = 1.0
            elif match := re.fullmatch(r"(?:Application (.+?)|My application (?:at|to) (.+?)) is (.+?)[.!]?",
                                      statement, re.I):
                entity = (match[1] or match[2]).strip() + " application"
                attribute, value, confidence = "status", match[3].strip(), 0.85
            elif match := re.fullmatch(r"My ([\w -]{1,60}) is (.+?)[.!]?", statement, re.I):
                entity, attribute, value = "user", match[1].strip().lower(), match[2].strip()
                confidence = 0.85
            elif match := re.fullmatch(r"I (?:prefer|like) (.+?)[.!]?", statement, re.I):
                entity, attribute, value = "user", "preference", match[1].strip()
                confidence = 0.75
            else:
                continue
            # Validate the candidate using the exact same rules as stored facts.
            Fact.from_dict(dict(id=uuid4().hex, entity=entity, attribute=attribute, value=value,
                                valid_from=effective, valid_to=None, recorded_at=_now(),
                                source=source, evidence=raw, confidence=confidence,
                                tags=[], links=[], topic=topic, supersedes=None, deleted=False))
            candidates.append((entity, attribute, value, effective, raw, confidence))
        with self._lock():
            self.store.refresh()
            existing = self.list(status="all")
            results = []
            created = ""
            for entity, attribute, value, effective, raw, confidence in candidates:
                signature = (entity, attribute, value, effective, source, raw)
                duplicate = next((p for p in existing if
                                  (p.entity, p.attribute, p.value, p.valid_from, p.source, p.evidence)
                                  == signature and p.status != "rejected"), None)
                if duplicate:
                    results.append(duplicate)
                    continue
                proposal = Proposal(id=uuid4().hex, entity=entity, attribute=attribute, value=value,
                                    valid_from=effective, source=source, evidence=raw, topic=topic,
                                    created_at=(created := _after(created)),
                                    confidence=confidence,
                                    **self._classify(entity, attribute, value, effective))
                self._save(proposal)
                existing.append(proposal)
                results.append(proposal)
            return results

    def approve(self, proposal_id: str) -> Fact:
        with self._lock():
            proposal = self.get(proposal_id)
            if proposal.status == "approved":
                return self.store.get(_required(proposal.result_fact_id, "result_fact_id"))
            if proposal.status != "pending":
                raise ValueError("Only pending proposals can be approved")
            self.store.refresh()
            current = self._classify(proposal.entity, proposal.attribute, proposal.value,
                                     proposal.valid_from)
            # Earlier facts approved from the same batch can supply a predecessor
            # for an add. The approved assertion still starts at its original date.
            # A changed correction target or same-instant collision needs a new review.
            append_with_predecessor = proposal.kind == "add" and current["kind"] == "update"
            if not append_with_predecessor and (
                current.get("kind"), current.get("target_id"), current.get("target_fingerprint")
            ) != (proposal.kind, proposal.target_id, proposal.target_fingerprint):
                raise ValueError("Memory changed since this proposal; reject it and extract again")
            if proposal.kind == "duplicate":
                result = self.store.get(_required(proposal.target_id, "target_id"))
            elif proposal.kind == "correct":
                result = self.store.correct(_required(proposal.target_id, "target_id"),
                                            proposal.value, source=proposal.source,
                                            evidence=proposal.evidence)
            else:
                result = self.store.add(entity=proposal.entity, attribute=proposal.attribute,
                                        value=proposal.value, valid_from=proposal.valid_from,
                                        source=proposal.source, evidence=proposal.evidence,
                                        confidence=proposal.confidence, topic=proposal.topic)
            proposal.status, proposal.result_fact_id = "approved", result.id
            self._save(proposal)
            return result

    def reject(self, proposal_id: str) -> Proposal:
        with self._lock():
            proposal = self.get(proposal_id)
            if proposal.status == "approved":
                raise ValueError("Approved proposals cannot be rejected; retract the fact instead")
            proposal.status = "rejected"
            self._save(proposal)
            return proposal
