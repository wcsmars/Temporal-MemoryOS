"""Validated, serializable records for the temporal memory engine."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, Union

TimeInput = Optional[Union[str, date, datetime]]
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


# The accepted grammar is spelled out instead of delegating to
# datetime.fromisoformat, whose accepted forms differ between Python versions.
# [0-9] rather than \d: other scripts' digits are not ISO 8601.
_TIME = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"(?:[Tt ](?P<hour>[0-9]{2}):(?P<minute>[0-9]{2})"
    r"(?::(?P<second>[0-9]{2})(?:\.(?P<fraction>[0-9]{1,6}))?)?"
    r"(?P<zone>[Zz]|[+-][0-9]{2}:[0-9]{2})?)?"
)


def _parse_time(text: str) -> datetime:
    match = _TIME.fullmatch(text)
    if not match:
        raise ValueError(
            "Time must be YYYY-MM-DD or YYYY-MM-DDTHH:MM[:SS[.ffffff]] with an optional "
            "Z or +HH:MM offset; got {!r}".format(text))
    parts = match.groupdict()
    zone = parts["zone"]
    if zone is None or zone in ("Z", "z"):
        tzinfo = timezone.utc
    else:
        sign = -1 if zone[0] == "-" else 1
        hours, minutes = int(zone[1:3]), int(zone[4:6])
        if hours > 23 or minutes > 59:
            raise ValueError("Invalid UTC offset in {!r}".format(text))
        tzinfo = timezone(sign * timedelta(hours=hours, minutes=minutes))
    try:
        return datetime(
            int(parts["year"]), int(parts["month"]), int(parts["day"]),
            int(parts["hour"] or 0), int(parts["minute"] or 0), int(parts["second"] or 0),
            int((parts["fraction"] or "0").ljust(6, "0")), tzinfo=tzinfo)
    except ValueError as exc:
        raise ValueError("Invalid date or time {!r}: {}".format(text, exc)) from exc


def normalize_time(value: TimeInput = None) -> str:
    """Return a sortable UTC ISO timestamp; dates and naive times mean UTC.

    A date represents the *start* of that date. All effective intervals are
    half open: ``valid_from <= instant < valid_to``. Strings must match one
    grammar on every Python version: ``YYYY-MM-DD``, optionally followed by
    ``THH:MM``, ``:SS``, ``.ffffff`` (1 to 6 digits) and ``Z`` or ``+HH:MM``.
    """
    if value is None:
        instant = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        instant = value
    elif isinstance(value, date):
        instant = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        instant = _parse_time(value.strip())
    else:
        raise ValueError("Time must be an ISO string, date, or datetime")
    if instant.utcoffset() is None:  # Naive, or a time zone that gives no offset.
        instant = instant.replace(tzinfo=timezone.utc)
    try:
        utc = instant.astimezone(timezone.utc)
    except OverflowError as exc:
        raise ValueError("Time is outside the supported range: {!r}".format(value)) from exc
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_topic(topic: str) -> str:
    """Validate a portable, nested path without allowing directory traversal."""
    if not isinstance(topic, str) or not topic or len(topic) > 512:
        raise ValueError("topic must be a nonempty relative folder path")
    parts = topic.split("/")
    if any(not _SAFE_SEGMENT.fullmatch(part) for part in parts):
        raise ValueError("topic must contain safe folder names separated by '/' (no '..')")
    return topic


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError("{} must be {}string".format(name, "a " if empty else "a nonempty "))
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:  # A lone surrogate, e.g. a hand-written "\ud800".
        raise ValueError("{} contains characters that are not valid Unicode text".format(
            name)) from exc
    return value


def _strings(value: Any, name: str) -> Tuple[str, ...]:
    """Validate and deduplicate, keeping order. Tuples keep shared records immutable."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("{} must be a list of strings".format(name))
    result: Dict[str, None] = {}
    for item in value:
        result[_text(item, name + " item")] = None
    return tuple(result)


@dataclass(frozen=True)
class Fact:
    """An immutable assertion or revision, as written to a Markdown record.

    ``valid_from`` describes the world; ``recorded_at`` describes when the
    engine learned the assertion. A correction supersedes an earlier record
    without destroying it. A tombstone retracts one effective interval.
    """

    id: str
    entity: str
    attribute: str
    value: str
    valid_from: str
    valid_to: Optional[str] = None
    recorded_at: str = field(default_factory=normalize_time)
    source: str = "manual"
    evidence: str = ""
    confidence: float = 1.0
    tags: Tuple[str, ...] = ()
    links: Tuple[str, ...] = ()
    topic: str = "general"
    supersedes: Optional[str] = None
    deleted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _SAFE_SEGMENT.fullmatch(self.id):
            raise ValueError("id must be a safe, nonempty record identifier")
        if not isinstance(self.deleted, bool):
            raise ValueError("deleted must be boolean")
        for name in ("entity", "attribute", "source"):
            _text(getattr(self, name), name)
        _text(self.value, "value", empty=self.deleted)
        _text(self.evidence, "evidence", empty=True)
        if self.valid_from is None or self.recorded_at is None:
            raise ValueError("valid_from and recorded_at are required")
        object.__setattr__(self, "valid_from", normalize_time(self.valid_from))
        object.__setattr__(self, "recorded_at", normalize_time(self.recorded_at))
        if self.valid_to is not None:
            object.__setattr__(self, "valid_to", normalize_time(self.valid_to))
            if self.valid_to <= self.valid_from:
                raise ValueError("valid_to must be later than valid_from")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (float, int)):
            raise ValueError("confidence must be a number between 0 and 1")
        if not 0 <= self.confidence <= 1 or not math.isfinite(self.confidence):
            raise ValueError("confidence must be a number between 0 and 1")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "tags", _strings(self.tags, "tags"))
        object.__setattr__(self, "links", _strings(self.links, "links"))
        validate_topic(self.topic)
        if self.supersedes is not None:
            if not isinstance(self.supersedes, str) or not _SAFE_SEGMENT.fullmatch(self.supersedes):
                raise ValueError("supersedes must be a valid record identifier")
            if self.supersedes == self.id:
                raise ValueError("A record cannot supersede itself")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tags"], data["links"] = list(self.tags), list(self.links)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        if not isinstance(data, dict):
            raise ValueError("Fact metadata must be a JSON object")
        missing = {"id", "entity", "attribute", "value", "valid_from", "recorded_at"} - data.keys()
        if missing:
            raise ValueError("Missing required fact fields: " + ", ".join(sorted(missing)))
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError("Invalid fact fields: {}".format(exc)) from exc
