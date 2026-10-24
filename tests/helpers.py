"""Helpers shared by several test files."""

import json
from datetime import datetime

from temporal_memory_os.models import Fact

MARKER = "<!-- memoryos:fact:v1 -->"


def hand_written_record(fact_id="b" * 32, value="tea", ensure_ascii=True):
    """A record as a person might write it: the marker is the very first line."""
    data = Fact(id=fact_id, entity="person", attribute="drink", value=value,
                valid_from="2025-01-01", recorded_at="2025-01-01").to_dict()
    return (MARKER + "\n```json\n" + json.dumps(data, ensure_ascii=ensure_ascii, indent=2)
            + "\n```\n")


class Python310Datetime(datetime):
    """A datetime whose fromisoformat rejects a "Z" suffix, as Python 3.10's does."""

    @classmethod
    def fromisoformat(cls, text):
        if text[-1:] in ("Z", "z"):
            raise ValueError("Invalid isoformat string: {!r}".format(text))
        return super().fromisoformat(text)


def approved_city(queue, line="user | city | London | 2026-01-01", source="chat"):
    """Extract one structured line and approve it; return the stored fact."""
    [proposal] = queue.extract(line, source=source)
    return queue.approve(proposal.id)
