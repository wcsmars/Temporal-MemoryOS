"""Synthetic, reproducible examples. No real personal data is included."""

from __future__ import annotations

from typing import Any, Dict, List

from .store import MemoryConflictError, MemoryStore


DEMO_SOURCE = "synthetic:temporal-memoryos-demo-v1"


def seed_demo(store: MemoryStore) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = [
        dict(entity="Acme application", attribute="status", value="pending",
             valid_from="2026-09-01", recorded_at="2026-09-01T09:00:00Z",
             topic="career/applications", links=["Acme"], tags=["career", "application"]),
        dict(entity="Acme application", attribute="status", value="interview invited",
             valid_from="2026-09-15", recorded_at="2026-09-15T09:00:00Z",
             topic="career/applications", links=["Acme"], tags=["career", "application"]),
        dict(entity="Acme application", attribute="status", value="interview completed",
             valid_from="2026-10-01", recorded_at="2026-10-01T09:00:00Z",
             topic="career/applications", links=["Acme"], tags=["career", "application"]),
        dict(entity="Acme", attribute="industry", value="robotics research",
             valid_from="2026-08-01", recorded_at="2026-08-01T09:00:00Z",
             topic="organizations", links=["Acme application"], tags=["company"]),
        dict(entity="Alex", attribute="preferred drink", value="tea",
             valid_from="2026-08-01", recorded_at="2026-08-01T09:00:00Z",
             topic="profile/preferences", tags=["preference"]),
        dict(entity="Alex", attribute="preferred drink", value="decaf coffee",
             valid_from="2026-10-05", recorded_at="2026-10-05T09:00:00Z",
             topic="profile/preferences", tags=["preference"]),
        dict(entity="Alex", attribute="home city", value="Riverton",
             valid_from="2026-08-01", recorded_at="2026-08-01T09:00:00Z",
             topic="profile/location", tags=["location"]),
    ]
    # Check the immutable audit records, not the latest state, for repeat runs.
    existing = {
        (f.entity, f.attribute, f.value, f.valid_from[:10])
        for f in store.all_facts() if f.source == DEMO_SOURCE
    }
    added, skipped = [], []
    for record in records:
        key = (record["entity"], record["attribute"], record["value"], record["valid_from"])
        if key in existing:
            continue
        try:
            fact = store.add(**record, source=DEMO_SOURCE,
                             evidence="Fictional example created for the project demonstration.")
        except MemoryConflictError:
            # Someone already recorded this entity/attribute at this time; the
            # demo never overwrites or corrects existing memory.
            skipped.append("{} / {} at {}".format(record["entity"], record["attribute"],
                                                  record["valid_from"]))
            continue
        added.append(fact.id)
    result: Dict[str, Any] = {"added": len(added), "fact_ids": added, "synthetic": True,
                              "try": 'search "Acme application status" --as-of 2026-09-10'}
    if skipped:
        result["skipped_existing"] = skipped
    return result
