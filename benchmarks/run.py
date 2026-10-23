#!/usr/bin/env python3
"""Reproducible synthetic temporal-retrieval diagnostic; no API calls or downloads."""

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Also supports running from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temporal_memory_os.retrieval import HybridRetriever  # noqa: E402
from temporal_memory_os.store import MemoryStore  # noqa: E402
from temporal_memory_os.embeddings import HashingEmbedder  # noqa: E402


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def lexical_lookup(query, corpus):
    """Simple, disclosed token-overlap baseline, with newest-record tie breaking."""
    query_terms = set(tokens(query))

    def rank(fact):
        terms = Counter(tokens(" ".join((fact.entity, fact.attribute, fact.value))))
        overlap = sum(min(terms[word], 1) for word in query_terms)
        return overlap, fact.recorded_at, fact.id

    ranked = sorted(corpus, key=rank, reverse=True)
    return ranked[0] if ranked and rank(ranked[0])[0] > 0 else None


def build_workload(store, entities):
    questions = []
    states = [
        ("2025-01-01", "2025-01-15", "pending"),
        ("2025-02-01", "2025-02-15", "interview invited"),
        ("2025-03-01", "2025-03-15", "interview completed"),
    ]
    for index in range(entities):
        entity = "candidate{:04d}".format(index)
        for effective, as_of, value in states:
            fact = store.add(
                entity=entity,
                attribute="application_status",
                value=value,
                valid_from=effective,
                recorded_at=effective,
                source="synthetic://benchmark/{}/{}".format(entity, effective),
                evidence="Synthetic status update for {}: {}.".format(entity, value),
                tags=["synthetic", "application"],
                topic="benchmark/applications",
            )
            questions.append({
                "query": "{} application_status".format(entity),
                "entity": entity,
                "attribute": "application_status",
                "as_of": as_of,
                "known_at": "2025-04-01",
                "expected_id": fact.id,
                "expected_value": value,
            })
    return questions


def build_corrections(store, entities):
    """A status is recorded, then corrected weeks later at the same effective date.

    Each entity yields one question asked with the knowledge available before
    the correction and one asked after it. Both concern the same effective day.
    """
    questions = []
    for index in range(entities):
        entity = "applicant{:04d}".format(index)
        original = store.add(
            entity=entity, attribute="application_status", value="pending",
            valid_from="2025-01-01", recorded_at="2025-01-01",
            source="synthetic://benchmark/{}/initial".format(entity),
            evidence="Synthetic initial status for {}.".format(entity),
            tags=["synthetic", "application"], topic="benchmark/corrections",
        )
        corrected = store.correct(
            original.id, "withdrawn", recorded_at="2025-02-10",
            source="synthetic://benchmark/{}/correction".format(entity),
            evidence="Synthetic late correction: {} had withdrawn on 1 January.".format(entity),
        )
        for known_at, fact in (("2025-01-20", original), ("2025-03-01", corrected)):
            questions.append({
                "query": "{} application_status".format(entity),
                "entity": entity,
                "attribute": "application_status",
                "as_of": "2025-01-15",
                "known_at": known_at,
                "expected_id": fact.id,
                "expected_value": fact.value,
            })
    return questions


def evaluate(questions, lookup):
    correct, stale, covered = 0, 0, 0
    latencies = []
    errors = []
    for question in questions:
        start = time.perf_counter()
        answer = lookup(question)
        latencies.append((time.perf_counter() - start) * 1000)
        if answer is not None:
            covered += 1
        is_correct = answer is not None and answer.id == question["expected_id"]
        correct += int(is_correct)
        is_stale = (
            answer is not None
            and answer.entity == question["entity"]
            and answer.attribute == question["attribute"]
            and answer.id != question["expected_id"]
        )
        stale += int(is_stale)
        if not is_correct and len(errors) < 3:
            errors.append({
                "query": question["query"], "as_of": question["as_of"],
                "expected": question["expected_value"],
                "retrieved": None if answer is None else answer.value,
            })
    count = len(questions)
    sorted_latency = sorted(latencies)
    return {
        "queries": count,
        "correct": correct,
        "top1_fact_accuracy": correct / count,
        "stale_hit_rate": stale / count,
        "retrieval_coverage": covered / count,
        "latency_ms_median": round(statistics.median(latencies), 3),
        "latency_ms_p95": round(sorted_latency[min(count - 1, int(count * 0.95))], 3),
        "example_errors": errors,
    }


def _methods(store, all_facts, latest):
    # Keep the diagnostic offline even if the caller configured a model.
    retriever = HybridRetriever(store, embedder=HashingEmbedder())

    def temporal(question, use_known_at=True):
        results = retriever.search(
            question["query"], as_of=question["as_of"],
            known_at=question["known_at"] if use_known_at else None, limit=1, mode="hybrid",
        )
        return results[0].fact if results else None

    return {
        "all_records_keyword": lambda q: lexical_lookup(q["query"], all_facts),
        "latest_state_keyword": lambda q: lexical_lookup(q["query"], latest),
        "valid_time_only": lambda q: temporal(q, use_known_at=False),
        "temporal_hybrid": temporal,
    }


WORKLOADS = {
    "state_changes": {
        "build": build_workload,
        "latest": ("2025-03-15", "2025-04-01"),
        "description": "Three dated status changes per entity; one query per historical state.",
    },
    "late_corrections": {
        "build": build_corrections,
        "latest": ("2025-01-15", "2025-03-01"),
        "description": "One status per entity, corrected later at the same effective date; "
                       "the same day is queried with knowledge from before and after the correction.",
    },
}


def run(entities=50):
    results = {}
    corpus = {}
    for name, workload in WORKLOADS.items():
        with tempfile.TemporaryDirectory(prefix="memoryos-benchmark-") as directory:
            with MemoryStore(directory) as store:
                questions = workload["build"](store, entities)
                all_facts = store.all_facts()
                as_of, known_at = workload["latest"]
                latest = store.snapshot(as_of=as_of, known_at=known_at)
                methods = _methods(store, all_facts, latest)
                results[name] = {method: evaluate(questions, lookup)
                                 for method, lookup in methods.items()}
                corpus[name] = {"entities": entities, "records": len(all_facts),
                                "queries": len(questions),
                                "description": workload["description"]}
    return {
        "benchmark": "synthetic_temporal_smoke_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "semantic_backend": "Default offline lexical hashing; no learned semantic model.",
        "methods": {
            "all_records_keyword": "Token overlap over every recorded version; ties prefer the newest record.",
            "latest_state_keyword": "Token overlap over the latest resolved state only.",
            "valid_time_only": "The package's hybrid search with as_of but no knowledge cutoff (ablation).",
            "temporal_hybrid": "The package's hybrid search with as_of and known_at.",
        },
        "metrics": {
            "top1_fact_accuracy": "Fraction retrieving the expected version's exact fact ID.",
            "stale_hit_rate": "Fraction returning the correct entity/attribute but the wrong version; includes future-state and future-knowledge leakage.",
            "retrieval_coverage": "Fraction with at least one returned result.",
            "latency": "Warm-process retrieval only; excludes corpus construction and indexing.",
        },
        "limitations": [
            "Synthetic exact-entity queries; not a language-model answer-quality benchmark.",
            "Baselines use token overlap, not trained vector RAG or a tuned production search system.",
            "All-record baseline breaks equal lexical scores by latest recorded time.",
            "Dates are provided as structured filters only to the temporal methods.",
            "These workloads demonstrate temporal filtering by construction; they do not establish general superiority.",
            "No inference, token-cost, hallucination, statistical-significance, or large-scale claims are made.",
        ],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=50)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    args = parser.parse_args()
    if not 1 <= args.entities <= 10000:
        parser.error("--entities must be between 1 and 10000")
    report = run(args.entities)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
