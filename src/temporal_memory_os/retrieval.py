"""Time-filtered keyword, embedding and one-hop relationship retrieval."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .embeddings import Embedder, create_embedder, normalized_tokens, tokenize
from .models import Fact, TimeInput

Ranking = List[Tuple[str, float]]


class FactSource(Protocol):
    """What retrieval needs from a store: a resolved temporal snapshot."""

    def snapshot(self, as_of: TimeInput = None, known_at: TimeInput = None,
                 entity: Optional[str] = None) -> List[Fact]:
        ...


@dataclass
class SearchResult:
    fact: Fact
    score: float
    channels: Dict[str, float]
    citation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact.to_dict(),
            "score": self.score,
            "channels": dict(self.channels),
            "citation": self.citation,
        }


def _validated(vector: Sequence[float]) -> Tuple[List[float], float]:
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding vectors must contain only finite numbers")
    return values, math.sqrt(sum(value * value for value in values))


class HybridRetriever:
    """Rank only facts valid at ``as_of`` and known by ``known_at``.

    Keyword retrieval uses SQLite FTS5 BM25 over the selected temporal snapshot,
    so old versions cannot affect corpus statistics or displace eligible facts.
    ``semantic`` selects the embedding channel; with the default offline
    provider that channel is transparently labeled ``lexical_embedding``.
    Graph retrieval expands entity links in either direction by exactly one hop.
    Hybrid scores use reciprocal rank fusion (k=60), not incomparable raw scores.
    """

    available_modes = ("keyword", "semantic", "graph", "hybrid")
    cache_limit = 100_000

    def __init__(self, store: FactSource, embedder: Optional[Embedder] = None,
                 semantic_threshold: float = 0.25):
        if not math.isfinite(semantic_threshold) or not -1.0 <= semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be a finite number between -1 and 1")
        self.store = store
        self.embedder: Embedder = embedder if embedder is not None else create_embedder()
        self.semantic_threshold = semantic_threshold
        # Records are immutable and a snapshot only shortens valid_to, which
        # is not embedded, so a fact ID identifies its document vector.
        self._vectors: Dict[str, Tuple[int, Tuple[Tuple[int, float], ...], float, frozenset]] = {}

    @property
    def embedding_channel(self) -> str:
        return "semantic" if getattr(self.embedder, "semantic", True) else "lexical_embedding"

    def search(
        self, query: str, as_of: TimeInput = None, known_at: TimeInput = None, limit: int = 5,
        mode: str = "hybrid", entity: Optional[str] = None,
    ) -> List[SearchResult]:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if mode not in self.available_modes:
            raise ValueError("mode must be one of: " + ", ".join(self.available_modes))
        if entity is not None and (not isinstance(entity, str) or not entity.strip()):
            raise ValueError("entity must be a non-empty string")
        # Resolve and validate dates even for an empty query.
        facts = self.store.snapshot(as_of=as_of, known_at=known_at)
        if entity is not None:
            facts = [fact for fact in facts if fact.entity.casefold() == entity.casefold()]
        if not facts or not tokenize(query):
            return []

        channels: Dict[str, Ranking] = {}
        if mode in ("keyword", "hybrid"):
            channels["keyword"] = self._keyword(query, facts)
        if mode in ("semantic", "hybrid"):
            channels[self.embedding_channel] = self._embedding(query, facts)
        if mode in ("graph", "hybrid"):
            channels["graph"] = self._graph(query, facts)

        raw_scores: Dict[str, Dict[str, float]] = {}
        fused: Dict[str, float] = {}
        for channel, ranking in channels.items():
            for rank, (fact_id, raw) in enumerate(ranking, start=1):
                raw_scores.setdefault(fact_id, {})[channel] = round(raw, 8)
                weight = 0.7 if channel == "graph" else 1.0
                fused[fact_id] = fused.get(fact_id, 0.0) + weight / (60.0 + rank)
        by_id = {fact.id: fact for fact in facts}
        ordered = sorted(fused, key=lambda fact_id: (-fused[fact_id], fact_id))[:limit]
        return [SearchResult(
            fact=by_id[fact_id], score=round(fused[fact_id], 8),
            channels=raw_scores[fact_id], citation=self._citation(by_id[fact_id]),
        ) for fact_id in ordered]

    @staticmethod
    def _text(fact: Fact) -> str:
        return " ".join([
            fact.entity, fact.entity, fact.attribute, fact.value,
            fact.evidence, " ".join(fact.tags), " ".join(fact.links), fact.topic,
        ])

    @staticmethod
    def _keyword(query: str, facts: List[Fact]) -> Ranking:
        terms = list(dict.fromkeys(tokenize(query)))[:64]
        if not terms:
            return []
        # MATCH receives quoted word tokens only; user operators are never SQL
        # or FTS syntax. A snapshot-local table enforces time filtering first.
        expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in terms)
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE selected USING fts5("
                "id UNINDEXED, entity, attribute, value, context, tokenize='unicode61')"
            )
            connection.executemany(
                "INSERT INTO selected VALUES (?, ?, ?, ?, ?)",
                [(fact.id, fact.entity, fact.attribute, fact.value,
                  " ".join([fact.evidence, " ".join(fact.tags), " ".join(fact.links), fact.topic]))
                 for fact in facts],
            )
            return [(row[0], float(-row[1])) for row in connection.execute(
                "SELECT id, bm25(selected, 0, 4, 2, 2, 1) AS score "
                "FROM selected WHERE selected MATCH ? ORDER BY score ASC, id ASC",
                (expression,),
            ).fetchall()]
        finally:
            connection.close()

    def _embedding(self, query: str, facts: List[Fact]) -> Ranking:
        texts = [self._text(fact) for fact in facts]
        missing = [index for index, fact in enumerate(facts) if fact.id not in self._vectors]
        if len(self._vectors) + len(missing) > self.cache_limit:
            self._vectors.clear()
            missing = list(range(len(facts)))
        vectors = self.embedder.embed([query] + [texts[index] for index in missing])
        if len(vectors) != len(missing) + 1:
            raise ValueError("Embedding provider returned the wrong number of vectors")
        query_vector, query_norm = _validated(vectors[0])
        for index, vector in zip(missing, vectors[1:]):
            values, norm = _validated(vector)
            if len(values) != len(query_vector):
                raise ValueError("Embedding provider returned inconsistent vector dimensions")
            # Sparse storage: hashing vectors are mostly zeros, and dense
            # vectors cost no more than before.
            self._vectors[facts[index].id] = (
                len(values), tuple((i, v) for i, v in enumerate(values) if v), norm,
                frozenset(normalized_tokens(texts[index])))
        lexical = not getattr(self.embedder, "semantic", True)
        threshold = 0.0 if lexical else self.semantic_threshold
        query_tokens = set(normalized_tokens(query))
        ranking: Ranking = []
        for fact in facts:
            dimensions, pairs, norm, tokens = self._vectors[fact.id]
            # Hash collisions must never produce apparently relevant results.
            if lexical and query_tokens.isdisjoint(tokens):
                continue
            if dimensions != len(query_vector):
                raise ValueError("Embedding provider returned inconsistent vector dimensions")
            if not norm or not query_norm:
                continue
            score = sum(query_vector[i] * v for i, v in pairs) / (query_norm * norm)
            score = max(-1.0, min(1.0, score))
            if score > threshold:
                ranking.append((fact.id, score))
        return sorted(ranking, key=lambda item: (-item[1], item[0]))

    @staticmethod
    def _graph(query: str, facts: List[Fact]) -> Ranking:
        query_tokens = set(tokenize(query))
        entities = {fact.entity.casefold(): set(tokenize(fact.entity)) for fact in facts}
        seeds = {entity for entity, tokens in entities.items() if tokens and tokens <= query_tokens}
        if not seeds:
            return []
        adjacent = set()
        for fact in facts:
            entity = fact.entity.casefold()
            linked = {link.casefold() for link in fact.links}
            if entity in seeds:
                adjacent.update(linked)
            if linked & seeds:
                adjacent.add(entity)
        ranking: Ranking = []
        for fact in facts:
            entity = fact.entity.casefold()
            if entity in seeds:
                ranking.append((fact.id, 1.0))
            elif entity in adjacent:
                ranking.append((fact.id, 0.5))
        return sorted(ranking, key=lambda item: (-item[1], item[0]))

    def _citation(self, fact: Fact) -> str:
        if hasattr(self.store, "path_for") and hasattr(self.store, "root"):
            path = Path(self.store.path_for(fact.id))
            try:
                return path.relative_to(Path(self.store.root)).as_posix()
            except ValueError:
                pass
        return (Path("memory") / fact.topic / (fact.id + ".md")).as_posix()
