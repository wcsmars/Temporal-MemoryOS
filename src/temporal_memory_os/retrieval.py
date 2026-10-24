"""Time-filtered keyword, embedding and one-hop relationship retrieval."""

from __future__ import annotations

import math
import numbers
import operator
import os
import sqlite3
import threading
import unicodedata
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Set, Tuple

from .embeddings import STOPWORDS, Embedder, create_embedder, normalized_tokens, tokenize
from .models import Fact, TimeInput

Ranking = List[Tuple[str, float]]
# Dimensions, indices of non-zero values (None when stored densely), values,
# norm, and the document's tokens. Arrays of doubles keep a 384-dimension
# vector near 3 KB instead of tens of KB as Python objects.
VectorEntry = Tuple[int, Optional[array], array, float, frozenset]
MAX_KEYWORD_TERMS = 64
# Accents are folded inside the index (Zurich finds Zürich) and combining marks
# stay inside words, matching the Python tokenizer.
_FTS_TOKENIZER = "unicode61 remove_diacritics 2 categories 'L* N* Co M*'"


def _name(text: str) -> str:
    """Comparable form of an entity name or link: NFKC, case-folded, trimmed."""
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).casefold()).strip()


@lru_cache(maxsize=65536)
def _words(text: str) -> str:
    """Tokenized text for the keyword index; cached because records repeat across queries."""
    return " ".join(tokenize(text))


def _content(tokens: Iterable[str], keep: Set[str]) -> List[str]:
    """Drop stopwords, except words that belong to an entity name in the snapshot."""
    return [token for token in tokens if token not in STOPWORDS or token in keep]


class FactSource(Protocol):
    """What retrieval needs from a store: a resolved temporal snapshot."""

    def snapshot(self, as_of: TimeInput = None, known_at: TimeInput = None,
                 entity: Optional[str] = None) -> List[Fact]:
        ...


class VectorCache:
    """Document vectors keyed by embedder name and the exact embedded text.

    Keying by text rather than fact ID means an edited record, or a different
    model, can never reuse a stale vector. One cache can be shared by many
    retrievers (the HTTP and MCP servers keep one per process); access is
    guarded by a lock.
    """

    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str], VectorEntry] = {}
        self._lock = threading.Lock()  # The servers share one cache across threads.

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: Tuple[str, str]) -> Optional[VectorEntry]:
        with self._lock:
            return self._entries.get(key)

    def store(self, entries: Dict[Tuple[str, str], VectorEntry], limit: int) -> None:
        with self._lock:
            if not entries or len(entries) > limit:
                return  # Nothing new, or larger than the whole cache.
            if len(self._entries) + len(entries) > limit:
                self._entries.clear()
            self._entries.update(entries)


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


def _competition_ranks(ranking: Ranking) -> Iterator[Tuple[str, float, int]]:
    """Yield (id, score, rank) for a ranking sorted best first; ties share a rank.

    Without this, reciprocal rank fusion would turn an exact tie (every seed
    entity scores 1.0 in the graph channel, for example) into different ranks
    decided by random fact IDs.
    """
    rank, previous = 0, None
    for position, (fact_id, score) in enumerate(ranking, start=1):
        if score != previous:
            rank, previous = position, score
        yield fact_id, score, rank


def _threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("semantic_threshold must be a finite number between -1 and 1")
    try:
        number = float(value)
    except OverflowError:  # An integer or fraction too large for a float.
        number = math.inf
    if not math.isfinite(number) or not -1.0 <= number <= 1.0:
        raise ValueError("semantic_threshold must be a finite number between -1 and 1")
    return number


# Marks that may surround a name written out in a query: opening quotes and
# brackets (and the Spanish \u00bf \u00a1) before it; closing quotes and brackets,
# sentence punctuation and a possessive "'s" after it.
_OPENING = "\"'([{\u00ab\u2039\u201c\u201e\u2018\u201a\u00bf\u00a1"
_CLOSING = ".,!?;:\"')]}\u00bb\u203a\u201d\u201c\u2019"
_APOSTROPHES = "'\u2019"


def _lead(word: str) -> int:
    """How many opening marks ``word`` starts with."""
    return len(word) - len(word.lstrip(_OPENING))


def _bare_end(word: str) -> int:
    """Where ``word`` ends once trailing closing marks and possessives are dropped."""
    end = len(word)
    while end:
        if word[end - 1] in _CLOSING:
            end -= 1
        elif word[end - 1] == "s" and end >= 2 and word[end - 2] in _APOSTROPHES:
            end -= 2
        else:
            break
    return end


def _bare(word: str) -> str:
    """A word without the marks that may surround it. A query word and a name part
    it writes always have the same bare form, so lookups are keyed by it."""
    word = word[_lead(word):]
    return word[:_bare_end(word)]


def _writes(word: str, part: str) -> bool:
    """True if a query word is ``part`` with only surrounding marks added.

    '"c++"?' and "c++'s" write "c++"; "st." writes "st." and "st"; "!!!?"
    writes "!!!" and "!"; but "c++" does not write "c", and "st" does not
    write "st.".
    """
    start = _lead(word) - _lead(part)
    return (start >= 0 and word.startswith(part, start)
            and _bare_end(word[start + len(part):]) == 0)


class _QueryWords:
    """A query's words, indexed by bare form so a name is found in constant time."""

    def __init__(self, query: str) -> None:
        self.words = _name(query).split()
        self.starts: Dict[str, List[int]] = {}
        for position, word in enumerate(self.words):
            self.starts.setdefault(_bare(word), []).append(position)

    def write(self, name: str) -> bool:
        """True if ``name`` occurs as consecutive whole words of the query."""
        parts = name.split()
        if not parts:
            return False
        words, count = self.words, len(parts)
        return any(start + count <= len(words)
                   and all(_writes(words[start + offset], part)
                           for offset, part in enumerate(parts))
                   for start in self.starts.get(_bare(parts[0]), ()))


def default_threshold() -> float:
    """``MEMORYOS_SEMANTIC_THRESHOLD`` or 0.25. Servers call this once at start-up."""
    configured = os.environ.get("MEMORYOS_SEMANTIC_THRESHOLD", "").strip()
    if not configured:
        return 0.25
    try:
        return _threshold(float(configured))
    except ValueError as exc:
        raise ValueError("MEMORYOS_SEMANTIC_THRESHOLD must be a number between -1 and 1") from exc


def _validated(vector: Sequence[float]) -> Tuple[List[float], float]:
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding vectors must contain only finite numbers")
    try:
        norm = math.sqrt(math.fsum(value * value for value in values))
    except OverflowError:  # Finite squares whose sum is not.
        norm = math.inf
    if not math.isfinite(norm):
        raise ValueError("Embedding vectors are too large to compare")
    return values, norm


def _entry(values: List[float], norm: float, tokens: frozenset) -> VectorEntry:
    nonzero = [index for index, value in enumerate(values) if value]
    if len(nonzero) * 2 < len(values):  # Hashing vectors are mostly zeros.
        return (len(values), array("i", nonzero), array("d", (values[i] for i in nonzero)),
                norm, tokens)
    return len(values), None, array("d", values), norm, tokens


def _dot(query: List[float], entry: VectorEntry) -> float:
    _, indices, values, _, _ = entry
    if indices is None:
        return math.fsum(map(operator.mul, query, values))
    return math.fsum(query[index] * value for index, value in zip(indices, values))


class HybridRetriever:
    """Rank only facts valid at ``as_of`` and known by ``known_at``.

    Keyword retrieval uses SQLite FTS5 BM25 over the selected temporal snapshot,
    so old versions cannot affect corpus statistics or displace eligible facts.
    ``semantic`` selects the embedding channel; with the default offline
    provider that channel is transparently labeled ``lexical_embedding``.
    Graph retrieval expands entity links in either direction by exactly one hop.
    Hybrid scores use reciprocal rank fusion (k=60) over competition ranks, so
    tied raw scores contribute equally. Common English function words are
    dropped from keyword and lexical queries (see ``embeddings.STOPWORDS``)
    unless they are part of an entity name in the snapshot; only the first 64
    distinct keyword terms are used. Keyword matching folds accents.

    ``semantic_threshold`` (default 0.25, or ``MEMORYOS_SEMANTIC_THRESHOLD``)
    applies to learned models only; different models need different values.
    Document vectors are cached per retriever (at most ``cache_limit``);
    pass a shared ``vector_cache`` to reuse them across retrievers.
    """

    available_modes = ("keyword", "semantic", "graph", "hybrid")
    cache_limit = 50_000

    def __init__(self, store: FactSource, embedder: Optional[Embedder] = None,
                 semantic_threshold: Optional[float] = None,
                 vector_cache: Optional[VectorCache] = None):
        self.semantic_threshold = (default_threshold() if semantic_threshold is None
                                   else _threshold(semantic_threshold))
        self.store = store
        self.embedder: Embedder = embedder if embedder is not None else create_embedder()
        self.vectors = vector_cache if vector_cache is not None else VectorCache()

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
            wanted = _name(entity)
            facts = [fact for fact in facts if _name(fact.entity) == wanted]
        semantic = getattr(self.embedder, "semantic", True)
        has_words = bool(tokenize(query))
        # A learned model can match a query with no word tokens (emoji, symbols).
        if not facts or not (has_words or (semantic and query.strip()
                                            and mode in ("semantic", "hybrid"))):
            return []

        channels: Dict[str, Ranking] = {}
        # Stopwords that are part of an entity name ("The Who") still count.
        keep = {token for fact in facts for token in normalized_tokens(fact.entity)}
        if mode in ("keyword", "hybrid") and has_words:
            channels["keyword"] = self._keyword(query, facts, keep)
        if mode in ("semantic", "hybrid"):
            channels[self.embedding_channel] = self._embedding(query, facts, keep)
        if mode in ("graph", "hybrid") and has_words:
            channels["graph"] = self._graph(query, facts)

        raw_scores: Dict[str, Dict[str, float]] = {}
        fused: Dict[str, float] = {}
        for channel, ranking in channels.items():
            for fact_id, raw, rank in _competition_ranks(ranking):
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
    def _keyword(query: str, facts: List[Fact], keep: Set[str]) -> Ranking:
        terms = list(dict.fromkeys(_content(tokenize(query), keep)))[:MAX_KEYWORD_TERMS]
        if not terms:
            return []
        # MATCH receives quoted word tokens only; user operators are never SQL
        # or FTS syntax. A snapshot-local table enforces time filtering first.
        # Documents and query go through the same Unicode normalization; the
        # alias table is not applied here, so documents are matched as written.
        expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in terms)
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE selected USING fts5("
                "id UNINDEXED, entity, attribute, value, context, "
                "tokenize=\"{}\")".format(_FTS_TOKENIZER)
            )
            connection.executemany(
                "INSERT INTO selected VALUES (?, ?, ?, ?, ?)",
                [(fact.id, _words(fact.entity), _words(fact.attribute), _words(fact.value),
                  " ".join(_words(text) for text in (fact.evidence, *fact.tags, *fact.links,
                                                     fact.topic)))
                 for fact in facts],
            )
            return [(row[0], float(-row[1])) for row in connection.execute(
                "SELECT id, bm25(selected, 0, 4, 2, 2, 1) AS score "
                "FROM selected WHERE selected MATCH ? ORDER BY score ASC, id ASC",
                (expression,),
            ).fetchall()]
        finally:
            connection.close()

    def _embedding(self, query: str, facts: List[Fact], keep: Set[str]) -> Ranking:
        name = str(getattr(self.embedder, "name", type(self.embedder).__name__))
        keys = [(name, self._text(fact)) for fact in facts]
        entries: Dict[Tuple[str, str], VectorEntry] = {}
        for key in keys:
            cached = self.vectors.get(key)
            if cached is not None:
                entries[key] = cached
        missing = list(dict.fromkeys(key for key in keys if key not in entries))
        lexical = not getattr(self.embedder, "semantic", True)
        query_tokens = _content(normalized_tokens(query), keep)
        # Word hashing compares words, so question words are left out of its
        # query; a learned model reads the whole question.
        embedded_query = " ".join(query_tokens) if lexical else query
        vectors = self.embedder.embed([embedded_query] + [text for _, text in missing])
        if len(vectors) != len(missing) + 1:
            raise ValueError("Embedding provider returned the wrong number of vectors")
        query_vector, query_norm = _validated(vectors[0])
        computed: Dict[Tuple[str, str], VectorEntry] = {}
        for key, vector in zip(missing, vectors[1:]):
            values, norm = _validated(vector)
            if len(values) != len(query_vector):
                raise ValueError("Embedding provider returned inconsistent vector dimensions")
            computed[key] = _entry(values, norm, frozenset(normalized_tokens(key[1])))
        self.vectors.store(computed, self.cache_limit)
        entries.update(computed)
        threshold = 0.0 if lexical else self.semantic_threshold
        wanted = set(query_tokens)
        ranking: Ranking = []
        for fact, key in zip(facts, keys):
            entry = entries[key]
            dimensions, _, _, norm, tokens = entry
            # Hash collisions must never produce apparently relevant results.
            if lexical and wanted.isdisjoint(tokens):
                continue
            if dimensions != len(query_vector):
                raise ValueError("Embedding provider returned inconsistent vector dimensions")
            if not norm or not query_norm:
                continue
            score = _dot(query_vector, entry) / (query_norm * norm)
            score = max(-1.0, min(1.0, score))
            if score > threshold:
                ranking.append((fact.id, score))
        return sorted(ranking, key=lambda item: (-item[1], item[0]))

    @staticmethod
    def _graph(query: str, facts: List[Fact]) -> Ranking:
        """Seed entities whose words all appear in the query, then follow links.

        Names that differ only in punctuation share their words (C, C#, C++);
        among those, the names written out in the query as whole words win
        (quotes, brackets and trailing punctuation around them are allowed). A
        name with no word characters seeds only when written out as a word.
        """
        query_tokens = set(tokenize(query))
        written = _QueryWords(query)
        entities = {_name(fact.entity): frozenset(tokenize(fact.entity)) for fact in facts}
        seeds = {entity for entity, tokens in entities.items()
                 if (tokens and tokens <= query_tokens) or (not tokens and written.write(entity))}
        groups: Dict[frozenset, List[str]] = {}
        for entity in seeds:
            groups.setdefault(entities[entity], []).append(entity)
        for members in groups.values():
            shown = [entity for entity in members if len(members) > 1 and written.write(entity)]
            if shown:
                # Names sharing their words: keep those written out in the query.
                seeds.difference_update(entity for entity in members if entity not in shown)
        if not seeds:
            return []
        adjacent = set()
        for fact in facts:
            entity = _name(fact.entity)
            linked = {_name(link) for link in fact.links}
            if entity in seeds:
                adjacent.update(linked)
            if linked & seeds:
                adjacent.add(entity)
        ranking: Ranking = []
        for fact in facts:
            entity = _name(fact.entity)
            if entity in seeds:
                ranking.append((fact.id, 1.0))
            elif entity in adjacent:
                ranking.append((fact.id, 0.5))
        return sorted(ranking, key=lambda item: (-item[1], item[0]))

    def _citation(self, fact: Fact) -> str:
        if hasattr(self.store, "path_for") and hasattr(self.store, "root"):
            try:
                path = Path(self.store.path_for(fact.id))
                return path.relative_to(Path(self.store.root)).as_posix()
            except (KeyError, ValueError):
                pass  # Removed or moved since the snapshot: cite the canonical location.
        return (Path("memory") / fact.topic / (fact.id + ".md")).as_posix()
