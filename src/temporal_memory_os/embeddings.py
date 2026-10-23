"""Embedding providers with a dependency-free, explicitly lexical default.

The hashing provider is useful for offline demos and reproducible tests. It is
not a pretrained semantic model. Install the ``semantic`` extra and explicitly
select SentenceTransformerEmbedder to use learned semantic embeddings.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List, Optional, Protocol, Sequence


_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
# Deliberately small and inspectable. This is query normalization, not an LLM.
_ALIASES = {
    "applications": "application", "applied": "application",
    "interviews": "interview", "interviewing": "interview",
    "completed": "complete", "finished": "complete",
    "invitation": "invite", "invitations": "invite", "invited": "invite",
    "preferences": "preference", "prefers": "preference",
    "employers": "employer", "companies": "company",
    "locations": "location", "cities": "city",
}


def tokenize(text: str) -> List[str]:
    """Casefolded Unicode words; punctuation never becomes query syntax."""
    return _TOKEN.findall(text.casefold())


def normalized_tokens(text: str) -> List[str]:
    return [_ALIASES.get(token, token) for token in tokenize(text)]


class Embedder(Protocol):
    name: str
    semantic: bool

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class HashingEmbedder:
    """Stable signed token hashing with L2 normalization, without model downloads.

    Python's randomized ``hash`` is intentionally avoided. The output depends
    only on input words and ``dimensions`` and is reproducible across processes.
    Small spelling/inflection aliases are listed in this module.
    """

    name = "lexical-hash-v1"
    semantic = False

    def __init__(self, dimensions: int = 384):
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 32:
            raise ValueError("dimensions must be an integer of at least 32")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        result = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in normalized_tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:7], "big") % self.dimensions
                vector[index] += 1.0 if digest[7] & 1 else -1.0
            magnitude = math.sqrt(sum(value * value for value in vector))
            result.append([value / magnitude for value in vector] if magnitude else vector)
        return result


class SentenceTransformerEmbedder:
    """Optional real semantic embeddings; first use may download model weights.

    Models must support ``SentenceTransformer.encode``. No remote code is
    trusted. Supply a local model directory for an entirely offline deployment.
    """

    semantic = True

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        self.name = "sentence-transformers:" + model_name
        self._model = None

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Semantic embeddings require the optional dependency: "
                    "pip install 'temporal-memory-os[semantic]'"
                ) from error
            self._model = SentenceTransformer(self.model_name, trust_remote_code=False)
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]


def create_embedder(model_name: Optional[str] = None) -> Embedder:
    """Choose an explicit model, the environment model, or offline hashing.

    ``MEMORYOS_EMBEDDING_MODEL`` accepts a Sentence Transformers model name or
    local model directory. Construction stays lazy: importing the dependency
    and loading/downloading weights only happens when embeddings are requested.
    An explicit non-blank model overrides the environment configuration.
    """
    if model_name is not None and not isinstance(model_name, str):
        raise ValueError("model_name must be a string or None")
    selected = (model_name or "").strip() or os.environ.get("MEMORYOS_EMBEDDING_MODEL", "").strip()
    return SentenceTransformerEmbedder(selected) if selected else HashingEmbedder()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding provider returned inconsistent vector dimensions")
    if not all(math.isfinite(float(value)) for value in (*left, *right)):
        raise ValueError("Embedding vectors must contain only finite numbers")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    value = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))
