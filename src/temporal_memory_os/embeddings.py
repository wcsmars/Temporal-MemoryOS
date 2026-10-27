"""Embedding providers with a dependency-free, explicitly lexical default.

The hashing provider is useful for offline demos and reproducible tests. It is
not a pretrained semantic model. Install the ``semantic`` extra and select
SentenceTransformerEmbedder (directly, or with ``MEMORYOS_EMBEDDING_MODEL``)
to use learned semantic embeddings.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import unicodedata
from typing import Any, List, Optional, Protocol, Sequence


_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
# Dropped from queries (not from documents) so that question words do not make
# every fact look relevant. A query made only of these words matches nothing,
# unless it names an entity whose name uses them ("The Who").
STOPWORDS = frozenset("""
a about an and are as at be been but by can could did do does for from had has have he her
hers him his how i if in into is it its just me my of on or our she so that the their them
then there these they this to was we were what when where which who whom why will with
would you your
""".split())
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
    """Casefolded Unicode words; punctuation never becomes query syntax.

    Text is NFKC-normalized and casefolded first, so composed and decomposed
    accents, ligatures and case expansions (such as German sharp s) compare
    equal. Combining marks continue a word, which keeps scripts such as
    Devanagari intact.
    """
    folded = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).casefold())
    if folded.isascii():
        return _TOKEN.findall(folded)
    tokens: List[str] = []
    current: List[str] = []
    for character in folded:
        if (character.isalnum() and character != "_") or (
                current and unicodedata.category(character).startswith("M")):
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def normalized_tokens(text: str) -> List[str]:
    return [_ALIASES.get(token, token) for token in tokenize(text)]


class EmbeddingUnavailableError(RuntimeError):
    """The configured embedding model could not be loaded or run, or returned unusable vectors."""


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

    semantic = False

    def __init__(self, dimensions: int = 384):
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 32:
            raise ValueError("dimensions must be an integer of at least 32")
        self.dimensions = dimensions
        # Part of the vector cache key, so it must change with the vector size.
        self.name = "lexical-hash-v1-{}".format(dimensions)

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

    Models must support ``SentenceTransformer.encode``. Code hosted with a model
    on the Hub is never executed (``trust_remote_code=False``), but a local
    model directory is loaded like a Python package and should be trusted
    like code. A model already in the local cache is loaded without network
    access; otherwise it is downloaded. The model is loaded once per embedder
    object and reused.
    """

    semantic = True

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        self.model_name = os.path.expanduser(model_name) if model_name.startswith("~") else model_name
        self.name = "sentence-transformers:" + model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        # One lock covers loading and encoding: server threads share this
        # object, and model inference is not documented as thread-safe.
        with self._lock:
            if self._model is None:
                self._load()
            try:
                vectors = self._model.encode(
                    list(texts), normalize_embeddings=True, show_progress_bar=False,
                )
            except Exception as error:
                raise EmbeddingUnavailableError("The embedding model failed: {}: {}".format(
                    type(error).__name__, error)) from error
        return [[float(value) for value in row] for row in vectors]

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as error:  # A broken torch install can fail with OSError and others.
            # Only a missing package gets the install hint. An installed one whose
            # own imports fail (a mismatched transformers, say) needs the cause.
            if isinstance(error, ModuleNotFoundError) and error.name == "sentence_transformers":
                raise EmbeddingUnavailableError(
                    "Semantic embeddings need the optional dependency: from a source checkout, "
                    "pip install -e '.[semantic]'"
                ) from error
            raise EmbeddingUnavailableError("Could not import sentence-transformers ({}: {})".format(
                type(error).__name__, error)) from error
        try:
            # A model already in the local cache loads without any network
            # access; offline, the online attempt would retry for over a minute.
            self._model = SentenceTransformer(self.model_name, trust_remote_code=False,
                                              local_files_only=True)
            return
        except Exception:
            pass
        try:
            self._model = SentenceTransformer(self.model_name, trust_remote_code=False)
        except Exception as error:  # The library raises many unrelated types.
            raise EmbeddingUnavailableError(
                "Could not load the embedding model {!r} ({}: {})".format(
                    self.model_name, type(error).__name__, error)) from error


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

