"""Embedding-provider behavior that mutation testing showed no other test
checked: argument validation, the default model and failure diagnostics.
"""

import sys
import types

import pytest

import temporal_memory_os.embeddings as embeddings_module
from temporal_memory_os.embeddings import (create_embedder, EmbeddingUnavailableError,
                                           HashingEmbedder, SentenceTransformerEmbedder)


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HubError(Exception):
    """An unrelated exception type, like the ones model libraries raise."""


def fake_sentence_transformers(monkeypatch, model_class):
    """Install a stand-in ``sentence_transformers`` module exposing ``model_class``."""
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=model_class))


class ByteOrderRequiredInt(int):
    """``int`` with Python 3.10's ``from_bytes``, whose byte order has no default."""

    @classmethod
    def from_bytes(cls, data, byteorder, *, signed=False):
        return int.from_bytes(data, byteorder, signed=signed)


# HashingEmbedder


@pytest.mark.parametrize("dimensions", [64.0, "64"], ids=["float", "string"])
def test_hashing_dimensions_that_are_not_integers_are_rejected(dimensions):
    """A vector size that is not an integer is rejected with ValueError, however large."""
    with pytest.raises(ValueError):
        HashingEmbedder(dimensions=dimensions)


def test_hashing_states_the_byte_order_python_3_10_requires(monkeypatch):
    """Hashed vectors do not rely on a byte-order default, which Python 3.10 lacks."""
    embedder = HashingEmbedder(dimensions=64)
    expected = embedder.embed(["decaf tea tea", "Acme application status"])
    monkeypatch.setattr(embeddings_module, "int", ByteOrderRequiredInt, raising=False)
    assert embedder.embed(["decaf tea tea", "Acme application status"]) == expected


# SentenceTransformerEmbedder


def test_default_learned_model_is_minilm(monkeypatch):
    """Without a name, the learned provider loads and is named after the MiniLM model."""
    loaded = []

    class Recording:
        def __init__(self, name, **options):
            loaded.append(name)

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            return [[1.0, 0.0] for _ in texts]

    fake_sentence_transformers(monkeypatch, Recording)
    embedder = SentenceTransformerEmbedder()
    assert embedder.embed(["text"]) == [[1.0, 0.0]]
    assert loaded == [DEFAULT_MODEL]
    assert embedder.name == "sentence-transformers:" + DEFAULT_MODEL


def test_model_load_failure_names_the_model_and_the_cause(monkeypatch):
    """A model that cannot load raises an error naming the model, the cause's type and its text."""
    class Failing:
        def __init__(self, name, **options):
            raise HubError("quota exceeded")

    fake_sentence_transformers(monkeypatch, Failing)
    with pytest.raises(EmbeddingUnavailableError) as caught:
        SentenceTransformerEmbedder("org/missing-model").embed(["text"])
    message = str(caught.value)
    assert "org/missing-model" in message
    assert "HubError" in message
    assert "quota exceeded" in message
    assert isinstance(caught.value.__cause__, HubError)


def test_encode_failure_names_the_cause(monkeypatch):
    """A model that fails while encoding raises an error naming the cause's type and its text."""
    class Broken:
        def __init__(self, name, **options):
            pass

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            raise HubError("CUDA out of memory")

    fake_sentence_transformers(monkeypatch, Broken)
    with pytest.raises(EmbeddingUnavailableError) as caught:
        SentenceTransformerEmbedder("local-model").embed(["text"])
    message = str(caught.value)
    assert "HubError" in message
    assert "CUDA out of memory" in message
    assert isinstance(caught.value.__cause__, HubError)


# ==========================================================================
# Argument errors name the rejected argument
# ==========================================================================


@pytest.mark.parametrize("dimensions", [31, 0, -64, True, 64.0, "64", None],
                         ids=["too-small", "zero", "negative", "bool", "float", "string", "none"])
def test_hashing_size_error_names_dimensions(dimensions):
    """A rejected hashing vector size raises ValueError whose text names ``dimensions``."""
    with pytest.raises(ValueError) as caught:
        HashingEmbedder(dimensions=dimensions)
    assert "dimensions" in str(caught.value)


@pytest.mark.parametrize("model_name", ["", None, 42, b"org/model"],
                         ids=["empty", "none", "int", "bytes"])
def test_learned_model_name_error_names_model_name(model_name):
    """An empty or non-string learned model name raises ValueError whose text names ``model_name``."""
    with pytest.raises(ValueError) as caught:
        SentenceTransformerEmbedder(model_name)
    assert "model_name" in str(caught.value)


@pytest.mark.parametrize("model_name", [42, b"org/model", ["org/model"]],
                         ids=["int", "bytes", "list"])
def test_factory_model_name_error_names_model_name(model_name):
    """A non-string model passed to create_embedder raises ValueError whose text names ``model_name``."""
    with pytest.raises(ValueError) as caught:
        create_embedder(model_name)
    assert "model_name" in str(caught.value)
