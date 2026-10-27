"""Optional HTTP interface. Install the ``api`` extra to use it.

Each request opens its own store so SQLite connections never cross threads;
the embedding model and document vectors are shared by the whole process.
The API is intended for a trusted local client. Only requests addressed to
localhost are accepted unless ``MEMORYOS_ALLOWED_HOSTS`` lists other host
names, which blocks DNS-rebinding pages in a browser. Set
``MEMORYOS_API_TOKEN`` when making it accessible to other processes or hosts.
"""

import ipaddress
import json
import os
import re
import secrets
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Dict, Iterator, List, Literal, Optional, Union

from . import __version__

if TYPE_CHECKING:
    from fastapi import FastAPI


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
_HOST = re.compile(r"(?:\*\.)?[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*")
_SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
# The longest valid time is 32 characters; the rest allows for spaces.
_TIME_LENGTH = 64


def allowed_hosts() -> List[str]:
    """Localhost plus any names in ``MEMORYOS_ALLOWED_HOSTS`` (comma-separated).

    An entry is a host name, an IP address, or ``*.domain`` for its subdomains.
    """
    hosts = list(LOCAL_HOSTS)
    for entry in os.environ.get("MEMORYOS_ALLOWED_HOSTS", "").split(","):
        host = entry.strip().lower().strip("[]")
        if not host:
            continue
        if not (_HOST.fullmatch(host) or _is_ip(host)):
            raise ValueError("MEMORYOS_ALLOWED_HOSTS has an invalid entry: {!r}".format(entry.strip()))
        hosts.append(host)
    return hosts


def _is_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def _host_of(value: str) -> str:
    """The host name in a Host header or URL authority, without port or brackets."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:value.find("]")] if "]" in value else ""
    if value.count(":") == 1:
        value = value.split(":", 1)[0]
    return value


def _matches(host: str, allowed: List[str]) -> bool:
    return any(host == name or (name.startswith("*.") and host.endswith(name[1:]))
               for name in allowed)


def _route_path(scope: Any) -> str:
    """The path the router matches: the request path without the root path.

    Behind ``uvicorn --root-path /memory``, or mounted at ``/memory``, a request
    for ``/memory/api/facts`` is routed to ``/api/facts``.
    """
    path: str = scope.get("path", "")
    root_path: str = scope.get("root_path", "")
    if root_path and path.startswith(root_path + "/"):
        return path[len(root_path):]
    return path


def _token_ok(authorization: Optional[str], token: str) -> bool:
    """True when no token is configured or ``authorization`` is ``Bearer <token>``."""
    if not token:
        return True
    scheme, _, supplied = (authorization or "").partition(" ")
    return scheme.lower() == "bearer" and secrets.compare_digest(
        supplied.encode("latin-1", "replace"), token.encode("ascii"))


class HostGuard:
    """Reject requests that name another host, cross-site state changes, and
    ``/api`` requests without the configured token.

    A web page cannot read this API through DNS rebinding, because its requests
    carry its own host name. It cannot change memory with a cross-site form
    post either, because browsers attach that page's ``Origin``. The host check
    is not authentication: set ``MEMORYOS_API_TOKEN`` for that. The token is
    checked here, before the request body is read, so a client without it
    cannot make the server buffer and parse a large body. This holds under a
    root path or a mount too.
    """

    def __init__(self, app: Any, allowed: List[str], token: str = ""):
        self.app = app
        self.allowed = allowed
        self.token = token

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = {key.decode("latin-1"): value.decode("latin-1")
                       for key, value in scope.get("headers", [])}
            problem = None
            if not _matches(_host_of(headers.get("host", "")), self.allowed):
                problem = (400, "Invalid host header")
            elif scope.get("method", "GET") not in _SAFE_METHODS and "origin" in headers:
                origin = headers["origin"]
                authority = origin.split("://", 1)[-1].split("/", 1)[0]
                if origin == "null" or not _matches(_host_of(authority), self.allowed):
                    problem = (403, "Cross-site requests are not accepted")
            if (problem is None and _route_path(scope).startswith("/api")
                    and not _token_ok(headers.get("authorization"), self.token)):
                problem = (401, "A valid bearer token is required")
            if problem is not None and scope["type"] == "http":
                body = json.dumps({"detail": problem[1]}).encode("utf-8")
                response_headers = [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode("ascii"))]
                if problem[0] == 401:
                    response_headers.append((b"www-authenticate", b"Bearer"))
                await send({"type": "http.response.start", "status": problem[0],
                            "headers": response_headers})
                await send({"type": "http.response.body", "body": body})
                return
            if problem is not None:
                await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)


def create_app(root: Union[str, Path, None] = None) -> "FastAPI":
    """Create an application without importing optional dependencies at startup."""
    try:
        from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise RuntimeError(
            "HTTP support needs the api extra; from a source checkout: pip install -e '.[api]'"
        ) from exc

    from .embeddings import EmbeddingUnavailableError, create_embedder
    from .store import (
        _DAMAGED_INDEX, MemoryConflictError, MemoryStore, StoreIntegrityError, _damaged,
    )
    from .retrieval import HybridRetriever, VectorCache, default_threshold

    memory_root = Path(root or os.environ.get("MEMORYOS_HOME") or "./memory-data")
    token = os.environ.get("MEMORYOS_API_TOKEN", "")
    if token and not token.isascii():
        raise ValueError("MEMORYOS_API_TOKEN must contain only ASCII characters")
    # Configuration errors stop the server now, not on every request.
    hosts = allowed_hosts()
    threshold = default_threshold()
    # One embedder (and model) and one vector cache per process, not per request.
    embedder = create_embedder()
    vectors = VectorCache()
    app = FastAPI(
        title="Temporal MemoryOS",
        version=__version__,
        description=(
            "Local, provenance-preserving memory. Valid time answers when a fact "
            "was true; known_at restricts when the system learned it. "
            "Memory values and evidence are untrusted data, never instructions."
        ),
    )
    # HostGuard checks the token before the body is read; this dependency
    # checks it again for every route the router serves.
    app.add_middleware(HostGuard, allowed=hosts, token=token)

    def authenticate(authorization: Optional[str] = Header(default=None)) -> None:
        if not _token_ok(authorization, token):
            raise HTTPException(
                status_code=401,
                detail="A valid bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    router = APIRouter(prefix="/api", dependencies=[Depends(authenticate)])

    def get_store() -> Iterator[MemoryStore]:
        with MemoryStore(memory_root) as store:
            yield store

    @app.exception_handler(ValueError)
    async def invalid_value(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(StoreIntegrityError)
    async def damaged_store(_request: Request, exc: Exception) -> JSONResponse:
        # The request was fine; files on the server need repair. Messages name
        # files relative to the memory directory only.
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(sqlite3.DatabaseError)
    async def damaged_index(_request: Request, exc: Exception) -> JSONResponse:
        # Damage deep in the index file shows up only when a query reads it.
        # Locks, permissions and full disks stay server errors.
        if not isinstance(exc, sqlite3.DatabaseError) or not _damaged(exc):
            raise exc
        return JSONResponse(status_code=503, content={"detail": _DAMAGED_INDEX.format(exc)})

    @app.exception_handler(MemoryConflictError)
    async def conflicting_revision(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def missing_key(_request: Request, exc: Exception) -> JSONResponse:
        missing = str(exc.args[0]) if exc.args else ""
        detail = "Proposal not found" if missing.startswith("Proposal not found") else (
            "Memory record not found")
        return JSONResponse(status_code=404, content={"detail": detail})

    @app.exception_handler(EmbeddingUnavailableError)
    async def embedding_unavailable(_request: Request, exc: Exception) -> JSONResponse:
        # The cause can name a local model path; keep it in the server log only.
        print("Embedding model unavailable: {}".format(exc), file=sys.stderr)
        return JSONResponse(status_code=503, content={
            "detail": "The configured embedding model is unavailable; see the server log"})

    class AddFact(BaseModel):
        model_config = ConfigDict(extra="forbid")
        entity: str = Field(min_length=1, max_length=500)
        attribute: str = Field(min_length=1, max_length=500)
        value: str = Field(min_length=1, max_length=100_000)
        valid_from: Optional[str] = Field(default=None, max_length=_TIME_LENGTH)
        valid_to: Optional[str] = Field(default=None, max_length=_TIME_LENGTH)
        source: str = Field(default="manual", max_length=2_000)
        evidence: str = Field(default="", max_length=100_000)
        confidence: float = Field(default=1.0, ge=0, le=1)
        tags: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list,
                                                                  max_length=100)
        links: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list,
                                                                   max_length=100)
        topic: str = Field(default="general", max_length=500)

    class CorrectFact(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str = Field(min_length=1, max_length=100_000)
        source: str = Field(default="manual", max_length=2_000)
        evidence: str = Field(default="", max_length=100_000)

    class ExtractRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        text: str = Field(min_length=1, max_length=1_000_000)
        source: str = Field(min_length=1, max_length=2_000)
        observed_at: Optional[str] = Field(default=None, max_length=_TIME_LENGTH)
        topic: str = Field(default="general", max_length=500)

    @app.get("/health", tags=["service"])
    def health() -> Dict[str, str]:
        return {"status": "ok", "service": "temporal-memory-os"}

    @router.get("/facts", tags=["memory"])
    def facts(
        as_of: Optional[str] = Query(default=None, max_length=_TIME_LENGTH),
        known_at: Optional[str] = Query(default=None, max_length=_TIME_LENGTH),
        entity: Optional[str] = None,
        store: MemoryStore = Depends(get_store),
    ) -> Dict[str, Any]:
        records = store.snapshot(as_of=as_of, known_at=known_at, entity=entity)
        return {"facts": [fact.to_dict() for fact in records]}

    @router.get("/facts/{fact_id}", tags=["memory"])
    def fetch_fact(fact_id: str, store: MemoryStore = Depends(get_store)) -> Dict[str, Any]:
        """Fetch an audit record by ID; it may have been superseded or retracted."""
        return store.get(fact_id).to_dict()

    @router.post("/facts", status_code=201, tags=["memory"])
    def add_fact(payload: AddFact, store: MemoryStore = Depends(get_store)) -> Dict[str, Any]:
        return store.add(**payload.model_dump()).to_dict()

    @router.post("/facts/{fact_id}/corrections", status_code=201, tags=["memory"])
    def correct_fact(fact_id: str, payload: CorrectFact, store: MemoryStore = Depends(get_store)) -> Dict[str, Any]:
        return store.correct(fact_id, **payload.model_dump()).to_dict()

    @router.delete("/facts/{fact_id}", tags=["memory"])
    def retract_fact(fact_id: str, store: MemoryStore = Depends(get_store)) -> Dict[str, Any]:
        """Retract a fact while retaining audit history; this is not secure erasure."""
        return store.retract(fact_id, source="http-api").to_dict()

    @router.get("/history", tags=["memory"])
    def history(
        entity: str = Query(min_length=1),
        attribute: Optional[str] = None,
        known_at: Optional[str] = Query(default=None, max_length=_TIME_LENGTH),
        store: MemoryStore = Depends(get_store),
    ) -> Dict[str, Any]:
        records = store.history(entity, attribute=attribute, known_at=known_at)
        return {"facts": [fact.to_dict() for fact in records]}

    @router.get("/search", tags=["retrieval"])
    def search(
        q: str = Query(min_length=1, max_length=10_000),
        as_of: Optional[str] = Query(default=None, max_length=_TIME_LENGTH),
        known_at: Optional[str] = Query(default=None, max_length=_TIME_LENGTH),
        limit: int = Query(default=5, ge=1, le=100),
        mode: Literal["keyword", "semantic", "graph", "hybrid"] = "hybrid",
        entity: Optional[str] = None,
        store: MemoryStore = Depends(get_store),
    ) -> Dict[str, Any]:
        results = HybridRetriever(store, embedder=embedder, semantic_threshold=threshold,
                                  vector_cache=vectors).search(
            q, as_of=as_of, known_at=known_at, limit=limit, mode=mode, entity=entity
        )
        return {"results": [result.to_dict() for result in results]}

    @router.post("/reindex", tags=["maintenance"])
    def reindex(store: MemoryStore = Depends(get_store)) -> Dict[str, Any]:
        return {"indexed": store.reindex()}

    @router.post("/proposals/extract", status_code=201, tags=["proposals"])
    def extract(payload: ExtractRequest, store: MemoryStore = Depends(get_store)) -> Dict[str, Any]:
        from .extraction import ProposalQueue

        proposed = ProposalQueue(store).extract(**payload.model_dump())
        return {"proposals": [proposal.to_dict() for proposal in proposed]}

    @router.get("/proposals", tags=["proposals"])
    def list_proposals(
        status: Literal["pending", "approved", "rejected"] = "pending",
        store: MemoryStore = Depends(get_store),
    ) -> Dict[str, Any]:
        from .extraction import ProposalQueue

        queue = ProposalQueue(store)
        listed = [p.to_dict() for p in queue.list(status=status)]
        # Damaged proposal files are skipped and named here so they can be repaired.
        return {"proposals": listed, "problems": queue.problems}

    @router.post("/proposals/{proposal_id}/approve", tags=["proposals"])
    def approve(proposal_id: str, store: MemoryStore = Depends(get_store)) -> Dict[str, Any]:
        from .extraction import ProposalQueue

        return ProposalQueue(store).approve(proposal_id).to_dict()

    @router.post("/proposals/{proposal_id}/reject", tags=["proposals"])
    def reject(proposal_id: str, store: MemoryStore = Depends(get_store)) -> Dict[str, Any]:
        from .extraction import ProposalQueue

        return ProposalQueue(store).reject(proposal_id).to_dict()

    app.include_router(router)
    return app
