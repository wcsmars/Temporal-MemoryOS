"""HTTP integration checks; optional dependencies are not needed by core tests.

conftest.py clears every MEMORYOS_* variable, so each test starts with no
token, the default allowed hosts and the dependency-free embedder.
"""

import sqlite3
import sys
import types
from contextlib import closing

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.routing import Mount  # noqa: E402

from temporal_memory_os.api import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://localhost") as test_client:
        yield test_client


def add(client, value="pending", valid_from="2026-09-01"):
    response = client.post(
        "/api/facts",
        json={
            "entity": "Acme application",
            "attribute": "status",
            "value": value,
            "valid_from": valid_from,
            "source": "synthetic:test",
            "topic": "career/applications",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_http_temporal_search_correction_and_retraction(client):
    first = add(client)
    add(client, "interview invited", "2026-09-15")
    historical = client.get("/api/search", params={"q": "Acme status", "as_of": "2026-09-10"})
    assert historical.status_code == 200, historical.text
    assert historical.json()["results"][0]["fact"]["value"] == "pending"

    correction = client.post(
        "/api/facts/{}/corrections".format(first["id"]),
        json={"value": "submitted", "source": "synthetic:correction"},
    )
    assert correction.status_code == 201, correction.text
    assert correction.json()["supersedes"] == first["id"]
    stale = client.post(
        "/api/facts/{}/corrections".format(first["id"]),
        json={"value": "stale client update"},
    )
    assert stale.status_code == 409, stale.text
    records = client.get("/api/facts", params={"as_of": "2026-09-10"}).json()["facts"]
    assert [fact["value"] for fact in records] == ["submitted"]

    removed = client.delete("/api/facts/{}".format(correction.json()["id"]))
    assert removed.status_code == 200, removed.text
    assert client.get("/api/facts", params={"as_of": "2026-09-10"}).json()["facts"] == []
    # Retraction keeps the original audit record available and later updates intact.
    assert client.get("/api/facts/{}".format(first["id"])).status_code == 200
    later = client.get("/api/facts", params={"as_of": "2026-09-20"}).json()["facts"]
    assert [fact["value"] for fact in later] == ["interview invited"]


def test_api_bearer_auth_protects_reads_and_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYOS_API_TOKEN", "local-test-token")
    with TestClient(create_app(tmp_path), base_url="http://localhost") as secured:
        assert secured.get("/health").json() == {"status": "ok", "service": "temporal-memory-os"}
        assert secured.get("/api/facts").status_code == 401
        assert secured.get("/api/search", params={"q": "tea"}).status_code == 401
        assert secured.post("/api/reindex").status_code == 401
        wrong = secured.get("/api/facts", headers={"Authorization": "Bearer wrong"})
        assert wrong.status_code == 401
        allowed = secured.get(
            "/api/facts", headers={"Authorization": "Bearer local-test-token"}
        )
        assert allowed.status_code == 200


def recording_bodies(app):
    """``app`` wrapped to record each request body message it reads, and that record."""
    bodies = []

    async def recording(scope, receive, send):
        async def tracked():
            message = await receive()
            if message["type"] == "http.request":
                bodies.append(message)
            return message

        await app(scope, tracked, send)

    return recording, bodies


def test_requests_without_the_token_are_refused_before_the_body_is_read(tmp_path, monkeypatch):
    """Without the token, a write gets 401 and the server never reads its body."""
    monkeypatch.setenv("MEMORYOS_API_TOKEN", "local-test-token")
    recording, bodies = recording_bodies(create_app(tmp_path))
    routes = ("/api/facts", "/api/facts/{}/corrections".format("0" * 32), "/api/proposals/extract")
    json_type = {"Content-Type": "application/json"}
    with TestClient(recording, base_url="http://localhost") as secured:
        for authorization in ({}, {"Authorization": "Bearer wrong"},
                              {"Authorization": "Basic local-test-token"}):
            for route in routes:
                response = secured.post(route, content=b"{not json",
                                        headers={**json_type, **authorization})
                assert response.status_code == 401, (route, authorization)
                assert response.json() == {"detail": "A valid bearer token is required"}
                assert response.headers["www-authenticate"] == "Bearer"
        assert bodies == []
        # With the token the body is read and validated as before.
        allowed = secured.post("/api/facts", content=b"{not json", headers={
            **json_type, "Authorization": "Bearer local-test-token"})
        assert allowed.status_code == 422 and bodies


@pytest.mark.parametrize("mounted", [False, True], ids=["root path", "mount"])
def test_the_token_is_checked_before_the_body_is_read_under_a_path_prefix(tmp_path, monkeypatch,
                                                                          mounted):
    """Behind ``uvicorn --root-path /memory``, or mounted at /memory, a write
    without the token still gets 401 before its body is read."""
    monkeypatch.setenv("MEMORYOS_API_TOKEN", "local-test-token")
    recording, bodies = recording_bodies(create_app(tmp_path))
    if mounted:
        served, options = Starlette(routes=[Mount("/memory", app=recording)]), {}
    else:
        served, options = recording, {"root_path": "/memory"}
    json_type = {"Content-Type": "application/json"}
    with TestClient(served, base_url="http://localhost", **options) as secured:
        refused = secured.post("/memory/api/facts", content=b"{not json", headers=json_type)
        assert refused.status_code == 401
        assert refused.json() == {"detail": "A valid bearer token is required"}
        assert refused.headers["www-authenticate"] == "Bearer"
        assert bodies == []
        assert secured.get("/memory/health").status_code == 200
        allowed = secured.post("/memory/api/facts", content=b"{not json", headers={
            **json_type, "Authorization": "Bearer local-test-token"})
        assert allowed.status_code == 422 and bodies


def test_tags_links_and_times_have_length_limits(client):
    """Each tag and link holds at most 500 characters, and each time string at most 64."""
    fact = {"entity": "Acme", "attribute": "status", "value": "open"}
    # Spaces around a time are allowed; each of these is exactly 64 characters.
    padded, later = "2026-01-01" + " " * 54, " " * 54 + "2026-02-01"
    accepted = client.post("/api/facts", json={**fact, "tags": ["t" * 500], "links": ["l" * 500],
                                               "valid_from": padded, "valid_to": later})
    assert accepted.status_code == 201, accepted.text
    for field, value, location in (("tags", ["t" * 501], ["body", "tags", 0]),
                                   ("links", ["ok", "l" * 501], ["body", "links", 1]),
                                   ("valid_from", padded + " ", ["body", "valid_from"]),
                                   ("valid_to", padded + " ", ["body", "valid_to"])):
        response = client.post("/api/facts", json={**fact, field: value})
        assert response.status_code == 422, field
        [error] = response.json()["detail"]
        assert (error["type"], error["loc"]) == ("string_too_long", location)
    extract = client.post("/api/proposals/extract", json={
        "text": "Acme | status | open", "source": "chat", "observed_at": padded + " "})
    assert extract.status_code == 422
    [error] = extract.json()["detail"]
    assert (error["type"], error["loc"]) == ("string_too_long", ["body", "observed_at"])
    stored = client.get("/api/facts", params={"as_of": "2026-01-15"}).json()["facts"]
    assert [fact["id"] for fact in stored] == [accepted.json()["id"]]


@pytest.mark.parametrize("route, params, name", [
    ("/api/facts", {}, "as_of"),
    ("/api/facts", {}, "known_at"),
    ("/api/search", {"q": "Acme"}, "as_of"),
    ("/api/search", {"q": "Acme"}, "known_at"),
    ("/api/history", {"entity": "Acme"}, "known_at"),
])
def test_time_query_parameters_have_the_body_length_limit(client, route, params, name):
    """A time in a query string, like one in a request body, holds at most 64 characters."""
    padded = "2026-01-01" + " " * 54
    assert client.get(route, params={**params, name: padded}).status_code == 200
    response = client.get(route, params={**params, name: padded + " "})
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert (error["type"], error["loc"]) == ("string_too_long", ["query", name])


def test_api_validation_and_missing_record(client):
    assert client.get("/api/facts/not-a-real-id").status_code == 404
    for response in (client.get("/api/facts/" + "0" * 32),
                     client.post("/api/facts/{}/corrections".format("0" * 32), json={"value": "x"}),
                     client.delete("/api/facts/" + "0" * 32)):
        assert (response.status_code, response.json()) == (
            404, {"detail": "Memory record not found"})
    assert client.get("/api/search", params={"q": "status", "limit": 0}).status_code == 422
    assert client.get("/api/facts", params={"as_of": "not-a-date"}).status_code == 422
    assert client.post(
        "/api/facts",
        json={"entity": "Acme", "attribute": "status", "value": "x", "recorded_at": "2000-01-01"},
    ).status_code == 422


def test_proposal_must_be_approved_before_memory_changes(client):
    response = client.post(
        "/api/proposals/extract",
        json={
            "text": "Acme application | status | pending",
            "source": "synthetic:conversation",
            "observed_at": "2026-09-01",
            "topic": "career/applications",
        },
    )
    assert response.status_code == 201, response.text
    proposal = response.json()["proposals"][0]
    assert client.get("/api/facts").json()["facts"] == []
    approved = client.post("/api/proposals/{}/approve".format(proposal["id"]))
    assert approved.status_code == 200, approved.text
    assert approved.json()["value"] == "pending"
    assert len(client.get("/api/facts").json()["facts"]) == 1


# Regressions fixed in 0.2.1


def test_every_documented_route_responds(client):
    first = add(client)
    assert [f["id"] for f in client.get("/api/facts").json()["facts"]] == [first["id"]]
    assert client.get("/api/facts/{}".format(first["id"])).json()["value"] == "pending"
    corrected = client.post("/api/facts/{}/corrections".format(first["id"]),
                            json={"value": "submitted", "source": "synthetic:test"})
    assert corrected.status_code == 201, corrected.text
    found = client.get("/api/search", params={"q": "Acme status", "mode": "keyword"})
    assert found.status_code == 200
    assert [hit["fact"]["value"] for hit in found.json()["results"]] == ["submitted"]
    timeline = client.get("/api/history", params={"entity": "Acme application"})
    assert [f["value"] for f in timeline.json()["facts"]] == ["submitted"]
    retracted = client.delete("/api/facts/{}".format(corrected.json()["id"]))
    assert retracted.status_code == 200 and retracted.json()["deleted"] is True
    extracted = client.post("/api/proposals/extract", json={
        "text": "Alex | city | Riverton\nAlex | drink | tea", "source": "synthetic:chat",
        "observed_at": "2026-01-01"})
    assert extracted.status_code == 201, extracted.text
    city, drink = (p["id"] for p in extracted.json()["proposals"])
    assert [p["id"] for p in client.get("/api/proposals").json()["proposals"]] == [city, drink]
    assert client.post("/api/proposals/{}/approve".format(city)).json()["value"] == "Riverton"
    assert client.post("/api/proposals/{}/reject".format(drink)).json()["status"] == "rejected"
    for status, expected in (("approved", [city]), ("rejected", [drink]), ("pending", [])):
        listed = client.get("/api/proposals", params={"status": status}).json()["proposals"]
        assert [p["id"] for p in listed] == expected
    assert client.post("/api/reindex").json() == {"indexed": 4}
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("base_url", ["http://evil.example", "http://localhost.evil.example",
                                      "http://127.0.0.1.nip.io:8000"])
def test_requests_for_a_foreign_host_are_rejected(tmp_path, base_url):
    with TestClient(create_app(tmp_path), base_url=base_url) as foreign:
        assert foreign.get("/api/facts").status_code == 400
        written = foreign.post("/api/facts", json={"entity": "x", "attribute": "y", "value": "z"})
        assert written.status_code == 400
    assert not list((tmp_path / "memory").rglob("*.md"))


@pytest.mark.parametrize("base_url", ["http://localhost", "http://localhost:8000",
                                      "http://127.0.0.1:8000", "http://[::1]:8000"])
def test_requests_for_localhost_are_accepted(tmp_path, base_url):
    with TestClient(create_app(tmp_path), base_url=base_url) as local:
        assert local.get("/api/facts").status_code == 200
    with TestClient(create_app(tmp_path), base_url="http://localhost") as local:
        assert local.get("/api/facts", headers={"Host": "evil.example"}).status_code == 400


def test_allowed_hosts_are_added_to_the_localhost_list(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYOS_ALLOWED_HOSTS",
                       " memory.internal , , OTHER.internal, *.example.com, 192.168.1.5, [fd00::1]")
    app = create_app(tmp_path)
    for host, status in (("memory.internal", 200), ("other.internal:8000", 200),
                         ("api.example.com", 200), ("a.b.example.com:8000", 200),
                         ("192.168.1.5:8000", 200), ("[fd00::1]:8000", 200),
                         # Setting the variable keeps local clients working.
                         ("localhost", 200), ("127.0.0.1:8000", 200), ("[::1]:8000", 200),
                         # The wildcard covers subdomains only.
                         ("example.com", 400), ("evilexample.com", 400),
                         ("example.com.evil.example", 400), ("evil.example", 400), ("", 400)):
        with TestClient(app, base_url="http://localhost") as test_client:
            response = test_client.get("/api/facts", headers={"Host": host})
            assert response.status_code == status, host
            if status == 400:
                assert response.json() == {"detail": "Invalid host header"}


@pytest.mark.parametrize("entry", ["*example.com", "memory.internal:8000", "*", "*.*.com",
                                   "http://memory.internal", "exa mple.com", "a..b"])
def test_invalid_allowed_host_entries_stop_create_app(tmp_path, monkeypatch, entry):
    monkeypatch.setenv("MEMORYOS_ALLOWED_HOSTS", "memory.internal," + entry)
    with pytest.raises(ValueError, match="MEMORYOS_ALLOWED_HOSTS has an invalid entry"):
        create_app(tmp_path)


def test_cross_site_writes_are_refused_but_local_ones_are_not(tmp_path):
    body = {"entity": "x", "attribute": "y", "value": "z"}
    with TestClient(create_app(tmp_path), base_url="http://localhost:8000") as test_client:
        for origin in ("http://evil.example", "null", "https://localhost.evil.example"):
            headers = {"Origin": origin}
            for response in (test_client.post("/api/facts", json=body, headers=headers),
                             test_client.post("/api/reindex", headers=headers),
                             test_client.delete("/api/facts/" + "0" * 32, headers=headers)):
                assert response.status_code == 403, origin
                assert response.json() == {"detail": "Cross-site requests are not accepted"}
            # Reads are not state changes; the host check still guards them.
            assert test_client.get("/api/facts", headers=headers).status_code == 200
        assert not list((tmp_path / "memory").rglob("*.md"))
        for headers in ({}, {"Origin": "http://localhost:8000"},
                        {"Origin": "http://127.0.0.1:8000"}, {"Origin": "http://[::1]:8000"}):
            assert test_client.post("/api/facts", json=body, headers=headers).status_code == 201
    assert len(list((tmp_path / "memory").rglob("*.md"))) == 4


def test_non_ascii_token_is_rejected_at_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYOS_API_TOKEN", "pässwort")
    with pytest.raises(ValueError, match="ASCII"):
        create_app(tmp_path)


def test_conflicting_approvals_return_409(client):
    old = client.post("/api/facts", json={"entity": "user", "attribute": "city", "value": "Paris",
                                          "valid_from": "2026-01-01"}).json()
    [stale] = client.post("/api/proposals/extract", json={
        "text": "user | city | Lyon | 2026-01-01", "source": "chat"}).json()["proposals"]
    client.post("/api/facts/{}/corrections".format(old["id"]), json={"value": "London"})
    response = client.post("/api/proposals/{}/approve".format(stale["id"]))
    assert response.status_code == 409, response.text
    assert "Memory changed" in response.json()["detail"]
    client.post("/api/proposals/{}/reject".format(stale["id"]))
    rejected = client.post("/api/proposals/{}/approve".format(stale["id"]))
    assert rejected.status_code == 409 and "Only pending" in rejected.json()["detail"]
    [fresh] = client.post("/api/proposals/extract", json={
        "text": "user | drink | tea | 2026-01-01", "source": "chat"}).json()["proposals"]
    assert client.post("/api/proposals/{}/approve".format(fresh["id"])).status_code == 200
    assert client.post("/api/proposals/{}/reject".format(fresh["id"])).status_code == 409


def test_unknown_and_malformed_proposal_ids(client):
    unknown = "f" * 32
    for action in ("approve", "reject"):
        response = client.post("/api/proposals/{}/{}".format(unknown, action))
        assert response.status_code == 404, response.text
        assert response.json() == {"detail": "Proposal not found"}
        malformed = client.post("/api/proposals/not-a-proposal/{}".format(action))
        assert malformed.status_code == 422 and "Invalid proposal ID" in malformed.text
    assert client.get("/api/facts").json()["facts"] == []


def test_damaged_records_are_server_errors_that_do_not_disclose_the_path(tmp_path, client):
    fact = add(client)
    [path] = (tmp_path / "memory").rglob("{}.md".format(fact["id"]))
    path.write_text(path.read_text(encoding="utf-8") + "\n<!-- memoryos:fact:v1 -->\n",
                    encoding="utf-8")
    # The request was fine; files on the server need repair.
    for route in ("/api/facts", "/api/search?q=status", "/api/history?entity=Acme"):
        response = client.get(route)
        assert response.status_code == 503, route
        detail = response.json()["detail"]
        assert detail.startswith("memory/career/applications/{}.md:".format(fact["id"]))
        assert str(tmp_path) not in response.text
        assert str(tmp_path.resolve()) not in response.text
    assert client.post("/api/reindex").status_code == 503
    path.write_text(path.read_text(encoding="utf-8").rsplit("\n<!--", 1)[0], encoding="utf-8")
    assert client.get("/api/facts").status_code == 200
    connection = sqlite3.connect(str(tmp_path / ".memoryos" / "index.sqlite3"))
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()
    newer = client.get("/api/facts")
    # An index from a newer release is a server-side condition, like other damage.
    assert newer.status_code == 503 and ".memoryos" in newer.json()["detail"]
    assert str(tmp_path.resolve()) not in newer.text


class HubError(Exception):
    """An exception type the API has no handler for, as model libraries raise."""


@pytest.mark.parametrize("failure", [HubError, OSError, FileNotFoundError, ValueError])
def test_unavailable_embedding_model_returns_503_without_its_path(tmp_path, monkeypatch,
                                                                  capsys, failure):
    secret = "/srv/private-models/minilm"

    class Failing:
        def __init__(self, name, trust_remote_code):
            raise failure("Path {} not found".format(name))

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=Failing))
    monkeypatch.setenv("MEMORYOS_EMBEDDING_MODEL", secret)
    with TestClient(create_app(tmp_path), base_url="http://localhost") as test_client:
        add(test_client)
        for mode in ("hybrid", "semantic"):
            response = test_client.get("/api/search", params={"q": "status", "mode": mode})
            assert response.status_code == 503, response.text
            assert secret not in response.text
            assert "embedding model is unavailable" in response.json()["detail"]
        keyword = test_client.get("/api/search", params={"q": "status", "mode": "keyword"})
        assert keyword.status_code == 200 and len(keyword.json()["results"]) == 1
    assert secret in capsys.readouterr().err  # The operator still sees the cause.


class BrokenInstallFinder:
    """An import hook failing the way a broken torch install does."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "sentence_transformers":
            raise OSError("dlopen(libtorch_cpu.dylib): Library not loaded")
        return None


def test_embedding_library_that_fails_to_import_returns_503(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BrokenInstallFinder(), *sys.meta_path])
    monkeypatch.setenv("MEMORYOS_EMBEDDING_MODEL", "local-model")
    with TestClient(create_app(tmp_path), base_url="http://localhost") as test_client:
        add(test_client)
        response = test_client.get("/api/search", params={"q": "status", "mode": "hybrid"})
        assert response.status_code == 503, response.text
        assert "embedding model is unavailable" in response.json()["detail"]


@pytest.mark.parametrize("configured", ["high", "2", "nan"])
def test_bad_semantic_threshold_stops_create_app(tmp_path, monkeypatch, configured):
    monkeypatch.setenv("MEMORYOS_SEMANTIC_THRESHOLD", configured)
    with pytest.raises(ValueError, match="MEMORYOS_SEMANTIC_THRESHOLD must be a number"):
        create_app(tmp_path)


def damage_index(root):
    (root / ".memoryos" / "index.sqlite3").write_bytes(b"this is not a database " * 200)


def test_damaged_index_returns_503_and_only_the_cli_repairs_it(tmp_path, client):
    fact = add(client)
    damage_index(tmp_path)
    for response in (client.get("/api/facts"), client.get("/api/facts/" + fact["id"]),
                     client.get("/api/search", params={"q": "status"}),
                     client.post("/api/reindex")):
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert "run `memoryos reindex`" in detail and str(tmp_path) not in detail
    # The server never renames files that other requests may have open.
    assert sorted(p.name for p in (tmp_path / ".memoryos").iterdir()) == ["index.sqlite3"]


def damage_pages_holding(root, text):
    """Overwrite the index pages that hold ``text``. The header and schema stay
    intact, so the store opens and the damage shows only when a query reads them."""
    index = root / ".memoryos" / "index.sqlite3"
    with closing(sqlite3.connect(str(index))) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = bytearray(index.read_bytes())
    size = int.from_bytes(raw[16:18], "big")
    pages = [start for start in range(size, len(raw), size)
             if text.encode() in raw[start:start + size]]
    assert pages
    for start in pages:
        raw[start:start + size] = b"\xab" * size
    index.write_bytes(bytes(raw))


def test_index_damage_found_by_a_query_returns_503_with_the_repair_hint(tmp_path, client,
                                                                        monkeypatch):
    """Damage that only a query reads gets the same 503 as damage found on opening."""
    fact = add(client, "pending-marker")
    damage_pages_holding(tmp_path, "pending-marker")
    for response in (client.get("/api/facts"), client.get("/api/facts/" + fact["id"]),
                     client.get("/api/search", params={"q": "status"}),
                     client.get("/api/history", params={"entity": "Acme application"}),
                     client.post("/api/facts", json={"entity": "Acme application",
                                                     "attribute": "status", "value": "x"}),
                     client.post("/api/reindex")):
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail.startswith("The index .memoryos/index.sqlite3 is damaged (")
        assert "run `memoryos reindex`" in detail and str(tmp_path) not in detail
    assert sorted(p.name for p in (tmp_path / ".memoryos").iterdir()) == ["index.sqlite3"]
    from temporal_memory_os.store import MemoryStore

    def locked(self, **_arguments):
        raise sqlite3.OperationalError("database is locked")

    # Other database errors are not damage and stay server errors.
    monkeypatch.setattr(MemoryStore, "snapshot", locked)
    with TestClient(create_app(tmp_path), base_url="http://localhost",
                    raise_server_exceptions=False) as test_client:
        response = test_client.get("/api/facts")
        assert response.status_code == 500 and "reindex" not in response.text


def replace_in_index(root, old, new):
    """Replace the only copy of ``old`` in the index file; every page stays valid."""
    index = root / ".memoryos" / "index.sqlite3"
    with closing(sqlite3.connect(str(index))) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = index.read_bytes()
    assert raw.count(old) == 1
    index.write_bytes(raw.replace(old, new))


@pytest.mark.parametrize("old, new, rebuilt", [
    # A column name in the schema: only the CLI may set the damaged file aside.
    (b" payload TEXT NOT NULL", b" payloax TEXT NOT NULL", False),
    # One record's stored JSON, still valid UTF-8: rebuilding in place replaces it.
    (b'"value": "pending-marker"', b'"value"; "pending-marker"', True),
], ids=["column name", "payload"])
def test_index_damage_that_passes_the_page_check_returns_503_not_422_or_500(tmp_path, client,
                                                                           old, new, rebuilt):
    fact = add(client, "pending-marker")
    replace_in_index(tmp_path, old, new)
    for response in (client.get("/api/facts"), client.get("/api/facts/" + fact["id"]),
                     client.get("/api/search", params={"q": "status"}),
                     client.get("/api/history", params={"entity": "Acme application"}),
                     client.post("/api/facts", json={"entity": "Acme application",
                                                     "attribute": "status", "value": "x"})):
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail.startswith("The index .memoryos/index.sqlite3 is damaged (")
        assert "run `memoryos reindex`" in detail and str(tmp_path) not in detail
    response = client.post("/api/reindex")
    assert response.status_code == (200 if rebuilt else 503), response.text
    assert sorted(p.name for p in (tmp_path / ".memoryos").iterdir()) == ["index.sqlite3"]
    if rebuilt:
        assert client.get("/api/facts/" + fact["id"]).json()["value"] == "pending-marker"


def test_proposal_listing_names_damaged_files_and_lists_the_rest(tmp_path, client):
    extracted = client.post("/api/proposals/extract", json={
        "text": "Alex | city | Riverton\nAlex | drink | tea", "source": "synthetic:chat",
        "observed_at": "2026-01-01"}).json()["proposals"]
    damaged, kept = (p["id"] for p in extracted)
    (tmp_path / "proposals" / (damaged + ".json")).write_bytes(b"{")
    listed = client.get("/api/proposals")
    assert listed.status_code == 200
    assert [p["id"] for p in listed.json()["proposals"]] == [kept]
    assert listed.json()["problems"] == [
        "proposals/{}.json is not valid JSON; fix or delete it".format(damaged)]
    approve = client.post("/api/proposals/{}/approve".format(damaged))
    assert approve.status_code == 422 and approve.json()["detail"].startswith("proposals/")
    assert client.get("/api/proposals", params={"status": "approved"}).json() == {
        "proposals": [], "problems": listed.json()["problems"]}


def test_file_not_found_is_not_reported_as_a_missing_record(tmp_path, monkeypatch):
    from temporal_memory_os.store import MemoryStore

    missing = str(tmp_path / "memory" / "general" / "lost.md")

    def vanished(self, **_arguments):
        raise FileNotFoundError(2, "No such file or directory", missing)

    monkeypatch.setattr(MemoryStore, "snapshot", vanished)
    with TestClient(create_app(tmp_path), base_url="http://localhost",
                    raise_server_exceptions=False) as test_client:
        fact = add(test_client)
        # A file system failure is a server error, not an answer about a record.
        response = test_client.get("/api/facts")
        assert response.status_code == 500
        assert "not found" not in response.text.lower()
        assert missing not in response.text
        # Records that really are missing still get a 404.
        assert test_client.get("/api/facts/{}".format("0" * 32)).status_code == 404
        assert test_client.get("/api/facts/{}".format(fact["id"])).status_code == 200


def test_one_embedder_and_vector_cache_serve_every_request(tmp_path, monkeypatch):
    import temporal_memory_os.embeddings as embeddings_module
    from temporal_memory_os.embeddings import HashingEmbedder

    created = []

    class Counting(HashingEmbedder):
        def __init__(self):
            super().__init__()
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return super().embed(texts)

    def factory(model_name=None):
        created.append(Counting())
        return created[-1]

    monkeypatch.setattr(embeddings_module, "create_embedder", factory)
    with TestClient(create_app(tmp_path), base_url="http://localhost") as test_client:
        add(test_client)
        add(test_client, "interview invited", "2026-09-15")
        for _ in range(3):
            response = test_client.get("/api/search", params={"q": "status", "mode": "semantic"})
            assert response.status_code == 200 and len(response.json()["results"]) == 1
    [embedder] = created
    # Documents are embedded on the first search only; later ones embed the query.
    assert [len(call) for call in embedder.calls] == [2, 1, 1]


def test_empty_memoryos_home_uses_the_default_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYOS_HOME", "")
    with TestClient(create_app(), base_url="http://localhost") as test_client:
        add(test_client)
    assert len(list((tmp_path / "memory-data" / "memory").rglob("*.md"))) == 1
    assert not (tmp_path / "memory").exists()
