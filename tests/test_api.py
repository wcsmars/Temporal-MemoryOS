"""HTTP integration checks; optional dependencies are not needed by core tests."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from temporal_memory_os.api import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORYOS_API_TOKEN", raising=False)
    with TestClient(create_app(tmp_path)) as test_client:
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
    with TestClient(create_app(tmp_path)) as secured:
        assert secured.get("/health").status_code == 200
        assert secured.get("/api/facts").status_code == 401
        assert secured.post("/api/reindex").status_code == 401
        assert secured.get("/api/facts", headers={"Authorization": "Bearer wrong"}).status_code == 401
        allowed = secured.get(
            "/api/facts", headers={"Authorization": "Bearer local-test-token"}
        )
        assert allowed.status_code == 200


def test_api_validation_and_missing_record(client):
    assert client.get("/api/facts/not-a-real-id").status_code == 404
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
