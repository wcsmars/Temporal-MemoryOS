"""Exercise the real MCP SDK over its official in-memory protocol transport."""

import asyncio
import json
import sqlite3
from contextlib import closing

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from temporal_memory_os.mcp_server import create_server  # noqa: E402
from temporal_memory_os.store import MemoryStore  # noqa: E402


def tool_payload(result):
    assert not result.isError, result
    if result.structuredContent is not None:
        return result.structuredContent
    text = [item.text for item in result.content if item.type == "text"]
    assert len(text) == 1
    return json.loads(text[0])


def test_mcp_read_only_temporal_search_fetch_and_history(tmp_path):
    with MemoryStore(tmp_path) as store:
        first = store.add(
            entity="Acme application",
            attribute="status",
            value="pending",
            valid_from="2026-09-01",
            recorded_at="2026-09-01",
            source="synthetic:application",
        )
        store.add(
            entity="Acme application",
            attribute="status",
            value="interview invited",
            valid_from="2026-09-15",
            recorded_at="2026-09-15",
            source="synthetic:invitation",
        )
        correction = store.correct(
            first.id,
            value="submitted",
            recorded_at="2026-10-01",
            source="synthetic:correction",
        )
        original_records = [fact.to_dict() for fact in store.all_facts()]

    async def scenario():
        server = create_server(tmp_path)
        async with create_connected_server_and_client_session(
            server, raise_exceptions=True
        ) as session:
            listing = await session.list_tools()
            assert {tool.name for tool in listing.tools} == {"search", "fetch", "memory_history"}
            for tool in listing.tools:
                assert tool.annotations.readOnlyHint is True
                assert tool.annotations.destructiveHint is False

            before_correction = tool_payload(await session.call_tool(
                "search",
                {
                    "query": "Acme status",
                    "as_of": "2026-09-10",
                    "known_at": "2026-09-20",
                },
            ))
            assert before_correction["results"][0]["id"] == first.id
            assert before_correction["results"][0]["text"] == "pending"

            after_correction = tool_payload(await session.call_tool(
                "search",
                {
                    "query": "Acme status",
                    "as_of": "2026-09-10",
                    "known_at": "2026-10-02",
                },
            ))
            assert after_correction["results"][0]["id"] == correction.id
            assert after_correction["results"][0]["text"] == "submitted"

            audit = tool_payload(await session.call_tool("fetch", {"id": first.id}))
            assert audit["url"] == "memoryos://fact/" + first.id
            assert json.loads(audit["text"])["value"] == "pending"
            assert audit["metadata"]["record_type"] == "audit"

            history = tool_payload(await session.call_tool(
                "memory_history",
                {"entity": "Acme application", "attribute": "status", "known_at": "2026-10-02"},
            ))
            assert [fact["value"] for fact in history["facts"]] == ["submitted", "interview invited"]

    asyncio.run(scenario())
    with MemoryStore(tmp_path) as store:
        assert [fact.to_dict() for fact in store.all_facts()] == original_records


def test_fetch_of_a_missing_record_names_it(tmp_path):
    missing = "0123456789abcdef0123456789abcdef"

    async def scenario():
        server = create_server(tmp_path)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("fetch", {"id": missing})
            assert result.isError
            [content] = result.content
            assert content.text.endswith("Memory record not found: " + missing)

    asyncio.run(scenario())


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


def test_index_damage_found_by_a_query_is_reported_with_the_repair_hint(tmp_path):
    """Damage that only a query reads gets the repair hint, not the raw database error."""
    with MemoryStore(tmp_path) as store:
        fact = store.add("person", "drink", "tea-marker", valid_from="2025-01-01")
    damage_pages_holding(tmp_path, "tea-marker")

    async def scenario():
        server = create_server(tmp_path)
        async with create_connected_server_and_client_session(server) as session:
            for name, arguments in (("search", {"query": "tea"}), ("fetch", {"id": fact.id}),
                                    ("memory_history", {"entity": "person"})):
                result = await session.call_tool(name, arguments)
                assert result.isError, name
                [content] = result.content
                assert "The index .memoryos/index.sqlite3 is damaged (" in content.text
                assert "run `memoryos reindex`" in content.text
                assert str(tmp_path) not in content.text

    asyncio.run(scenario())


@pytest.mark.parametrize("configured", ["high", "2", "nan"])
def test_bad_semantic_threshold_stops_the_server_at_start_up(tmp_path, monkeypatch, configured):
    monkeypatch.setenv("MEMORYOS_SEMANTIC_THRESHOLD", configured)
    with pytest.raises(ValueError, match="MEMORYOS_SEMANTIC_THRESHOLD must be a number"):
        create_server(tmp_path)
