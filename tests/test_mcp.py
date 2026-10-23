"""Exercise the real MCP SDK over its official in-memory protocol transport."""

import asyncio
import json

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
