"""Every interface says what a date alone means (0.3.1).

A date alone means 00:00 UTC, which is 08:00 at UTC+8. A fact added there at
15:00 is recorded at 07:00 UTC, so a cutoff of that day's date hides it and an
assistant answers with the older value.
"""

import json
import re

import pytest

from temporal_memory_os import MemoryStore, cli
from temporal_memory_os.agent import SYSTEM_PROMPT
from temporal_memory_os.models import normalize_time

ALONE = "A date alone means 00:00 UTC at the start of that day."
RULE = ("A date alone means 00:00 UTC at the start of that day, not the whole day and not "
        "local time. For the end of a local day, pass a timestamp with a UTC offset, such as "
        "2026-11-01T23:59:59.999999+08:00; leave the cutoff out to mean now.")
IN_URL = " In a URL, write + as %2B."
AS_OF = "When the facts were true. " + RULE
KNOWN_AT = "Use only facts recorded by this time. " + RULE


@pytest.fixture
def memory(tmp_path):
    """Tea since October; matcha added on 20 October at 15:00 UTC+8."""
    with MemoryStore(tmp_path) as store:
        store.add("user", "favourite drink", "tea", valid_from="2026-10-01",
                  recorded_at="2026-10-01")
        store.add("user", "favourite drink", "matcha", valid_from="2026-10-20T15:00+08:00",
                  recorded_at="2026-10-20T15:00+08:00")
    return tmp_path


def run_mcp(root, scenario):
    """Run ``scenario(session, initialize_result)`` against an in-memory MCP session."""
    pytest.importorskip("mcp")
    import anyio
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    from temporal_memory_os.mcp_server import create_server

    low_level = create_server(root)._mcp_server

    async def main():
        async with create_client_server_memory_streams() as (client, served):
            async with anyio.create_task_group() as group:
                group.start_soon(lambda: low_level.run(
                    served[0], served[1], low_level.create_initialization_options()))
                async with ClientSession(client[0], client[1]) as session:
                    await scenario(session, await session.initialize())
                group.cancel_scope.cancel()

    anyio.run(main)


def payload(result):
    assert not result.isError, result
    [content] = result.content
    return json.loads(content.text)


def test_mcp_instructions_and_date_parameters_state_the_rule(tmp_path):
    async def scenario(session, initialized):
        assert RULE in initialized.instructions
        tools = {tool.name: tool.inputSchema["properties"]
                 for tool in (await session.list_tools()).tools}
        assert tools["search"]["as_of"]["description"] == AS_OF
        assert tools["search"]["known_at"]["description"] == KNOWN_AT
        assert tools["memory_history"]["known_at"]["description"] == KNOWN_AT
        # Only the descriptions are new: the type and default are unchanged.
        for schema in (tools["search"]["as_of"], tools["search"]["known_at"],
                       tools["memory_history"]["known_at"]):
            assert schema["anyOf"] == [{"type": "string"}, {"type": "null"}]
            assert schema["default"] is None

    run_mcp(tmp_path, scenario)


def test_mcp_search_reports_the_utc_cutoffs_it_used(memory):
    async def search(session, **dates):
        return payload(await session.call_tool(
            "search", {"query": "favourite drink", "limit": 1, **dates}))

    async def scenario(session, _initialized):
        # The day's date means 00:00 UTC, before the matcha was added.
        found = await search(session, as_of="2026-10-20", known_at="2026-10-20")
        assert [hit["text"] for hit in found["results"]] == ["tea"]
        assert found["as_of"] == found["known_at"] == "2026-10-20"  # As given, as before.
        assert found["as_of_utc"] == found["known_at_utc"] == "2026-10-20T00:00:00.000000Z"
        # Both later times the rule names include it.
        for later in ("2026-10-21", "2026-10-20T23:59+08:00"):
            found = await search(session, as_of=later, known_at=later)
            assert [hit["text"] for hit in found["results"]] == ["matcha"], later
        assert found["as_of_utc"] == found["known_at_utc"] == "2026-10-20T15:59:00.000000Z"
        # Left out, a cutoff means now, and both share one now.
        before = normalize_time()
        found = await search(session)
        omitted_known = await search(session, as_of="2026-10-20T23:59+08:00")
        after = normalize_time()
        assert [hit["text"] for hit in found["results"]] == ["matcha"]
        assert found["as_of"] is None and found["known_at"] is None
        assert before <= found["as_of_utc"] == found["known_at_utc"] <= after
        assert omitted_known["as_of_utc"] == "2026-10-20T15:59:00.000000Z"
        assert omitted_known["known_at"] is None
        assert before <= omitted_known["known_at_utc"] <= after

    run_mcp(memory, scenario)


def test_mcp_history_reports_the_utc_cutoff_it_used(memory):
    async def history(session, **dates):
        return payload(await session.call_tool(
            "memory_history", {"entity": "user", "attribute": "favourite drink", **dates}))

    async def scenario(session, _initialized):
        found = await history(session, known_at="2026-10-20")
        assert [fact["value"] for fact in found["facts"]] == ["tea"]
        assert found["known_at"] == "2026-10-20"
        assert found["known_at_utc"] == "2026-10-20T00:00:00.000000Z"
        found = await history(session, known_at="2026-10-20T23:59+08:00")
        assert [fact["value"] for fact in found["facts"]] == ["tea", "matcha"]
        assert found["known_at_utc"] == "2026-10-20T15:59:00.000000Z"
        before = normalize_time()
        found = await history(session)
        after = normalize_time()
        assert [fact["value"] for fact in found["facts"]] == ["tea", "matcha"]
        assert found["known_at"] is None
        assert before <= found["known_at_utc"] <= after

    run_mcp(memory, scenario)


@pytest.mark.parametrize("tool, arguments, bad", [
    ("search", {"query": "drink", "as_of": "today"}, "today"),
    ("search", {"query": "drink", "known_at": "2026-10-20 3pm"}, "2026-10-20 3pm"),
    ("memory_history", {"entity": "user", "known_at": "20/10/2026"}, "20/10/2026"),
])
def test_mcp_an_unreadable_time_is_a_tool_error_that_quotes_it(memory, tool, arguments, bad):
    async def scenario(session, _initialized):
        result = await session.call_tool(tool, arguments)
        assert result.isError
        [content] = result.content
        assert "Time must be YYYY-MM-DD" in content.text
        assert repr(bad) in content.text
        assert str(memory) not in content.text

    run_mcp(memory, scenario)


def test_agent_prompt_says_to_copy_the_cutoffs_exactly():
    # The agent supplies full timestamps; shortening one to its date moves it
    # to midnight UTC. Leaving it out would unfreeze it, so that advice is omitted.
    assert ("Use the supplied as_of and known_at cutoffs in every search. "
            "Use known_at in history.\n"
            "Copy them exactly: a date alone means 00:00 UTC at the start of that day.\n"
            ) in SYSTEM_PROMPT


def test_http_api_description_and_time_parameters_state_the_rule(tmp_path):
    pytest.importorskip("fastapi")
    from temporal_memory_os.api import create_app

    schema = create_app(tmp_path).openapi()
    assert RULE + IN_URL in schema["info"]["description"]
    described = {(path, parameter["name"]): parameter.get("description")
                 for path in ("/api/facts", "/api/search", "/api/history")
                 for parameter in schema["paths"][path]["get"]["parameters"]
                 if parameter["name"] in ("as_of", "known_at")}
    assert described == {
        ("/api/facts", "as_of"): AS_OF + IN_URL, ("/api/facts", "known_at"): KNOWN_AT + IN_URL,
        ("/api/search", "as_of"): AS_OF + IN_URL, ("/api/search", "known_at"): KNOWN_AT + IN_URL,
        ("/api/history", "known_at"): KNOWN_AT + IN_URL,
    }


def test_http_api_whole_local_day_cutoff_works_with_plus_written_as_2b(memory):
    """The example in the rule, as a URL would carry it, finds the fact added that day."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from temporal_memory_os.api import create_app

    day_end = "2026-10-20T23:59:59.999999%2B08:00"
    with TestClient(create_app(memory), base_url="http://localhost") as client:
        whole_day = client.get("/api/facts?as_of={0}&known_at={0}".format(day_end))
        date_alone = client.get("/api/facts?as_of=2026-10-20&known_at=2026-10-20")
    assert [fact["value"] for fact in whole_day.json()["facts"]] == ["matcha"]
    assert [fact["value"] for fact in date_alone.json()["facts"]] == ["tea"]


@pytest.mark.parametrize("command, options", [
    ("search", {"--as-of AS_OF": AS_OF, "--known-at KNOWN_AT": KNOWN_AT}),
    ("history", {"--known-at KNOWN_AT": KNOWN_AT}),
    ("agent", {"--as-of AS_OF": AS_OF, "--known-at KNOWN_AT": KNOWN_AT}),
    ("add", {"--from VALID_FROM": "When the fact became true (default: now). " + ALONE,
             "--to VALID_TO": "When it stopped being true (default: still true). " + ALONE}),
    ("extract", {"--observed-at OBSERVED_AT": "When the text was written, the date of lines "
                                              "without one (default: now). " + ALONE}),
])
def test_cli_time_options_state_the_rule(capsys, command, options):
    with pytest.raises(SystemExit) as stopped:
        cli.main([command, "--help"])
    assert stopped.value.code == 0
    # argparse wraps help to the terminal width, and Python 3.14 colors it when
    # FORCE_COLOR is set; compare without colors or spacing.
    shown = "".join(re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out).split())
    for option, text in options.items():
        assert "".join(option.split()) + "".join(text.split()) in shown, option
