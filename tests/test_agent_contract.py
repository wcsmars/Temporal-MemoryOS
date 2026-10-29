"""Exercise the installed SDK transport against an offline protocol peer.

The real SDK constructs CLI flags, starts and reaps a child, parses results and
handles cancellation. Only the model runtime executable is replaced; the peer
never opens a network connection or launches the configured MCP server.
"""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import sys

import pytest

from temporal_memory_os import MemoryStore
from temporal_memory_os.agent import MEMORY_TOOLS, ask


PEER = r'''
import json
import os
from pathlib import Path
import signal
import sys
import time

if "--version" in sys.argv or "-v" in sys.argv:
    print("2.1.248 (offline test peer)")
    sys.exit(0)

mode, directory = sys.argv[1:3]
directory = Path(directory)

def terminated(signum, frame):
    (directory / "terminated").touch()
    sys.exit(0)

signal.signal(signal.SIGTERM, terminated)
(directory / "started.json").write_text(json.dumps({
    "pid": os.getpid(), "cwd": os.getcwd(), "argv": sys.argv[3:]
}))
print("private-provider-diagnostic-sentinel", file=sys.stderr, flush=True)

for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "control_request":
        if message["request"]["subtype"] == "initialize":
            (directory / "initialize.json").write_text(json.dumps(message["request"]))
        print(json.dumps({"type": "control_response", "response": {
            "subtype": "success", "request_id": message["request_id"], "response": {}
        }}), flush=True)
    elif message["type"] == "user":
        (directory / "prompt.json").write_text(json.dumps(message))
        if mode == "process_error":
            sys.exit(1)
        if mode == "timeout":
            continue
        failed = mode == "result_error"
        print(json.dumps({
            "type": "result", "subtype": "error_max_turns" if failed else "success",
            "duration_ms": 1, "duration_api_ms": 0, "is_error": failed,
            "num_turns": 1, "session_id": "offline-peer", "total_cost_usd": 0,
            "result": "private-provider-result-sentinel" if failed else "Synthetic answer"
        }), flush=True)
(directory / "stdin_closed").touch()
if mode == "timeout":
    # One-shot SDK requests close stdin immediately. Model work still continues
    # after EOF; only transport teardown should terminate this stalled child.
    while True:
        time.sleep(1)
'''


@pytest.fixture
def sdk_peer(tmp_path, monkeypatch):
    pytest.importorskip("claude_agent_sdk")
    import claude_agent_sdk._internal.client as sdk_client

    memory = tmp_path / "memory-root"
    with MemoryStore(memory) as store:
        store.add("Acme", "status", "pending", valid_from="2026-01-01")
    script = tmp_path / "peer.py"
    script.write_text(PEER, encoding="utf-8")
    real_transport = sdk_client.SubprocessCLITransport

    def configure(mode):
        observed = tmp_path / mode
        observed.mkdir()
        launcher = observed / "agent-peer"
        command = [sys.executable, str(script), mode, str(observed)]
        launcher.write_text("#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n',
                            encoding="utf-8")
        launcher.chmod(0o700)

        def transport(*, prompt, options):
            return real_transport(prompt=prompt, options=replace(options, cli_path=str(launcher)))

        monkeypatch.setattr(sdk_client, "SubprocessCLITransport", transport)
        return memory, observed

    return configure


def assert_reaped(observed):
    started = json.loads((observed / "started.json").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(started["pid"], 0)
    assert not Path(started["cwd"]).exists()
    return started


def test_real_sdk_passes_only_the_configured_read_tools_and_no_settings(sdk_peer, capfd):
    memory, observed = sdk_peer("success")
    answer = asyncio.run(ask(memory, "Synthetic question", as_of="2026-01-01"))
    assert answer["answer"] == "Synthetic answer"
    assert answer["session_id"] == "offline-peer" and answer["total_cost_usd"] == 0
    started = assert_reaped(observed)
    argv = started["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--allowedTools") + 1].split(",") == MEMORY_TOOLS
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert json.loads((observed / "initialize.json").read_text())["skills"] == []
    assert {"--strict-mcp-config", "--setting-sources=", "--no-session-persistence"} <= set(argv)
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert list(config["mcpServers"]) == ["memory"]
    assert config["mcpServers"]["memory"]["args"][-1] == "mcp"
    prompt = json.loads((observed / "prompt.json").read_text())
    assert json.loads(prompt["message"]["content"])["as_of"] == answer["as_of"]
    assert (observed / "stdin_closed").exists()
    captured = capfd.readouterr()
    assert "private-provider" not in captured.out + captured.err


@pytest.mark.parametrize("mode", ["result_error", "process_error"])
def test_real_sdk_failure_is_sanitized_and_child_is_reaped(sdk_peer, capfd, mode):
    memory, observed = sdk_peer(mode)
    with pytest.raises(RuntimeError) as error:
        asyncio.run(ask(memory, "Synthetic question"))
    assert "private-provider" not in str(error.value)
    assert_reaped(observed)
    captured = capfd.readouterr()
    assert "private-provider" not in captured.out + captured.err


def test_real_sdk_timeout_closes_child_before_removing_workspace(sdk_peer, capfd):
    memory, observed = sdk_peer("timeout")
    with pytest.raises(RuntimeError, match="configured timeout"):
        asyncio.run(ask(memory, "Synthetic question", timeout=2))
    assert_reaped(observed)
    assert (observed / "stdin_closed").exists()
    assert (observed / "terminated").exists()
    captured = capfd.readouterr()
    assert "private-provider" not in captured.out + captured.err


@pytest.mark.parametrize("outcome", ["success", "runtime_error"])
def test_timeout_wins_when_sdk_swallows_cancellation(tmp_path, monkeypatch, outcome):
    sdk = pytest.importorskip("claude_agent_sdk")
    import anyio

    with MemoryStore(tmp_path):
        pass
    seen = {}

    async def query(*, prompt, options):
        seen["cwd"] = options.cwd
        try:
            await anyio.sleep_forever()
        except anyio.get_cancelled_exc_class():
            # SDK cleanup may turn cancellation into stream exhaustion, another
            # exception, or a buffered result. None makes the expired run succeed.
            if outcome == "runtime_error":
                raise RuntimeError("private-provider-response-sentinel")
            yield sdk.ResultMessage(
                subtype="success", duration_ms=1, duration_api_ms=0, is_error=False,
                num_turns=1, session_id="buffered-result", result="Late answer",
            )

    monkeypatch.setattr(sdk, "query", query)
    with pytest.raises(RuntimeError, match="configured timeout") as error:
        asyncio.run(ask(tmp_path, "Synthetic question", timeout=0.01))
    assert "private-provider" not in str(error.value)
    assert not Path(seen["cwd"]).exists()
