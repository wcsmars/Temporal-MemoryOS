"""Command-line checks, in child processes and in this process.

conftest.py clears every MEMORYOS_* variable before each test.
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

from temporal_memory_os import cli


def invoke(root, *args):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-m", "temporal_memory_os", "--root", str(root), *args],
        capture_output=True, text=True, env=environment, timeout=30,
    )


def test_demo_is_repeatable_and_historical_query_works(tmp_path):
    first = invoke(tmp_path, "demo")
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["added"] == 7
    second = invoke(tmp_path, "demo")
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["added"] == 0
    historical = invoke(tmp_path, "search", "Acme application status",
                        "--as-of", "2026-09-10", "--entity", "Acme application")
    assert historical.returncode == 0, historical.stderr
    result = json.loads(historical.stdout)
    assert result[0]["fact"]["value"] == "pending"
    assert result[0]["fact"]["source"].startswith("synthetic:")


def test_cli_invalid_input_fails_without_traceback(tmp_path):
    result = invoke(tmp_path, "add", "entity", "attribute", "value", "--from", "bad-date")
    assert result.returncode == 1
    assert "error" in json.loads(result.stderr)
    assert not result.stdout


def test_cli_extraction_requires_approval(tmp_path):
    result = invoke(tmp_path, "extract", "--text", "Alex | favorite color | blue",
                    "--source", "synthetic:test", "--observed-at", "2026-01-01")
    assert result.returncode == 0, result.stderr
    proposals = json.loads(result.stdout)
    assert len(proposals) == 1
    assert json.loads(invoke(tmp_path, "init").stdout)["facts"] == 0
    approval = invoke(tmp_path, "approve", proposals[0]["id"])
    assert approval.returncode == 0, approval.stderr
    assert json.loads(approval.stdout)["value"] == "blue"
    assert json.loads(invoke(tmp_path, "init").stdout)["facts"] == 1


# In-process regressions fixed in 0.2.1


def run(capsys, *arguments):
    """Call the CLI in this process; return (exit code, parsed stdout, raw stderr)."""
    code = cli.main(list(arguments))
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out.strip() else None, captured.err


def error_of(stderr):
    return json.loads(stderr)["error"]


def test_empty_memoryos_home_uses_memory_data_in_the_working_directory(tmp_path, monkeypatch,
                                                                         capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYOS_HOME", "")
    code, result, _ = run(capsys, "init")
    assert code == 0
    assert result["root"] == str((tmp_path / "memory-data").resolve())
    assert (tmp_path / "memory-data" / "memory").is_dir()
    assert not (tmp_path / "memory").exists()


def test_missing_ids_are_reported_plainly(tmp_path, capsys):
    root = str(tmp_path)
    for command in (["correct", "0" * 32, "x"], ["retract", "0" * 32]):
        code, output, error = run(capsys, "--root", root, *command)
        assert (code, output) == (1, None)
        assert error_of(error) == "Memory record not found: " + "0" * 32
    code, _, error = run(capsys, "--root", root, "approve", "f" * 32)
    assert code == 1 and error_of(error) == "Proposal not found: " + "f" * 32


def damage_index(root, pages):
    """Overwrite the index, or (``pages``) every page after its header and schema."""
    index = root / ".memoryos" / "index.sqlite3"
    raw = index.read_bytes()
    keep = int.from_bytes(raw[16:18], "big") if pages else 0
    index.write_bytes(raw[:keep] + b"\xab" * (len(raw) - keep))


@pytest.mark.parametrize("pages", [False, True], ids=["header", "pages"])
def test_damaged_index_is_reported_by_every_command_and_repaired_by_reindex(tmp_path, capsys,
                                                                           pages):
    root = str(tmp_path)
    assert run(capsys, "--root", root, "add", "person", "drink", "tea")[0] == 0
    damage_index(tmp_path, pages)
    for command in (["init"], ["search", "tea"], ["add", "person", "city", "Riverton"],
                    ["history", "person"], ["proposals"]):
        code, output, error = run(capsys, "--root", root, *command)
        assert (code, output) == (1, None), command
        message = error_of(error)
        assert message.startswith("The index .memoryos/index.sqlite3 is damaged (")
        assert "run `memoryos reindex`" in message and root not in message
        assert sorted(p.name for p in (tmp_path / ".memoryos").iterdir()) == ["index.sqlite3"]
    code, result, error = run(capsys, "--root", root, "reindex")
    assert code == 0, error
    assert result == {"indexed": 1}
    [aside] = [p.name for p in (tmp_path / ".memoryos").iterdir() if "corrupt" in p.name]
    assert re.fullmatch(r"index\.sqlite3\.corrupt-\d{8}T\d{6}", aside)
    code, history, _ = run(capsys, "--root", root, "history", "person")
    assert code == 0 and [fact["value"] for fact in history] == ["tea"]


def test_sqlite_and_import_errors_are_json_errors(tmp_path, capsys, monkeypatch):
    from temporal_memory_os.store import MemoryStore

    def locked(self):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(MemoryStore, "reindex", locked)
    code, output, error = run(capsys, "--root", str(tmp_path), "reindex")
    assert (code, output) == (1, None) and error_of(error) == "database is locked"
    monkeypatch.setitem(sys.modules, "temporal_memory_os.git_history", None)
    code, output, error = run(capsys, "--root", str(tmp_path), "git-log")
    assert (code, output) == (1, None) and "git_history" in error_of(error)


class HubError(Exception):
    """An exception type the CLI does not list, as model libraries raise."""


def test_failing_embedding_model_is_a_json_error(tmp_path, capsys, monkeypatch):
    class Failing:
        def __init__(self, name, trust_remote_code):
            raise HubError("Repository not found: " + name)

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=Failing))
    root = str(tmp_path)
    assert run(capsys, "--root", root, "add", "person", "drink", "tea")[0] == 0
    monkeypatch.setenv("MEMORYOS_EMBEDDING_MODEL", "org/missing-model")
    code, output, error = run(capsys, "--root", root, "search", "tea")
    assert (code, output) == (1, None)
    assert "Could not load the embedding model" in error_of(error)
    assert "Traceback" not in error
    # Keyword search never loads the model.
    code, hits, _ = run(capsys, "--root", root, "search", "tea", "--mode", "keyword")
    assert code == 0 and len(hits) == 1


def test_serve_on_all_interfaces_warns_without_a_token(tmp_path, capsys, monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.append((host, port)))
    root = str(tmp_path)
    code, output, error = run(capsys, "--root", root, "serve", "--host", "0.0.0.0",
                              "--port", "8123")
    assert (code, output, calls) == (0, None, [("0.0.0.0", 8123)])
    assert error.splitlines() == [
        "Warning: serving on 0.0.0.0 without MEMORYOS_API_TOKEN lets anyone who can reach this "
        "port read and change memory.",
        "Note: requests must name localhost as their host; add the names other clients use to "
        "MEMORYOS_ALLOWED_HOSTS. This check is not access control: only the token is."]
    monkeypatch.setenv("MEMORYOS_API_TOKEN", "local-test-token")
    monkeypatch.setenv("MEMORYOS_ALLOWED_HOSTS", "memory.internal")
    assert run(capsys, "--root", root, "serve", "--host", "0.0.0.0")[2] == ""
    monkeypatch.delenv("MEMORYOS_API_TOKEN")
    monkeypatch.delenv("MEMORYOS_ALLOWED_HOSTS")
    assert run(capsys, "--root", root, "serve")[2] == ""
    assert calls[1:] == [("0.0.0.0", 8000), ("127.0.0.1", 8000)]


def test_serve_without_the_api_extra_names_the_source_install(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    code, _, error = run(capsys, "--root", str(tmp_path), "serve")
    assert code == 1 and "pip install -e '.[api]'" in error_of(error)


def test_mcp_without_the_mcp_extra_names_the_source_install(tmp_path, capsys, monkeypatch):
    # Importing the SDK now fails, as it does when the extra is not installed.
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    code, output, error = run(capsys, "--root", str(tmp_path), "mcp")
    assert (code, output) == (1, None)
    assert "pip install -e '.[mcp]'" in error_of(error)
    assert "Traceback" not in error


def test_mcp_command_runs_the_server_on_the_resolved_root(tmp_path, capsys, monkeypatch):
    import temporal_memory_os.mcp_server as mcp_server

    started = []
    monkeypatch.setattr(mcp_server, "run_server", started.append)
    monkeypatch.chdir(tmp_path)
    code, output, error = run(capsys, "--root", "store", "mcp")
    assert (code, output, error) == (0, None, "")
    assert started == [(tmp_path / "store").resolve()]


def test_demo_after_a_conflicting_manual_add_reports_skipped_slots(tmp_path, capsys):
    root = str(tmp_path)
    assert run(capsys, "--root", root, "add", "Alex", "preferred drink", "water",
               "--from", "2026-08-01")[0] == 0
    for added in (6, 0):
        code, result, error = run(capsys, "--root", root, "demo")
        assert code == 0, error
        assert result["added"] == added
        assert result["skipped_existing"] == ["Alex / preferred drink at 2026-08-01"]
    code, history, _ = run(capsys, "--root", root, "history", "Alex",
                           "--attribute", "preferred drink")
    assert [fact["value"] for fact in history] == ["water", "decaf coffee"]


def test_extract_file_with_a_byte_order_mark(tmp_path, capsys):
    conversation = tmp_path / "conversation.txt"
    conversation.write_bytes(b"\xef\xbb\xbfuser | city | Paris | 2026-09-01\nMy drink is tea.\n")
    code, proposals, error = run(capsys, "--root", str(tmp_path / "store"), "extract",
                                 "--file", str(conversation), "--source", "synthetic:file",
                                 "--observed-at", "2026-10-01")
    assert code == 0, error
    assert [(p["entity"], p["attribute"]) for p in proposals] == [("user", "city"),
                                                                  ("user", "drink")]


def test_extract_file_holding_only_a_byte_order_mark_is_empty(tmp_path, capsys):
    conversation = tmp_path / "conversation.txt"
    conversation.write_bytes(b"\xef\xbb\xbf\n")
    code, output, error = run(capsys, "--root", str(tmp_path / "store"), "extract",
                              "--file", str(conversation), "--source", "synthetic:file")
    assert (code, output) == (1, None)
    assert error_of(error) == "Conversation text must not be empty"


def test_extract_file_byte_order_mark_does_not_count_toward_the_size_limit(tmp_path, capsys):
    # Exactly the 1,000,000-character limit once the byte order mark is removed.
    line = "user | city | Paris | 2026-09-01\n"
    conversation = tmp_path / "conversation.txt"
    conversation.write_bytes(b"\xef\xbb\xbf" + (line + "\n" * (1_000_000 - len(line))).encode())
    code, proposals, error = run(capsys, "--root", str(tmp_path / "store"), "extract",
                                 "--file", str(conversation), "--source", "synthetic:file")
    assert code == 0, error
    assert [(p["entity"], p["value"]) for p in proposals] == [("user", "Paris")]
    conversation.write_bytes(conversation.read_bytes() + b"\n")  # One character over.
    code, _, error = run(capsys, "--root", str(tmp_path / "store"), "extract",
                         "--file", str(conversation), "--source", "synthetic:file")
    assert code == 1 and "1 MB extraction limit" in error_of(error)


def test_malformed_proposal_file_is_a_warning_when_listing_and_an_error_when_used(tmp_path,
                                                                                capsys):
    root = str(tmp_path)
    code, proposals, _ = run(capsys, "--root", root, "extract", "--text",
                             "Alex | city | Riverton\nAlex | drink | tea",
                             "--source", "synthetic:chat", "--observed-at", "2026-01-01")
    assert code == 0 and len(proposals) == 2
    damaged, kept = proposals[0]["id"], proposals[1]["id"]
    path = tmp_path / "proposals" / "{}.json".format(damaged)
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "created_at": None}), encoding="utf-8")
    warning = {"warning": "proposals/{}.json has an inconsistent id, status, kind or field "
                          "type; fix or delete it".format(damaged)}
    for status in ("pending", "all"):
        code, listed, error = run(capsys, "--root", root, "proposals", "--status", status)
        assert code == 0 and [p["id"] for p in listed] == [kept]
        assert [json.loads(line) for line in error.splitlines()] == [warning]
    for command in ("approve", "reject"):
        code, output, error = run(capsys, "--root", root, command, damaged)
        assert (code, output) == (1, None)
        assert error_of(error) == warning["warning"]
    code, approved, _ = run(capsys, "--root", root, "approve", kept)
    assert code == 0 and approved["value"] == "tea"
    path.unlink()
    assert run(capsys, "--root", root, "proposals")[2] == ""


def test_cli_errors_do_not_disclose_the_absolute_path(tmp_path, capsys):
    root = tmp_path / "store"
    _, fact, _ = run(capsys, "--root", str(root), "add", "person", "drink", "tea")
    [path] = (root / "memory").rglob("*.md")
    path.write_bytes(b"")
    code, _, error = run(capsys, "--root", str(root), "search", "tea")
    assert code == 1
    assert error_of(error).startswith("memory/general/{}.md:".format(fact["id"]))
    assert str(root) not in error and str(root.resolve()) not in error
