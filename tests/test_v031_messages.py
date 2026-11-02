"""0.3.1: messages that point at the right fix.

A damaged index names a repair that rebuilds this memory directory, not the
default one, and a typo in a hand-edited record is reported at the line an
editor shows, with its character column, whatever the file's line endings.
"""

import asyncio
import json
import re
import shlex
import sqlite3
from contextlib import closing

import pytest
from helpers import hand_written_record

from temporal_memory_os import cli
from temporal_memory_os.store import MemoryStore, StoreIntegrityError

REPAIR = "`memoryos --root <memory directory> reindex`"


def run(capsys, *arguments):
    """Call the CLI in this process; return (exit code, parsed stdout, raw stderr)."""
    code = cli.main(list(arguments))
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out.strip() else None, captured.err


def error_of(stderr):
    return json.loads(stderr)["error"]


def damage_index(root):
    (root / ".memoryos" / "index.sqlite3").write_bytes(b"this is not a database " * 200)


# The repair hint names this memory directory.


def test_the_damage_message_says_to_reindex_this_memory_directory(tmp_path):
    with MemoryStore(tmp_path) as store:
        store.add("person", "drink", "tea", valid_from="2025-01-01")
    damage_index(tmp_path)
    with pytest.raises(StoreIntegrityError) as caught:
        MemoryStore(tmp_path)
    message = str(caught.value)
    assert message.startswith("The index .memoryos/index.sqlite3 is damaged (file is not a "
                              "database). Stop other programs using this memory directory, then ")
    assert "reindex with --root set to this memory directory" in message
    assert "(the same --root or MEMORYOS_HOME used to open it)" in message
    assert message.endswith("for example " + REPAIR + ".")
    # Run as written, a bare reindex rebuilds ./memory-data or MEMORYOS_HOME instead.
    assert "`memoryos reindex`" not in message
    assert str(tmp_path) not in message


def test_the_hinted_command_repairs_this_memory_from_any_working_directory(tmp_path, capsys,
                                                                          monkeypatch):
    root, elsewhere = tmp_path / "my memory", tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert run(capsys, "--root", str(root), "init")[0] == 0
    assert run(capsys, "--root", str(root), "add", "person", "drink", "tea")[0] == 0
    damage_index(root)
    code, output, error = run(capsys, "--root", str(root), "history", "person")
    assert (code, output) == (1, None)
    message = error_of(error)
    assert str(root) not in message
    [command] = re.findall(r"`([^`]*)`", message)
    # The placeholder filled in as a person would, quoted for the shell.
    argv = shlex.split(command.replace("<memory directory>", shlex.quote(str(root))))
    assert argv[0] == "memoryos"
    code, result, error = run(capsys, *argv[1:])
    assert (code, result) == (0, {"indexed": 1}), error
    assert [p.name for p in (root / ".memoryos").iterdir() if ".corrupt-" in p.name]
    code, history, _ = run(capsys, "--root", str(root), "history", "person")
    assert code == 0 and [fact["value"] for fact in history] == ["tea"]
    assert list(elsewhere.iterdir()) == []  # No other directory was created or rebuilt.



# A column the engine reads that is missing from the index is damage too. SQLite
# names the column as the engine's query spells it, inside the repair message.


def damage_message(cause):
    return ("The index .memoryos/index.sqlite3 is damaged ({}). Stop other programs using "
            "this memory directory, then rebuild the index from the Markdown records by "
            "running reindex with --root set to this memory directory (the same --root or "
            "MEMORYOS_HOME used to open it), for example {}.".format(cause, REPAIR))


def replace_in_index(root, old, new):
    """Replace the only copy of ``old`` in the index file; every page stays valid."""
    index = root / ".memoryos" / "index.sqlite3"
    with closing(sqlite3.connect(str(index))) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # Everything in the main file.
    raw = index.read_bytes()
    assert raw.count(old) == 1
    index.write_bytes(raw.replace(old, new))


def rename_index_column(root, column):
    """Another program renames a column; SQLite rewrites the index definitions with it."""
    with closing(sqlite3.connect(str(root / ".memoryos" / "index.sqlite3"))) as connection:
        connection.execute("ALTER TABLE facts RENAME COLUMN {0} TO {0}_renamed".format(column))
        connection.commit()


@pytest.mark.parametrize("column", ["payload", "recorded_at", "valid_from"])
def test_history_names_an_index_column_renamed_by_another_program(tmp_path, capsys, column):
    """A column renamed outside the engine (with a SQLite tool, say) stops history with the
    whole repair message, and the hinted reindex repairs the index."""
    root = str(tmp_path)
    assert run(capsys, "--root", root, "init")[0] == 0
    assert run(capsys, "--root", root, "add", "person", "drink", "tea")[0] == 0
    rename_index_column(tmp_path, column)
    code, output, error = run(capsys, "--root", root, "history", "person")
    assert (code, output) == (1, None)
    assert error_of(error) == damage_message("no such column: " + column)
    code, result, error = run(capsys, "--root", root, "reindex")
    assert (code, result) == (0, {"indexed": 1}), error
    code, history, _ = run(capsys, "--root", root, "history", "person")
    assert code == 0 and [fact["value"] for fact in history] == ["tea"]


@pytest.mark.parametrize("damage, column", [
    (lambda root: replace_in_index(root, b" payload TEXT NOT NULL", b" payloax TEXT NOT NULL"),
     "payload"),
    (lambda root: rename_index_column(root, "recorded_at"), "recorded_at"),
], ids=["damaged column name", "renamed column"])
def test_approval_and_rejection_name_a_missing_index_column_and_the_repair(tmp_path, capsys,
                                                                          damage, column):
    """Approving or rejecting a proposal first reads every record of its names. If the
    index lacks a column there, both stop with the repair message, and after the hinted
    reindex the approval goes through."""
    root = str(tmp_path)
    assert run(capsys, "--root", root, "init")[0] == 0
    assert run(capsys, "--root", root, "add", "Alex", "city", "Riverton",
               "--from", "2025-01-01")[0] == 0
    code, proposals, _ = run(capsys, "--root", root, "extract", "--text",
                             "2025-06-01: Alex | city | Lakeside", "--source", "synthetic:chat")
    assert code == 0
    [proposal] = proposals
    damage(tmp_path)
    for command in ("reject", "approve"):
        code, output, error = run(capsys, "--root", root, command, proposal["id"])
        assert (code, output) == (1, None)
        assert error_of(error) == damage_message("no such column: " + column)
    assert run(capsys, "--root", root, "reindex")[:2] == (0, {"indexed": 1})
    code, approved, _ = run(capsys, "--root", root, "approve", proposal["id"])
    assert code == 0 and approved["value"] == "Lakeside"

# A JSON typo is reported where an editor shows it.

ENDINGS = pytest.mark.parametrize("encode", [
    lambda text: text.encode("utf-8"),
    lambda text: text.replace("\n", "\r\n").encode("utf-8"),
    lambda text: text.replace("\n", "\r").encode("utf-8"),
    lambda text: b"\xef\xbb\xbf" + text.encode("utf-8"),
    lambda text: b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"),
    lambda text: b"\xef\xbb\xbf" + text.replace("\n", "\r").encode("utf-8"),
], ids=["lf", "crlf", "cr", "bom", "bom-crlf", "bom-cr"])

# Each typo puts a ";" where JSON stops reading.
TYPOS = pytest.mark.parametrize("old, new, cause", [
    ('"value": "tea"', '"value"; "tea"', "Expecting ':' delimiter"),
    ("```json\n{\n", "```json\n{;\n", "Expecting property name enclosed in double quotes"),
    ("false\n}\n```", "false\n;\n```", "Expecting ',' delimiter"),
], ids=["middle", "first line", "last line"])


def where_an_editor_shows(raw):
    """Line and column of the only ";" in ``raw``; CRLF, CR and LF each end a line."""
    lines = re.split(rb"\r\n|\r|\n", raw)
    [(line, column)] = [(number, text.index(b";") + 1)
                        for number, text in enumerate(lines, 1) if b";" in text]
    return line, column


@ENDINGS
@TYPOS
def test_a_json_typo_is_reported_at_its_line_and_column_in_the_file(tmp_path, encode, old,
                                                                     new, cause):
    # Starts with an empty line, which counts like any other.
    text = "\n# person · drink\n\nA note above the record, written by hand.\n\n" + (
        hand_written_record())
    assert text.count(old) == 1 and ";" not in text
    raw = encode(text.replace(old, new))
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "hand.md").write_bytes(raw)
    line, column = where_an_editor_shows(raw)
    assert line > 6  # The premise: the JSON block starts several lines into the file.
    with pytest.raises(StoreIntegrityError) as caught:
        MemoryStore(tmp_path)
    assert str(caught.value) == "memory/hand.md: {}: line {} column {}".format(
        cause, line, column)


def test_the_cli_reports_a_typo_at_the_line_an_editor_shows(tmp_path, capsys):
    root = str(tmp_path)
    code, fact, _ = run(capsys, "--root", root, "add", "person", "drink", "tea")
    assert code == 0
    path = tmp_path / "memory" / "general" / (fact["id"] + ".md")
    text = path.read_text(encoding="utf-8").replace('"value": "tea"', '"value"; "tea"')
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))  # Saved by a Windows editor.
    assert where_an_editor_shows(path.read_bytes()) == (11, 10)
    code, output, error = run(capsys, "--root", root, "search", "tea")
    assert (code, output) == (1, None)
    assert error_of(error) == ("memory/general/{}.md: Expecting ':' delimiter: line 11 "
                               "column 10".format(fact["id"]))


def test_the_mcp_tools_report_a_typo_at_the_line_an_editor_shows(tmp_path):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    from temporal_memory_os.mcp_server import create_server

    with MemoryStore(tmp_path) as store:
        fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
        path = store.path_for(fact.id)
    path.write_text(path.read_text(encoding="utf-8").replace('"value": "tea"', '"value"; "tea"'),
                    encoding="utf-8")
    assert where_an_editor_shows(path.read_bytes()) == (11, 10)

    async def scenario():
        server = create_server(tmp_path)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("search", {"query": "tea"})
            assert result.isError
            [content] = result.content
            assert content.text.endswith("memory/general/{}.md: Expecting ':' delimiter: "
                                         "line 11 column 10".format(fact.id))
            assert str(tmp_path) not in content.text

    asyncio.run(scenario())
