"""A missing memory directory is reported, never silently replaced (0.3.1).

A memory folder can be moved or renamed while a shell alias, an MCP client or an
HTTP server still names the old path. Every command that reads or writes
memory reports it, add and extract included; init, demo and git-init create it;
the HTTP and MCP servers never create it. A new directory is private (0700).

conftest.py clears every MEMORYOS_* variable before each test.
"""

import asyncio
import errno
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from temporal_memory_os import cli
from temporal_memory_os import store as store_module
from temporal_memory_os.store import (
    MemoryStore, MissingMemoryDirectoryError, StoreIntegrityError, _create_root,
)

MISSING = ("Memory directory is missing (moved, renamed or deleted?); move it back, "
           "or run memoryos init with the same --root or MEMORYOS_HOME to start a new, "
           "empty memory")
LOOP = "[Errno {}] Cannot open the memory directory: {}".format(errno.ELOOP,
                                                               os.strerror(errno.ELOOP))


@pytest.fixture(autouse=True)
def no_agent_sdk(monkeypatch):
    """The agent checks the folder before it starts the SDK. If a check is ever lost,
    the import fails here instead of starting a real agent process."""
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)


@pytest.fixture
def usual_umask():
    """The common 022 umask, under which a plain mkdir makes a world-readable folder."""
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def run(capsys, *arguments):
    """Call the CLI in this process; return (exit code, parsed stdout, raw stderr)."""
    code = cli.main(list(arguments))
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out.strip() else None, captured.err


def lines(stderr):
    return [json.loads(line) for line in stderr.splitlines()]


def moved_memory(tmp_path):
    """A memory with one fact whose folder was then renamed; return (old path, new path, fact)."""
    root, moved = tmp_path / "TemporalMemory", tmp_path / "Moved"
    with MemoryStore(root) as store:
        fact = store.add("user", "snack", "popcorn", valid_from="2026-11-01")
    root.rename(moved)
    return root, moved, fact


# Python API ---------------------------------------------------------------


def test_create_false_reports_a_missing_directory_and_creates_nothing(tmp_path):
    with pytest.raises(MissingMemoryDirectoryError) as caught:
        MemoryStore(tmp_path / "moved", create=False)
    # A StoreIntegrityError: the request was fine, the files need attention (HTTP 503).
    assert isinstance(caught.value, StoreIntegrityError)
    assert str(caught.value) == MISSING
    assert list(tmp_path.iterdir()) == []


def test_a_new_directory_is_private_and_reported(tmp_path, usual_umask):
    root = tmp_path / "memory-data" / "personal"
    with MemoryStore(root) as store:
        assert store.created is True
        store.add("person", "drink", "tea", valid_from="2025-01-01")
    assert mode(root) == 0o700
    assert mode(root.parent) == 0o755  # Missing parents are made as by mkdir -p.
    with MemoryStore(root, create=False) as store:
        assert store.created is False
        assert [fact.value for fact in store.all_facts()] == ["tea"]


def test_an_existing_empty_folder_is_an_empty_memory_and_keeps_its_mode(tmp_path, usual_umask):
    root = tmp_path / "made-by-hand"
    root.mkdir(mode=0o755)
    with MemoryStore(root, create=False) as store:
        assert (store.created, store.all_facts()) == (False, [])
    assert mode(root) == 0o755  # Never tightened behind the owner's back.
    assert (root / "memory").is_dir() and (root / ".memoryos" / "index.sqlite3").is_file()


@pytest.mark.parametrize("create", [True, False])
def test_a_file_in_place_of_the_directory_is_refused_without_its_path(tmp_path, create):
    root = tmp_path / "notes.txt"
    root.write_text("not a folder", encoding="utf-8")
    with pytest.raises(StoreIntegrityError) as caught:
        MemoryStore(root, create=create)
    assert not isinstance(caught.value, MissingMemoryDirectoryError)
    assert str(caught.value) == "The memory directory path exists but is not a folder"
    assert root.read_text(encoding="utf-8") == "not a folder"


def test_a_directory_another_process_created_first_is_not_claimed(tmp_path):
    root = tmp_path / "raced"
    assert _create_root(root) is True
    assert _create_root(root) is False


@pytest.mark.parametrize("looks", [1, 2, 3])
def test_a_new_store_another_process_creates_while_this_one_looks_opens(
        tmp_path, monkeypatch, looks):
    """Another process can make the folder just after this one looked and found none.
    Every way of looking ends in os.stat (on Python 3.10, through pathlib's own
    reference to it, patched too), so making it after the first, second or
    third look that finds none covers any check; a check that looked twice (is_dir,
    then exists) used to call the new folder "not a folder"."""
    root = tmp_path / "raced"
    names = {str(root), str(root.resolve())}
    real_stat, missed, made = os.stat, [], []

    def stat_then_other_process(path, *args, **kwargs):
        try:
            return real_stat(path, *args, **kwargs)
        except FileNotFoundError:
            if os.fspath(path) in names:
                missed.append(path)
                if len(missed) == looks:
                    os.mkdir(root, 0o700)  # The other process wins the race here.
                    made.append(root)
            raise

    monkeypatch.setattr(os, "stat", stat_then_other_process)
    accessor = getattr(pathlib, "_NormalAccessor", None)
    if accessor is not None and "stat" in vars(accessor):
        # Python 3.10: Path methods call the os.stat that pathlib bound at import.
        monkeypatch.setattr(accessor, "stat", staticmethod(stat_then_other_process))
    with MemoryStore(root) as store:
        store.add("user", "city", "Paris", valid_from="2026-01-01")
        assert store.created is not bool(made)


def test_a_new_store_another_process_creates_just_before_mkdir_opens(tmp_path, monkeypatch):
    """The other window: after the check found nothing, before this process's mkdir."""
    root = tmp_path / "raced"
    create = store_module._create_root

    def other_process_first(path):
        path.mkdir(mode=0o700)  # The other process wins the race here.
        return create(path)

    monkeypatch.setattr(store_module, "_create_root", other_process_first)
    with MemoryStore(root) as store:
        store.add("user", "city", "Paris", valid_from="2026-01-01")
        assert store.created is False


def test_a_file_another_process_put_there_first_is_not_claimed(tmp_path):
    root = tmp_path / "raced"
    root.write_text("not a folder", encoding="utf-8")
    with pytest.raises(StoreIntegrityError) as caught:
        _create_root(root)
    assert str(caught.value) == "The memory directory path exists but is not a folder"
    assert root.read_text(encoding="utf-8") == "not a folder"


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser can read any folder")
def test_a_folder_that_cannot_be_reached_is_not_called_missing(tmp_path, capsys):
    """Permission problems are reported as such, without the path: following the
    "missing" advice would start a new, empty memory beside the real one."""
    hidden = tmp_path / "hidden"
    with MemoryStore(hidden / "memory-data") as store:
        store.add("user", "snack", "popcorn", valid_from="2026-11-01")
    hidden.chmod(0o600)  # Its entries can no longer be looked up.
    try:
        for command in (["search", "popcorn"], ["git-init"], ["git-log"],
                        ["agent", "What snack?"]):
            code, output, error = run(capsys, "--root", str(hidden / "memory-data"), *command)
            assert (code, output) == (1, None), command
            assert lines(error) == [{"error": "[Errno 13] Cannot open the memory directory: "
                                              "Permission denied"}], command
        with pytest.raises(PermissionError) as caught:
            MemoryStore(hidden / "memory-data", create=True)
    finally:
        hidden.chmod(0o700)
    assert str(tmp_path) not in str(caught.value)
    assert sorted(path.name for path in hidden.iterdir()) == ["memory-data"]


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser can read any folder")
def test_the_agent_reports_a_folder_it_cannot_search_without_its_path(tmp_path, capsys):
    """The folder is there, but memory/ inside it cannot be looked up."""
    root = tmp_path / "TemporalMemory"
    with MemoryStore(root) as store:
        store.add("user", "snack", "popcorn", valid_from="2026-11-01")
    root.chmod(0o600)
    try:
        code, output, error = run(capsys, "--root", str(root), "agent", "What snack?")
    finally:
        root.chmod(0o700)
    assert (code, output) == (1, None)
    assert lines(error) == [{"error": "[Errno 13] Cannot open the memory directory: "
                                      "Permission denied"}]


def test_a_path_through_a_file_is_a_missing_directory(tmp_path, capsys):
    """A folder above it was replaced by a file, such as an alias left by a move."""
    (tmp_path / "Documents").write_text("not a folder", encoding="utf-8")
    root = tmp_path / "Documents" / "TemporalMemory"
    with pytest.raises(MissingMemoryDirectoryError):
        MemoryStore(root, create=False)
    code, output, error = run(capsys, "--root", str(root), "search", "popcorn")
    assert (code, output, lines(error)) == (1, None, [{"error": MISSING}])


@pytest.mark.parametrize("name", ["memory", ".memoryos"])
def test_a_file_in_place_of_a_store_folder_is_refused_without_its_path(tmp_path, capsys, name):
    root = tmp_path / "TemporalMemory"
    root.mkdir()
    (root / name).write_text("not a folder", encoding="utf-8")
    code, output, error = run(capsys, "--root", str(root), "history", "user")
    assert (code, output) == (1, None)
    assert lines(error) == [{"error": name + "/ in the memory directory is not a folder"}]


@pytest.mark.parametrize("command", [["proposals"],
                                     ["extract", "--text", "user | drink | tea", "--source", "chat"]])
def test_a_file_in_place_of_the_proposals_folder_is_refused_without_its_path(tmp_path, capsys,
                                                                             command):
    root = tmp_path / "TemporalMemory"
    with MemoryStore(root):
        pass
    (root / "proposals").write_text("not a folder", encoding="utf-8")
    code, output, error = run(capsys, "--root", str(root), *command)
    assert (code, output) == (1, None)
    assert lines(error) == [{"error": "proposals/ in the memory directory is not a folder"}]


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser can write to any folder")
def test_a_read_only_memory_without_a_proposals_folder_is_refused_without_its_path(tmp_path,
                                                                                   capsys):
    root = tmp_path / "TemporalMemory"
    with MemoryStore(root) as store:
        store.add("user", "snack", "popcorn", valid_from="2026-11-01")
    root.chmod(0o500)
    try:
        code, output, error = run(capsys, "--root", str(root), "proposals")
    finally:
        root.chmod(0o700)
    assert (code, output) == (1, None)
    assert lines(error) == [{"error": "[Errno 13] Cannot open the memory directory: "
                                      "Permission denied"}]
    assert not (root / "proposals").exists()


def test_a_root_that_is_a_loop_of_links_is_reported_without_its_path(tmp_path, capsys):
    """Python 3.10 to 3.12 name the path in their own error for a loop; 3.13 and later
    leave the loop to the folder check. Either way the message is the same."""
    link = tmp_path / "memory-link"
    os.symlink("memory-link", link)  # A typo such as: ln -sfn memory-link memory-link
    for command in (["search", "popcorn"], ["init"], ["git-log"], ["git-init"],
                    ["agent", "What snack?"]):
        code, output, error = run(capsys, "--root", str(link), *command)
        assert (code, output, lines(error)) == (1, None, [{"error": LOOP}]), command
    with pytest.raises(OSError) as caught:
        MemoryStore(link)
    assert str(caught.value) == LOOP
    assert [path.name for path in tmp_path.iterdir()] == ["memory-link"] and link.is_symlink()


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser can write to any folder")
def test_a_read_only_folder_without_records_is_refused_without_its_path(tmp_path, capsys):
    root = tmp_path / "TemporalMemory"
    root.mkdir(mode=0o500)
    try:
        code, output, error = run(capsys, "--root", str(root), "history", "user")
    finally:
        root.chmod(0o700)
    assert (code, output) == (1, None)
    assert lines(error) == [{"error": "[Errno 13] Cannot open the memory directory: "
                                      "Permission denied"}]
    assert list(root.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser can write to any folder")
def test_a_directory_that_cannot_be_created_is_reported_without_its_path(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        with pytest.raises(PermissionError) as caught:
            MemoryStore(locked / "memory-data")
    finally:
        locked.chmod(0o700)
    assert str(caught.value) == "[Errno {}] Cannot create the memory directory: {}".format(
        errno.EACCES, os.strerror(errno.EACCES))
    assert str(tmp_path) not in str(caught.value)


def test_a_directory_under_a_file_cannot_be_created_and_the_file_is_left_alone(tmp_path):
    """A file stands where a parent folder should be: the error gives the system's
    reason without the path, and the file is not touched."""
    blocker = tmp_path / "notes.txt"
    blocker.write_text("A file, not a folder.\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError) as caught:
        MemoryStore(blocker / "memory-data")
    assert str(caught.value) == "[Errno {}] Cannot create the memory directory: {}".format(
        errno.ENOTDIR, os.strerror(errno.ENOTDIR))
    assert str(tmp_path) not in str(caught.value)
    assert blocker.read_text(encoding="utf-8") == "A file, not a folder.\n"


@pytest.mark.parametrize("create", [False, True])
def test_a_directory_moved_away_while_opening_is_not_recreated(tmp_path, monkeypatch, create):
    """The check and the folder creation are separate steps; a move between them is
    reported with the usual advice instead of recreating the old path (no
    ``parents=True`` after the check), even when a missing directory may be created."""
    root = tmp_path / "TemporalMemory"
    root.mkdir()
    is_symlink = Path.is_symlink

    def moving(self):  # Called between the check and the creation of memory/.
        if root.exists():
            root.rename(tmp_path / "Moved")
        return is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", moving)
    with pytest.raises(MissingMemoryDirectoryError) as caught:
        MemoryStore(root, create=create)
    monkeypatch.undo()
    assert str(caught.value) == MISSING
    assert sorted(path.name for path in tmp_path.iterdir()) == ["Moved"]


# CLI ------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    ["search", "popcorn"], ["history", "user"], ["add", "user", "snack", "popcorn"],
    ["correct", "0" * 32, "crisps"], ["retract", "0" * 32], ["reindex"],
    ["extract", "--text", "user | snack | popcorn", "--source", "synthetic:chat"],
    ["proposals"], ["approve", "f" * 32], ["reject", "f" * 32],
    ["checkpoint", "--message", "Save"], ["git-log"], ["agent", "What snack do I like?"],
], ids=lambda command: command[0])
def test_cli_commands_that_need_an_existing_memory_report_it_missing(tmp_path, capsys, command):
    root = tmp_path / "TemporalMemory"
    code, output, error = run(capsys, "--root", str(root), *command)
    assert (code, output) == (1, None)
    assert lines(error) == [{"error": MISSING}]
    assert not root.exists()


def test_demo_creates_a_private_directory_quietly(tmp_path, capsys, usual_umask):
    root = tmp_path / "TemporalMemory"
    for _ in range(2):  # Safe to repeat.
        code, output, error = run(capsys, "--root", str(root), "demo")
        assert (code, error) == (0, "") and output
    assert mode(root) == 0o700


@pytest.mark.parametrize("command", [
    ["init"],
    pytest.param(["git-init"], marks=pytest.mark.skipif(not shutil.which("git"),
                                                         reason="Git unavailable")),
], ids=lambda command: command[0])
def test_set_up_commands_create_a_private_directory_quietly(tmp_path, capsys, usual_umask,
                                                            command):
    root = tmp_path / "TemporalMemory"
    code, result, error = run(capsys, "--root", str(root), *command)
    assert (code, error) == (0, "")
    assert result["root"] == str(root.resolve())  # Their output names the new folder.
    assert mode(root) == 0o700


def test_a_write_to_a_moved_memory_is_refused_rather_than_filed_in_a_new_folder(tmp_path,
                                                                                capsys):
    root, moved, _ = moved_memory(tmp_path)
    for command in (["add", "user", "drink", "tea"],
                    ["extract", "--text", "user | drink | tea", "--source", "synthetic:chat"]):
        code, output, error = run(capsys, "--root", str(root), *command)
        assert (code, output, lines(error)) == (1, None, [{"error": MISSING}]), command
    assert sorted(path.name for path in tmp_path.iterdir()) == ["Moved"]
    moved.rename(root)
    code, fact, error = run(capsys, "--root", str(root), "add", "user", "drink", "tea")
    assert (code, error) == (0, "") and fact["value"] == "tea"


@pytest.mark.parametrize("command", [["git-init"], ["checkpoint", "--message", "Save"],
                                     ["git-log"], ["agent", "What snack do I like?"]],
                         ids=lambda command: command[0])
def test_commands_that_check_the_folder_themselves_refuse_a_file_in_its_place(tmp_path, capsys,
                                                                              command):
    """Not "run git-init" or "run init": both would fail on the file."""
    root = tmp_path / "notes.txt"
    root.write_text("not a folder", encoding="utf-8")
    code, output, error = run(capsys, "--root", str(root), *command)
    assert (code, output) == (1, None)
    assert lines(error) == [{"error": "The memory directory path exists but is not a folder"}]
    assert root.read_text(encoding="utf-8") == "not a folder"


@pytest.mark.parametrize("parents", [[], ["new", "sub"]])
def test_a_failed_git_init_takes_back_the_folder_it_made(tmp_path, capsys, monkeypatch, parents):
    """Otherwise later commands would accept the new, empty folder as the memory."""
    from temporal_memory_os import git_history
    monkeypatch.setattr(git_history.shutil, "which", lambda name: None)
    root = tmp_path.joinpath(*parents, "TemporalMemory")
    code, output, error = run(capsys, "--root", str(root), "git-init")
    assert (code, output) == (1, None)
    assert lines(error) == [{"error": "Git is not installed; install Git to use checkpoints."}]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not shutil.which("git"), reason="Git unavailable")
def test_a_git_init_inside_another_repository_takes_back_the_folders_it_made(tmp_path, capsys):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=environment)
    code, output, error = run(capsys, "--root", str(tmp_path / "a" / "b" / "TemporalMemory"),
                              "git-init")
    assert (code, output) == (1, None)
    assert "inside another Git repository" in lines(error)[0]["error"]
    assert [path.name for path in tmp_path.iterdir()] == [".git"]


def test_the_queue_does_not_recreate_a_folder_moved_after_the_store_opened(tmp_path):
    from temporal_memory_os.extraction import ProposalQueue
    root, moved = tmp_path / "TemporalMemory", tmp_path / "Moved"
    with MemoryStore(root) as store:
        root.rename(moved)
        with pytest.raises(MissingMemoryDirectoryError) as caught:
            ProposalQueue(store)
    assert str(caught.value) == MISSING  # The same message as every other command.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["Moved"]
    assert not (moved / "proposals").exists()


def test_a_moved_memory_is_reported_until_it_is_moved_back(tmp_path, capsys):
    root, moved, fact = moved_memory(tmp_path)
    for command in (["search", "popcorn"], ["history", "user"], ["retract", fact.id]):
        code, output, error = run(capsys, "--root", str(root), *command)
        assert (code, output, lines(error)) == (1, None, [{"error": MISSING}]), command
    assert not root.exists()
    moved.rename(root)
    code, history, error = run(capsys, "--root", str(root), "history", "user")
    assert (code, error) == (0, "") and [item["value"] for item in history] == ["popcorn"]


def test_the_default_and_environment_locations_are_not_created_by_reads(tmp_path, capsys,
                                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, _, error = run(capsys, "search", "tea")  # ./memory-data
    assert (code, lines(error)) == (1, [{"error": MISSING}])
    monkeypatch.setenv("MEMORYOS_HOME", str(tmp_path / "home"))
    code, _, error = run(capsys, "--root", "", "history", "person")  # Empty: not given.
    assert (code, lines(error)) == (1, [{"error": MISSING}])
    assert list(tmp_path.iterdir()) == []


def test_init_creates_memoryos_home_with_its_parents_for_add(tmp_path, capsys, monkeypatch):
    """The README's first steps: MEMORYOS_HOME names a folder that does not exist yet."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYOS_HOME", "./memory-data/personal")
    arguments = ("add", "Acme application", "status", "pending", "--from", "2026-09-01")
    code, _, error = run(capsys, *arguments)
    assert (code, lines(error)) == (1, [{"error": MISSING}])
    assert list(tmp_path.iterdir()) == []
    code, result, error = run(capsys, "init")
    home = tmp_path / "memory-data" / "personal"
    assert (code, error, result["root"]) == (0, "", str(home.resolve()))
    code, fact, error = run(capsys, *arguments)
    assert (code, error) == (0, "")
    assert (home / "memory" / "general" / (fact["id"] + ".md")).is_file()
    code, history, error = run(capsys, "history", "Acme application")
    assert (code, error) == (0, "") and [item["value"] for item in history] == ["pending"]


def test_an_empty_folder_made_by_hand_answers_as_an_empty_memory(tmp_path, capsys):
    code, results, error = run(capsys, "--root", str(tmp_path), "search", "tea")
    assert (code, results, error) == (0, [], "")


# MCP server -----------------------------------------------------------------


def test_mcp_tools_report_a_missing_directory_and_recover_when_it_is_back(tmp_path):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    from temporal_memory_os.mcp_server import create_server

    root, moved, fact = moved_memory(tmp_path)

    async def scenario():
        server = create_server(root)  # Starts anyway, so each call can say what is wrong.
        async with create_connected_server_and_client_session(server) as session:
            for name, arguments in (("search", {"query": "popcorn"}), ("fetch", {"id": fact.id}),
                                    ("memory_history", {"entity": "user"})):
                result = await session.call_tool(name, arguments)
                assert result.isError, name
                [content] = result.content
                assert content.text.endswith(": " + MISSING), name
            assert not root.exists()
            moved.rename(root)  # No restart needed once the folder is back.
            result = await session.call_tool("memory_history", {"entity": "user"})
            assert not result.isError
            assert [item["value"] for item in json.loads(result.content[0].text)["facts"]] == [
                "popcorn"]

    asyncio.run(scenario())


@pytest.mark.parametrize("make, message", [
    (lambda memory: memory.write_text("x", encoding="utf-8"),
     "memory/ in the memory directory is not a folder"),
    (lambda memory: memory.symlink_to(memory.parent / "elsewhere", target_is_directory=True),
     "Memory and index folders must not be symbolic links"),
], ids=["file", "link"])
def test_the_agent_names_a_memory_entry_that_is_not_a_folder(tmp_path, make, message):
    """Before any model call; "holds no records yet" is only for a missing memory/."""
    from temporal_memory_os.agent import ask
    root = tmp_path / "TemporalMemory"
    root.mkdir()
    (root / "elsewhere").mkdir()
    make(root / "memory")
    with pytest.raises(StoreIntegrityError) as caught:
        asyncio.run(ask(root, "What snack do I like?"))
    assert str(caught.value) == message


# Servers --------------------------------------------------------------------


@pytest.mark.parametrize("loop", [False, True], ids=["link", "loop"])
@pytest.mark.parametrize("command", ["mcp", "serve"])
def test_servers_keep_a_linked_root_unresolved(tmp_path, monkeypatch, command, loop):
    """Each request opens the store, which resolves the link then, so a link
    re-pointed after the memory moved is followed without a restart. A link
    that loops does not stop a server from starting, on any Python version."""
    root = tmp_path / "TemporalMemory"
    root.mkdir()
    link = tmp_path / "memory-link"
    if loop:
        os.symlink("memory-link", link)
    else:
        link.symlink_to(root, target_is_directory=True)
    started = []
    if command == "mcp":
        pytest.importorskip("mcp")
        from temporal_memory_os import mcp_server
        monkeypatch.setattr(mcp_server, "run_server", started.append)
    else:
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        import uvicorn
        from temporal_memory_os import api
        monkeypatch.setattr(api, "create_app", started.append)
        monkeypatch.setattr(uvicorn, "run", lambda app, **options: None)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--root", "memory-link", command]) == 0
    assert started == [link]


def test_mcp_follows_a_link_re_pointed_after_the_memory_moved(tmp_path):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    from temporal_memory_os.mcp_server import create_server

    root, moved, fact = moved_memory(tmp_path)
    moved.rename(root)
    link = tmp_path / "memory-link"
    link.symlink_to(root, target_is_directory=True)

    async def scenario():
        server = create_server(link)
        async with create_connected_server_and_client_session(server) as session:
            root.rename(moved)
            result = await session.call_tool("fetch", {"id": fact.id})
            assert result.isError
            link.unlink()
            link.symlink_to(moved, target_is_directory=True)
            result = await session.call_tool("fetch", {"id": fact.id})
            assert not result.isError
            assert json.loads(json.loads(result.content[0].text)["text"])["value"] == "popcorn"

    asyncio.run(scenario())


def test_mcp_reports_a_link_re_pointed_into_a_loop_without_its_path(tmp_path):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    from temporal_memory_os.mcp_server import create_server

    root, moved, fact = moved_memory(tmp_path)
    moved.rename(root)
    link = tmp_path / "memory-link"
    link.symlink_to(root, target_is_directory=True)

    async def scenario():
        server = create_server(link)
        async with create_connected_server_and_client_session(server) as session:
            link.unlink()
            os.symlink("memory-link", link)  # A typo while re-pointing it.
            result = await session.call_tool("fetch", {"id": fact.id})
            assert result.isError
            [content] = result.content
            assert content.text.endswith(": " + LOOP)
            assert str(tmp_path) not in content.text
            assert str(tmp_path.resolve()) not in content.text

    asyncio.run(scenario())


# HTTP API -------------------------------------------------------------------


def test_api_follows_a_link_re_pointed_after_the_memory_moved(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from temporal_memory_os.api import create_app

    root, moved, fact = moved_memory(tmp_path)
    moved.rename(root)
    link = tmp_path / "memory-link"
    link.symlink_to(root, target_is_directory=True)
    with TestClient(create_app(link), base_url="http://localhost") as client:
        root.rename(moved)
        assert client.get("/api/facts/" + fact.id).status_code == 503
        link.unlink()
        link.symlink_to(moved, target_is_directory=True)
        response = client.get("/api/facts/" + fact.id)
        assert response.status_code == 200 and response.json()["value"] == "popcorn"


def test_api_reports_a_missing_directory_as_503_and_never_creates_it(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from temporal_memory_os.api import create_app

    root, moved, fact = moved_memory(tmp_path)
    requests = [
        ("get", "/api/facts", {}),
        ("get", "/api/facts/" + fact.id, {}),
        ("get", "/api/search", {"params": {"q": "popcorn"}}),
        ("get", "/api/history", {"params": {"entity": "user"}}),
        ("get", "/api/proposals", {}),
        ("post", "/api/facts", {"json": {"entity": "user", "attribute": "drink", "value": "tea"}}),
        ("post", "/api/facts/{}/corrections".format(fact.id), {"json": {"value": "crisps"}}),
        ("delete", "/api/facts/" + fact.id, {}),
        ("post", "/api/proposals/extract",
         {"json": {"text": "user | drink | tea", "source": "synthetic:chat"}}),
        ("post", "/api/proposals/{}/approve".format("f" * 32), {}),
        ("post", "/api/proposals/{}/reject".format("f" * 32), {}),
        ("post", "/api/reindex", {}),
    ]
    with TestClient(create_app(root), base_url="http://localhost") as client:
        assert client.get("/health").status_code == 200
        for method, path, options in requests:
            response = getattr(client, method)(path, **options)
            assert (response.status_code, response.json()) == (503, {"detail": MISSING}), path
        assert not root.exists()
        moved.rename(root)
        response = client.get("/api/history", params={"entity": "user"})
        assert response.status_code == 200
        assert [item["value"] for item in response.json()["facts"]] == ["popcorn"]


def test_api_reports_a_file_in_place_of_the_proposals_folder_as_503(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from temporal_memory_os.api import create_app

    root = tmp_path / "TemporalMemory"
    with MemoryStore(root):
        pass
    (root / "proposals").write_text("not a folder", encoding="utf-8")
    requests = [
        ("get", "/api/proposals", {}),
        ("post", "/api/proposals/extract",
         {"json": {"text": "user | drink | tea", "source": "synthetic:chat"}}),
    ]
    with TestClient(create_app(root), base_url="http://localhost") as client:
        for method, path, options in requests:
            response = getattr(client, method)(path, **options)
            assert (response.status_code, response.json()) == (
                503, {"detail": "proposals/ in the memory directory is not a folder"}), path
