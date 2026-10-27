"""Git checkpoints: what they commit, where, with which identity, and when.

conftest.py gives every test an empty global Git config and no Git identity,
so each test below sets the identity it needs.
"""

import json
import math
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

from temporal_memory_os import git_history
from temporal_memory_os.embeddings import HashingEmbedder
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.git_history import GitHistory
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.store import MemoryStore


pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="Git unavailable")

WAIT = 120  # Seconds; generous for slow CI machines. Passing runs take far less.
# Variables that would point the helper below at another repository.
LOCATION_VARIABLES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                      "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR")
EXCLUDE_HEADER = "# Temporal MemoryOS derived/local files"
# Every rule git-init writes, in order. "!*.sqlite3*/" and "!*.tmp/" keep topic
# folders named like "app.sqlite3" or "notes.tmp" in; the last two leave editor
# backups ("notes.md~", "#notes.md#") out.
EXCLUDE_RULES = (".memoryos/", "*.sqlite3*", "!*.sqlite3*/", ".env", "proposals/.lock", "*.tmp",
                 "!*.tmp/", "/memory/**/.*", "/proposals/.*", "/memory/**/*~", "/memory/**/#*#")
# How the refusals tell you to mark a repository you made yourself for checkpoints.
OPT_IN = "git -C <memory directory> config memoryos.checkpoints true"


def run_git(root, *arguments):
    environment = {key: value for key, value in os.environ.items()
                   if key not in LOCATION_VARIABLES}
    return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True,
                          env=environment)


def git(root, *arguments):
    result = run_git(root, *arguments)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def identify(root, name="MemoryOS Test", email="memoryos-test@example.invalid"):
    git(root, "config", "user.name", name)
    git(root, "config", "user.email", email)


def tree(root, revision="HEAD"):
    return git(root, "ls-tree", "-r", "--name-only", revision).splitlines()


def add_fact(root, value="Riverton", attribute="home city"):
    with MemoryStore(root) as store:
        fact = store.add("Alex", attribute, value, valid_from="2025-01-01")
        return fact, store.path_for(fact.id).relative_to(store.root).as_posix()


@pytest.fixture
def history(tmp_path):
    history = GitHistory(tmp_path)
    assert history.initialize()["initialized"] is True
    identify(tmp_path)
    return history


def in_thread(function):
    """Start ``function`` in a thread; the list receives its result or exception."""
    outcome = []

    def target():
        try:
            outcome.append(function())
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome


def test_checkpoint_only_contains_memory_and_preserves_other_staging(tmp_path):
    history = GitHistory(tmp_path)
    assert history.initialize()["initialized"]
    assert not history.initialize()["initialized"]
    identify(tmp_path)
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "a.md").write_text("A synthetic fact", encoding="utf-8")
    (tmp_path / "proposals").mkdir()
    (tmp_path / "proposals" / ".lock").write_text("", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("must remain staged", encoding="utf-8")
    git(tmp_path, "add", "unrelated.txt")
    result = history.checkpoint("Add synthetic memory")
    assert result["committed"]
    assert git(tmp_path, "ls-tree", "-r", "--name-only", "HEAD") == "memory/a.md"
    assert git(tmp_path, "diff", "--cached", "--name-only") == "unrelated.txt"
    assert not history.checkpoint("No changes")["committed"]
    assert history.log()[0]["message"] == "Add synthetic memory"
    shutil.rmtree(tmp_path / "memory")
    assert history.checkpoint("Remove memory files")["committed"]
    assert git(tmp_path, "ls-tree", "-r", "--name-only", "HEAD") == ""
    assert git(tmp_path, "diff", "--cached", "--name-only") == "unrelated.txt"


def test_git_init_refuses_parent_repository(tmp_path):
    GitHistory(tmp_path).initialize()
    child = tmp_path / "nested"
    child.mkdir()
    with pytest.raises(ValueError, match="inside another"):
        GitHistory(child).initialize()
    with pytest.raises(ValueError, match="needs its own repository"):
        GitHistory(child).checkpoint("Nested")
    with pytest.raises(ValueError, match="needs its own repository"):
        GitHistory(child).log()


def test_git_checkpoint_refuses_symlinks(tmp_path):
    history = GitHistory(tmp_path)
    history.initialize()
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "outside").symlink_to(tmp_path / "outside-file")
    with pytest.raises(ValueError, match="symbolic links"):
        history.checkpoint("Cannot include links")


def test_checkpoint_and_log_reject_bad_input_and_missing_repositories(
        tmp_path, tmp_path_factory, history):
    with pytest.raises(ValueError, match="non-empty"):
        history.checkpoint("   ")
    for limit in (0, 1001, True, "5"):
        with pytest.raises(ValueError, match="between 1 and 1000"):
            history.log(limit)
    (tmp_path / "memory").write_text("not a folder", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        history.checkpoint("File in place of the memory folder")
    plain = GitHistory(tmp_path_factory.mktemp("not-a-repository"))
    with pytest.raises(ValueError, match="Run git-init first"):
        plain.checkpoint("Nothing to commit to")
    with pytest.raises(ValueError, match="Run git-init first"):
        plain.log()
    assert not (plain.root / ".git").exists()


def test_git_location_variables_cannot_redirect_memory_into_another_repository(
        tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    git(project, "init", "--quiet")
    identify(project, "Project Person", "project@example.invalid")
    (project / "app.py").write_text("print('project')\n", encoding="utf-8")
    git(project, "add", "app.py")
    git(project, "commit", "--quiet", "-m", "Project")
    head = git(project, "rev-parse", "HEAD")
    index = (project / ".git" / "index").read_bytes()
    # What a Git hook or `git rebase --exec` exports to the commands it runs.
    monkeypatch.setenv("GIT_DIR", str(project / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(project))
    monkeypatch.setenv("GIT_INDEX_FILE", str(project / ".git" / "index"))
    root = tmp_path / "private-memory"
    history = GitHistory(root)
    assert history.initialize()["initialized"] is True
    assert (root / ".git").is_dir()
    identify(root)
    _, path = add_fact(root, "private value", "secret")
    assert history.checkpoint("Hook checkpoint")["committed"]
    assert tree(root) == [path]
    assert [entry["message"] for entry in history.log()] == ["Hook checkpoint"]
    # The project repository is exactly as it was.
    assert git(project, "rev-parse", "HEAD") == head
    assert git(project, "log", "--format=%s") == "Project"
    assert tree(project) == ["app.py"]
    assert (project / ".git" / "index").read_bytes() == index
    assert not (project / "memory").exists()


def test_checkpoint_handles_more_paths_than_fit_on_a_command_line(tmp_path, history):
    # Earlier releases passed every path under memory/ to `git add` and
    # `git commit` as arguments, which fails with E2BIG once their total size
    # passes the system limit (1 MiB on macOS, typically 2 MiB on Linux, where
    # a store of ten to thirteen thousand records failed). The limit counts
    # bytes, not paths, so long names exceed it with a few thousand files
    # (about 2,400 on macOS and 4,800 on Linux) and keep this test fast.
    # Each relative path is about 530 bytes, well inside PATH_MAX (1,024 on
    # macOS) even under a long temporary directory. The notes share one text,
    # so Git stores a single object and the time goes into the paths.
    folder = tmp_path.joinpath("memory", *["career-applications-and-follow-ups-" * 3 + part
                                           for part in ("2025", "recruiters", "next-steps")])
    folder.mkdir(parents=True)
    template = "follow-up-note-{:05d}" + "-with-recruiter-contacts-and-next-steps" * 4 + ".md"
    relative = folder.relative_to(tmp_path).as_posix() + "/" + template.format(0)
    # Argument text, NUL and pointer, as the old code spent on each path.
    per_path = len(":(literal)") + len(relative) + 1 + 8
    try:
        limit = os.sysconf("SC_ARG_MAX")
    except (AttributeError, ValueError, OSError):
        limit = -1
    if not 0 < limit <= 4 * 1024 * 1024:
        limit = 2 * 1024 * 1024  # Unknown or unlimited: use the common Linux default.
    count = math.ceil(1.25 * limit / per_path)
    for number in range(count):
        (folder / template.format(number)).write_text("Follow up next week.\n", encoding="utf-8")
    result = history.checkpoint("Many notes")
    assert result["committed"]
    assert len(tree(tmp_path)) == count
    # Later checkpoints keep working and commit only what changed.
    _, path = add_fact(tmp_path)
    assert history.checkpoint("One more record")["committed"]
    assert git(tmp_path, "diff", "--name-only", "HEAD~1", "HEAD") == path


def test_checkpoint_leaves_out_temporary_and_hidden_files(tmp_path, history):
    _, fact_path = add_fact(tmp_path)
    with MemoryStore(tmp_path) as store:
        proposal = ProposalQueue(store).extract("Alex | employer | Acme | 2025-01-01",
                                                source="synthetic:test")[0]
    general = tmp_path / "memory" / "general"
    # An interrupted write, editor lock and swap files, and another tool's folder.
    (general / ".memoryos-interrupted.tmp").write_text("partial record", encoding="utf-8")
    (general / ".#draft.md").symlink_to("someone@host.12345")  # A dangling Emacs lock link.
    (general / ".draft.md.swp").write_bytes(b"\x00swap")
    (general / "scratch.tmp").write_text("scratch", encoding="utf-8")
    (tmp_path / "memory" / ".obsidian").mkdir()
    (tmp_path / "memory" / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")
    (tmp_path / "proposals" / ".proposal-interrupted.tmp").write_text("{", encoding="utf-8")
    assert history.checkpoint("Memory only")["committed"]
    assert tree(tmp_path) == sorted([fact_path, "proposals/{}.json".format(proposal.id)])
    assert not history.checkpoint("Nothing new")["committed"]


def test_git_init_refuses_an_existing_project_repository(tmp_path):
    git(tmp_path, "init", "--quiet")
    identify(tmp_path)
    (tmp_path / "app.py").write_text("print('project')\n", encoding="utf-8")
    git(tmp_path, "add", "app.py")
    git(tmp_path, "commit", "--quiet", "-m", "Project")
    exclude = (tmp_path / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    history = GitHistory(tmp_path)
    with pytest.raises(ValueError, match="MemoryOS did not create") as refused:
        history.initialize()
    # Both refusals say how to opt in, so neither sends you back to the other.
    assert "Use a separate memory directory" in str(refused.value)
    assert OPT_IN in str(refused.value)
    add_fact(tmp_path)
    with pytest.raises(ValueError, match="was not created by `memoryos git-init`") as refused:
        history.checkpoint("Must not reach the project")
    assert OPT_IN in str(refused.value)
    assert "Run git-init first" not in str(refused.value)
    assert run_git(tmp_path, "config", "--local", "--get", "memoryos.checkpoints").returncode == 1
    assert (tmp_path / ".git" / "info" / "exclude").read_text(encoding="utf-8") == exclude
    assert git(tmp_path, "log", "--format=%s") == "Project"
    assert git(tmp_path, "diff", "--cached", "--name-only") == ""


def test_git_init_adopts_a_clone_of_a_checkpoint_repository(tmp_path):
    origin = tmp_path / "origin"
    GitHistory(origin).initialize()
    identify(origin)
    fact, path = add_fact(origin)
    assert GitHistory(origin).checkpoint("Original")["committed"]
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "--quiet", str(origin), str(clone))
    # A clone copies neither the local marker nor the local exclude rules.
    assert run_git(clone, "config", "--local", "--get", "memoryos.checkpoints").returncode == 1
    history = GitHistory(clone)
    assert history.initialize() == {"initialized": False, "root": str(clone.resolve())}
    assert history.initialize()["initialized"] is False  # Repeating changes nothing.
    assert git(clone, "config", "--local", "--get", "memoryos.checkpoints") == "true"
    exclude = (clone / ".git" / "info" / "exclude").read_text(encoding="utf-8").splitlines()
    assert all(exclude.count(rule) == 1 for rule in EXCLUDE_RULES)
    with MemoryStore(clone) as store:  # Builds the derived index inside the clone.
        assert store.all_facts() == [fact]
        ProposalQueue(store).extract("Alex | employer | Acme | 2025-01-01",
                                     source="synthetic:test")
    status = git(clone, "status", "--porcelain", "--untracked-files=all").splitlines()
    assert [line for line in status if ".memoryos" in line or ".lock" in line] == []
    identify(clone)
    assert history.checkpoint("From the clone")["committed"]
    assert path in tree(clone)


def test_git_init_creates_the_exclude_file_when_the_template_has_no_info_folder(
        tmp_path, monkeypatch):
    template = tmp_path / "empty-template"
    template.mkdir()
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(template))
    # The premise: with this template, `git init` creates no info/ folder.
    probe = tmp_path / "probe"
    probe.mkdir()
    git(probe, "init", "--quiet")
    assert not (probe / ".git" / "info").exists()
    root = tmp_path / "memory-store"
    history = GitHistory(root)
    assert history.initialize()["initialized"] is True
    exclude = root / ".git" / "info" / "exclude"
    # Exactly the documented rules, in order: "!*.tmp/" only works after "*.tmp".
    assert exclude.read_text(encoding="utf-8").splitlines() == [EXCLUDE_HEADER, *EXCLUDE_RULES]
    assert history.initialize()["initialized"] is False
    assert exclude.read_text(encoding="utf-8").splitlines() == [EXCLUDE_HEADER, *EXCLUDE_RULES]


def test_checkpoint_refuses_to_guess_an_identity(tmp_path):
    history = GitHistory(tmp_path)
    history.initialize()
    add_fact(tmp_path)
    with pytest.raises(ValueError, match="user.name"):
        history.checkpoint("No identity")
    assert git(tmp_path, "diff", "--cached", "--name-only") == ""  # Nothing was staged.
    assert run_git(tmp_path, "rev-parse", "--verify", "--quiet", "HEAD").returncode != 0


@pytest.mark.parametrize("source", ["environment", "global config", "repository config"])
def test_checkpoint_uses_an_identity_configured_in_git(tmp_path, monkeypatch, source):
    history = GitHistory(tmp_path)
    history.initialize()
    if source == "environment":
        for role in ("AUTHOR", "COMMITTER"):
            monkeypatch.setenv("GIT_{}_NAME".format(role), "Environment Person")
            monkeypatch.setenv("GIT_{}_EMAIL".format(role), "environment@example.invalid")
        expected = "Environment Person <environment@example.invalid>"
    elif source == "global config":
        with open(os.environ["GIT_CONFIG_GLOBAL"], "w", encoding="utf-8") as handle:
            handle.write("[user]\n\tname = Global Person\n\temail = global@example.invalid\n")
        expected = "Global Person <global@example.invalid>"
    else:
        identify(tmp_path, "Local Person", "local@example.invalid")
        expected = "Local Person <local@example.invalid>"
    add_fact(tmp_path)
    assert history.checkpoint("With identity")["committed"]
    assert git(tmp_path, "log", "-1", "--format=%an <%ae>|%cn <%ce>") == "{0}|{0}".format(expected)
    if source != "repository config":  # Never copied into the repository's config.
        assert run_git(tmp_path, "config", "--local", "--get", "user.name").returncode == 1


def test_checkpoint_waits_for_an_approval_in_progress(tmp_path, history, monkeypatch):
    store = MemoryStore(tmp_path)
    try:
        queue = ProposalQueue(store)
        proposal = queue.extract("Alex | employer | Acme | 2025-01-01",
                                 source="synthetic:test")[0]
        assert history.checkpoint("Pending proposal")["committed"]
        # Pause approve() after it wrote the fact and before it saves the
        # proposal as approved, while it holds the proposal lock.
        fact_written, resume = threading.Event(), threading.Event()
        save = ProposalQueue._save

        def paused_save(self, item):
            if item.status == "approved":
                fact_written.set()
                assert resume.wait(WAIT)
            save(self, item)

        monkeypatch.setattr(ProposalQueue, "_save", paused_save)
        approver, approved = in_thread(lambda: queue.approve(proposal.id))
        try:
            assert fact_written.wait(WAIT)
            checkpointer, checkpointed = in_thread(lambda: history.checkpoint("During approval"))
            checkpointer.join(1.0)
            assert checkpointer.is_alive() and checkpointed == []  # Waiting for the lock.
        finally:
            resume.set()  # Never leave the approver holding the lock after a failure.
        approver.join(WAIT)
        checkpointer.join(WAIT)
        assert len(approved) == 1 and not isinstance(approved[0], BaseException), approved
        assert len(checkpointed) == 1 and not isinstance(checkpointed[0], BaseException), \
            checkpointed
        fact = approved[0]
        assert checkpointed[0]["committed"]
        # The commit holds the fact together with the proposal that approved it.
        assert store.path_for(fact.id).relative_to(tmp_path.resolve()).as_posix() in tree(
            tmp_path)
        committed = json.loads(git(tmp_path, "show", "HEAD:proposals/{}.json".format(proposal.id)))
        assert (committed["status"], committed["result_fact_id"]) == ("approved", fact.id)
    finally:
        store.close()


def paused_commits(monkeypatch):
    """Pause every checkpoint inside its locks, just before it stages anything.

    Returns (staging, resume): ``staging`` is set once a checkpoint is paused,
    and setting ``resume`` lets every paused checkpoint continue.
    """
    staging, resume = threading.Event(), threading.Event()
    commit = GitHistory._commit

    def paused_commit(self, message, targets):
        staging.set()
        assert resume.wait(WAIT)
        return commit(self, message, targets)

    monkeypatch.setattr(GitHistory, "_commit", paused_commit)
    return staging, resume


def test_checkpoint_holds_the_store_write_lock_while_committing(tmp_path, history, monkeypatch):
    _, first = add_fact(tmp_path)
    staging, resume = paused_commits(monkeypatch)
    checkpointer, checkpointed = in_thread(lambda: history.checkpoint("Paused"))
    try:
        assert staging.wait(WAIT)
        writer, written = in_thread(lambda: add_fact(tmp_path, "Lakeside", "employer"))
        writer.join(1.0)
        assert writer.is_alive() and written == []  # Waits for the checkpoint.
    finally:
        resume.set()
    checkpointer.join(WAIT)
    writer.join(WAIT)
    assert len(checkpointed) == 1 and checkpointed[0]["committed"], checkpointed
    assert len(written) == 1 and not isinstance(written[0], BaseException), written
    assert tree(tmp_path) == [first]  # The later record is not in this commit.
    assert history.checkpoint("Later record")["committed"]
    assert tree(tmp_path) == sorted([first, written[0][1]])


def test_concurrent_checkpoints_take_turns(tmp_path, history, monkeypatch):
    _, path = add_fact(tmp_path)
    staging, resume = paused_commits(monkeypatch)
    first, first_result = in_thread(lambda: history.checkpoint("First"))
    try:
        assert staging.wait(WAIT)
        second, second_result = in_thread(lambda: GitHistory(tmp_path).checkpoint("Second"))
        second.join(1.0)
        # Waiting for the first checkpoint's locks, not racing it for Git's index.
        assert second.is_alive() and second_result == []
    finally:
        resume.set()
    first.join(WAIT)
    second.join(WAIT)
    assert len(first_result) == 1 and first_result[0]["committed"], first_result
    assert second_result == [{"committed": False, "reason": "No memory changes."}]
    assert [entry["message"] for entry in history.log()] == ["First"]
    assert tree(tmp_path) == [path]


def test_a_git_command_that_overruns_becomes_an_error_that_says_which_lock_to_delete(
        tmp_path, history, monkeypatch):
    _, path = add_fact(tmp_path)
    hook = tmp_path / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text("#!/bin/sh\nsleep 3\n", encoding="utf-8")
    hook.chmod(0o755)
    with monkeypatch.context() as patch:
        patch.setattr(git_history, "_TIMEOUT", 1)
        with pytest.raises(RuntimeError, match="did not finish within 1 seconds") as caught:
            history.checkpoint("Slow hook")
    message = str(caught.value)
    assert str(tmp_path) not in message and str(tmp_path.resolve()) not in message
    assert "If no other Git command is running" in message
    assert "delete .git/index.lock" in message
    assert run_git(tmp_path, "rev-parse", "--verify", "--quiet", "HEAD").returncode != 0
    # Git was stopped while it held the index lock. MemoryOS cannot tell that
    # lock from one held by a Git command you started, so it leaves it alone.
    lock = tmp_path / ".git" / "index.lock"
    assert lock.exists()
    hook.unlink()
    with pytest.raises(RuntimeError, match="index.lock"):  # Git refuses while it is there.
        history.checkpoint("Lock still in place")
    lock.unlink()  # What the message asks you to do.
    assert history.checkpoint("After the lock is removed")["committed"]
    assert tree(tmp_path) == [path]
    assert [entry["message"] for entry in history.log()] == ["After the lock is removed"]


def test_checkpoint_commits_deleting_every_record_and_skips_an_untracked_proposals_folder(
        tmp_path, history):
    _, path = add_fact(tmp_path)
    (tmp_path / "proposals").mkdir()  # Empty, so Git has never tracked anything in it.
    assert history.checkpoint("One record")["committed"]
    assert tree(tmp_path) == [path]
    assert [entry.name for entry in (tmp_path / "proposals").iterdir()] == [".lock"]
    (tmp_path / path).unlink()  # The folders stay; only the record goes.
    assert (tmp_path / "memory" / "general").is_dir()
    assert history.checkpoint("Delete the record")["committed"]
    assert tree(tmp_path) == []
    assert git(tmp_path, "diff", "--name-status", "HEAD~1", "HEAD") == "D\t" + path
    assert not history.checkpoint("Nothing left")["committed"]


def test_a_repository_made_by_0_2_0_gets_the_dot_and_temporary_rules_before_staging(tmp_path,
                                                                                    history):
    # 0.2.0 wrote only these rules and no ownership marker. Checkpoints now stage
    # whole folders, so the missing rules must be added before anything is staged.
    exclude = tmp_path / ".git" / "info" / "exclude"
    exclude.write_text("\n# Temporal MemoryOS derived/local files\n.memoryos/\n*.sqlite3*\n.env\n"
                       "proposals/.lock\n", encoding="utf-8")
    git(tmp_path, "config", "--local", "--unset", "memoryos.checkpoints")
    _, path = add_fact(tmp_path)
    general = tmp_path / "memory" / "general"
    (general / ".draft.md.swp").write_bytes(b"\x00swap")
    (general / "scratch.tmp").write_text("scratch", encoding="utf-8")
    (tmp_path / "memory" / ".obsidian").mkdir()
    (tmp_path / "memory" / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")
    assert history.checkpoint("After upgrading")["committed"]
    assert tree(tmp_path) == [path]
    lines = exclude.read_text(encoding="utf-8").splitlines()
    assert all(lines.count(rule) == 1 for rule in EXCLUDE_RULES)


def test_a_repository_made_by_0_2_1_keeps_temporary_rules_off_topic_folders(tmp_path, history):
    # 0.2.1 wrote "*.tmp" without "!*.tmp/", which also hid every record in a
    # topic folder such as "notes.tmp", and had no rules for editor backups.
    exclude = tmp_path / ".git" / "info" / "exclude"
    old = ["# A comment from Git's template", "", EXCLUDE_HEADER, ".memoryos/", "*.sqlite3*",
           ".env", "proposals/.lock", "*.tmp", "/memory/**/.*", "/proposals/.*"]
    exclude.write_text("\n".join(old) + "\n", encoding="utf-8")
    with MemoryStore(tmp_path) as store:
        fact = store.add("Alex", "home city", "Riverton", valid_from="2025-01-01",
                         topic="notes.tmp")
        path = store.path_for(fact.id).relative_to(store.root).as_posix()
    (tmp_path / "memory" / "notes.tmp" / "notes.md~").write_text("backup", encoding="utf-8")
    assert history.checkpoint("After upgrading")["committed"]
    assert tree(tmp_path) == [path]
    lines = exclude.read_text(encoding="utf-8").splitlines()
    # Lines before the block stay as they were; the block gets every rule in the documented
    # order, since Git applies the last rule that matches.
    assert lines[:2] == old[:2]
    assert lines[2:] == [EXCLUDE_HEADER, *EXCLUDE_RULES]


def test_a_repository_made_by_0_2_2_keeps_dot_and_backup_folders_out_of_checkpoints(
        tmp_path, history):
    """Upgraded rules keep "!*.sqlite3*/" before the dot-file and backup rules; your own stay."""
    exclude = tmp_path / ".git" / "info" / "exclude"
    rules_0_2_2 = [".memoryos/", "*.sqlite3*", ".env", "proposals/.lock", "*.tmp", "!*.tmp/",
                   "/memory/**/.*", "/proposals/.*", "/memory/**/*~", "/memory/**/#*#"]
    own = ["# A rule of your own", "local-notes/"]
    exclude.write_text("\n".join(["# A comment from Git's template", "", EXCLUDE_HEADER,
                                  *rules_0_2_2, *own]) + "\n", encoding="utf-8")
    with MemoryStore(tmp_path) as store:
        fact = store.add("Inventory db", "engine", "SQLite 3.45", valid_from="2025-01-01",
                         topic="tools/app.sqlite3")
        path = store.path_for(fact.id).relative_to(store.root).as_posix()
    memory = tmp_path / "memory"
    for folder in (".x.sqlite3", "n.sqlite3~", "#n.sqlite3#"):
        (memory / folder).mkdir()
        (memory / folder / "f.md").write_text("left out", encoding="utf-8")
    (memory / ".x.sqlite3" / "f.txt").write_text("left out", encoding="utf-8")
    assert history.checkpoint("After upgrading")["committed"]
    assert tree(tmp_path) == [path]
    lines = exclude.read_text(encoding="utf-8").splitlines()
    assert lines == ["# A comment from Git's template", "", EXCLUDE_HEADER, *EXCLUDE_RULES,
                     *own]
    assert not history.checkpoint("Nothing new")["committed"]


def test_checkpoint_commits_topic_folders_named_like_temporary_files_but_not_backups(
        tmp_path, history):
    paths = []
    with MemoryStore(tmp_path) as store:
        for topic in ("general", "notes.tmp", "notes.tmp/drafts.tmp"):
            fact = store.add("Alex", "note in " + topic, "kept", valid_from="2025-01-01",
                             topic=topic)
            paths.append(store.path_for(fact.id).relative_to(store.root).as_posix())
    memory = tmp_path / "memory"
    for folder in (memory, memory / "general", memory / "notes.tmp" / "drafts.tmp"):
        # Temporary files and the backups Emacs and other editors leave next to a note.
        for name in ("scratch.tmp", "notes.md~", "#notes.md#"):
            (folder / name).write_text("left out", encoding="utf-8")
    assert history.checkpoint("Topics named like temporary files")["committed"]
    assert tree(tmp_path) == sorted(paths)
    assert not history.checkpoint("Nothing new")["committed"]


def test_checkpoint_commits_topic_folders_named_like_database_files(tmp_path, history):
    """A topic folder such as tools/app.sqlite3 is committed; database files stay out."""
    with MemoryStore(tmp_path) as store:
        fact = store.add("Inventory db", "engine", "SQLite 3.45", valid_from="2025-01-01",
                         topic="tools/app.sqlite3")
        path = store.path_for(fact.id).relative_to(store.root).as_posix()
    tools = tmp_path / "memory" / "tools"
    (tools / "cache.sqlite3").write_bytes(b"derived")
    (tools / "app.sqlite3" / "cache.sqlite3-wal").write_bytes(b"derived")
    assert history.checkpoint("Database topic")["committed"]
    assert tree(tmp_path) == [path]
    assert not history.checkpoint("Nothing new")["committed"]


@pytest.mark.parametrize("setting", ["default location", "core.excludesFile"])
def test_your_personal_ignore_file_does_not_hide_records_from_checkpoints(tmp_path, history,
                                                                          setting):
    """Topic folders that your personal Git ignore file names are still committed."""
    personal = Path(os.environ["XDG_CONFIG_HOME"]) / "git" / "ignore"
    if setting == "core.excludesFile":
        personal = Path(os.environ["XDG_CONFIG_HOME"]) / "personal-ignore"
        git(tmp_path, "config", "--file", os.environ["GIT_CONFIG_GLOBAL"], "core.excludesFile",
            str(personal))
    personal.parent.mkdir(parents=True, exist_ok=True)
    personal.write_text("build/\nlogs/\n", encoding="utf-8")
    paths = []
    with MemoryStore(tmp_path) as store:
        for topic in ("general", "projects/build", "work/logs"):
            fact = store.add("Alex", "note in " + topic, "kept", valid_from="2025-01-01",
                             topic=topic)
            paths.append(store.path_for(fact.id).relative_to(store.root).as_posix())
    # The premise: Git itself ignores these folders in this repository.
    assert git(tmp_path, "check-ignore", "memory/projects/build", "memory/work/logs").split() == [
        "memory/projects/build", "memory/work/logs"]
    assert history.checkpoint("Records in every topic")["committed"]
    assert tree(tmp_path) == sorted(paths)
    assert not history.checkpoint("Nothing new")["committed"]


@pytest.mark.parametrize("variable", ["GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS",
                                      "GIT_ICASE_PATHSPECS", "GIT_LITERAL_PATHSPECS"])
def test_pathspec_settings_in_your_environment_do_not_break_checkpoints(
        tmp_path, history, monkeypatch, variable):
    _, path = add_fact(tmp_path)
    monkeypatch.setenv(variable, "1")
    assert history.checkpoint("With " + variable)["committed"]
    assert [entry["message"] for entry in history.log()] == ["With " + variable]
    assert tree(tmp_path) == [path]


def test_commit_hooks_match_pathspecs_the_usual_way(tmp_path, history):
    # A hook that filters staged files with a glob, as secret scanners and
    # linters do, must see the record: no pathspec mode is forced on it.
    _, path = add_fact(tmp_path)
    seen = tmp_path / ".git" / "hook-saw.txt"
    hook = tmp_path / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text("#!/bin/sh\ngit diff --cached --name-only -- '*.md' > '{}'\n".format(seen),
                    encoding="utf-8")
    hook.chmod(0o755)
    assert history.checkpoint("Hook sees the record")["committed"]
    assert seen.read_text(encoding="utf-8").split() == [path]


def test_checkpoints_rewrite_neither_git_config_nor_exclude_file_once_both_are_set(
        tmp_path, history):
    _, path = add_fact(tmp_path)
    config = tmp_path / ".git" / "config"
    exclude = tmp_path / ".git" / "info" / "exclude"
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write("# A rule of your own\nlocal-notes/\n")
    an_hour_ago = time.time() - 3600
    for file in (config, exclude):
        os.utime(file, (an_hour_ago, an_hour_ago))
    before = {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in (config, exclude)}
    assert history.checkpoint("First")["committed"]
    assert not history.checkpoint("Nothing new")["committed"]
    assert history.initialize()["initialized"] is False
    # Not even rewritten with the same content: writing takes Git's config lock,
    # which a concurrent checkpoint or your own Git command may hold.
    assert {file: (file.read_bytes(), file.stat().st_mtime_ns)
            for file in (config, exclude)} == before
    assert tree(tmp_path) == [path]


def test_adoption_waits_for_the_checkpoint_locks(tmp_path, history, monkeypatch):
    # A repository without the marker (one made by 0.2.0) is adopted by its first
    # checkpoints. Adopting writes Git's config, so two checkpoints must take turns.
    git(tmp_path, "config", "--local", "--unset", "memoryos.checkpoints")
    _, path = add_fact(tmp_path)
    adopting, resume = threading.Event(), threading.Event()
    entered = []
    adopt = GitHistory._adopt

    def paused_adopt(self):
        entered.append(self)
        if len(entered) == 1:
            adopting.set()
            assert resume.wait(WAIT)
        return adopt(self)

    monkeypatch.setattr(GitHistory, "_adopt", paused_adopt)
    first, first_result = in_thread(lambda: history.checkpoint("First"))
    try:
        assert adopting.wait(WAIT)
        second, second_result = in_thread(lambda: GitHistory(tmp_path).checkpoint("Second"))
        second.join(1.0)
        # Waiting for the first checkpoint's locks before it adopts anything.
        assert second.is_alive() and second_result == [] and len(entered) == 1
    finally:
        resume.set()
    first.join(WAIT)
    second.join(WAIT)
    assert len(first_result) == 1 and first_result[0]["committed"], first_result
    assert second_result == [{"committed": False, "reason": "No memory changes."}]
    assert len(entered) == 2
    assert git(tmp_path, "config", "--local", "--get", "memoryos.checkpoints") == "true"
    assert tree(tmp_path) == [path]


@pytest.mark.parametrize("history_before", ["a root .gitignore", "no commits"])
def test_a_repository_you_made_yourself_works_once_marked_as_the_message_says(
        tmp_path, history_before):
    # 0.2.0 accepted any repository at the memory directory, so its users may
    # have run `git init` there themselves and committed a .gitignore.
    git(tmp_path, "init", "--quiet")
    identify(tmp_path)
    committed = []
    if history_before == "a root .gitignore":
        (tmp_path / ".gitignore").write_text(".memoryos/\n", encoding="utf-8")
        git(tmp_path, "add", ".gitignore")
        git(tmp_path, "commit", "--quiet", "-m", "Ignore the index")
        committed.append(".gitignore")
    _, path = add_fact(tmp_path)
    history = GitHistory(tmp_path)
    for refused_call in (lambda: history.checkpoint("Not yet"), history.log, history.initialize):
        with pytest.raises(ValueError, match="MemoryOS did not create|not created by") as caught:
            refused_call()
        assert OPT_IN in str(caught.value)
        assert str(tmp_path.resolve()) not in str(caught.value)
    assert run_git(tmp_path, "config", "--local", "--get", "memoryos.checkpoints").returncode == 1
    git(tmp_path, "config", "memoryos.checkpoints", "true")  # The command in the message.
    assert history.checkpoint("Marked")["committed"]
    assert tree(tmp_path) == sorted(committed + [path])
    assert history.log()[0]["message"] == "Marked"
    exclude = (tmp_path / ".git" / "info" / "exclude").read_text(encoding="utf-8").splitlines()
    assert all(exclude.count(rule) == 1 for rule in EXCLUDE_RULES)


def test_a_repository_you_made_that_tracks_only_memory_is_adopted(tmp_path):
    git(tmp_path, "init", "--quiet")
    identify(tmp_path)
    _, first = add_fact(tmp_path)
    git(tmp_path, "add", "memory")
    git(tmp_path, "commit", "--quiet", "-m", "By hand")
    _, second = add_fact(tmp_path, "Acme", "employer")
    history = GitHistory(tmp_path)
    assert history.checkpoint("Adopted")["committed"]
    assert git(tmp_path, "config", "--local", "--get", "memoryos.checkpoints") == "true"
    assert tree(tmp_path) == sorted([first, second])
    assert [entry["message"] for entry in history.log()] == ["Adopted", "By hand"]


def test_a_root_spelled_in_another_letter_case_is_the_same_repository(tmp_path):
    root = tmp_path / "casetest"
    assert GitHistory(root).initialize()["initialized"] is True
    identify(root)
    respelled = tmp_path / "CaseTest"
    if not respelled.is_dir():
        pytest.skip("needs a case-insensitive file system")
    _, path = add_fact(respelled)
    history = GitHistory(respelled)
    # The premise: the path keeps the spelling it was given, unlike Git's answer.
    assert history.root.name == "CaseTest"
    assert history.initialize() == {"initialized": False, "root": str(respelled.resolve())}
    assert history.checkpoint("Respelled root")["committed"]
    assert [entry["message"] for entry in history.log()] == ["Respelled root"]
    assert tree(root) == [path]


def test_reads_go_on_while_a_checkpoint_commits_and_writes_wait(tmp_path, history, monkeypatch):
    _, first = add_fact(tmp_path)
    # Opening reads without the write lock when the files are as indexed and were
    # verified after they last changed: the usual state a few seconds after a write.
    with closing(sqlite3.connect(str(tmp_path / ".memoryos" / "index.sqlite3"))) as index:
        index.execute("UPDATE sources SET verified_ns = mtime_ns + 60000000000")
        index.commit()
    staging, resume = paused_commits(monkeypatch)
    checkpointer, checkpointed = in_thread(lambda: history.checkpoint("Paused"))

    def read():
        with MemoryStore(tmp_path) as store:
            hits = HybridRetriever(store, embedder=HashingEmbedder()).search(
                "Alex home city", as_of="2025-02-01")
            return ([fact.value for fact in store.snapshot(as_of="2025-02-01")],
                    [hit.fact.value for hit in hits])

    try:
        assert staging.wait(WAIT)
        writer, written = in_thread(lambda: add_fact(tmp_path, "Acme", "employer"))
        reader, read_result = in_thread(read)
        reader.join(5)  # Far longer than a read takes; a blocked open waits 30 s.
        assert read_result == [(["Riverton"], ["Riverton"])]
        assert writer.is_alive() and written == []  # The write waits for the checkpoint.
    finally:
        resume.set()
    checkpointer.join(WAIT)
    writer.join(WAIT)
    assert len(checkpointed) == 1 and checkpointed[0]["committed"], checkpointed
    assert len(written) == 1 and not isinstance(written[0], BaseException), written
    assert tree(tmp_path) == [first]
