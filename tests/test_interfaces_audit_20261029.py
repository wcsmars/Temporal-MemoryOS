"""Checkpoint regressions from the interface audit."""

import shutil
import subprocess

import pytest

from temporal_memory_os.git_history import GitHistory
from temporal_memory_os.store import MemoryStore


pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="Git unavailable")


def git(root, *arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments], text=True,
                            capture_output=True, check=True)
    return result.stdout.strip()


def identify(root):
    git(root, "config", "user.name", "MemoryOS Test")
    git(root, "config", "user.email", "memoryos-test@example.invalid")


@pytest.mark.parametrize("topic", ["nested", ""])
@pytest.mark.parametrize("git_directory", [True, False])
def test_checkpoint_refuses_embedded_repository_without_staging(tmp_path, topic, git_directory):
    history = GitHistory(tmp_path)
    history.initialize()
    identify(tmp_path)
    with MemoryStore(tmp_path) as store:
        fact = store.add("Ada", "drink", "tea", topic=topic or "general")
        fact_path = store.path_for(fact.id)
    nested = tmp_path / "memory" / topic
    if git_directory:
        git(nested, "init", "--quiet")
    else:
        # A linked worktree uses a .git file rather than a directory.
        external = tmp_path / "nested-git-metadata"
        git(nested, "init", "--quiet", "--separate-git-dir", str(external))
    identify(nested)
    git(nested, "add", ".")
    git(nested, "commit", "--quiet", "-m", "Inner record")
    (tmp_path / "unrelated.txt").write_text("keep staged", encoding="utf-8")
    git(tmp_path, "add", "unrelated.txt")
    before = (tmp_path / ".git" / "index").read_bytes()
    with pytest.raises(ValueError, match="nested Git repositories"):
        history.checkpoint("Must include facts, not just a repository pointer")
    assert (tmp_path / ".git" / "index").read_bytes() == before
    assert fact_path.is_file()
    assert not git(tmp_path, "ls-files", "--", "memory")


def test_checkpoint_refuses_uninitialized_gitlink(tmp_path):
    history = GitHistory(tmp_path)
    history.initialize()
    identify(tmp_path)
    with MemoryStore(tmp_path) as store:
        store.add("Ada", "drink", "tea")
    assert history.checkpoint("Initial")["committed"]
    head = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "update-index", "--add", "--cacheinfo", "160000", head, "memory/nested")
    before = (tmp_path / ".git" / "index").read_bytes()
    with pytest.raises(ValueError, match="Git submodules"):
        history.checkpoint("Must not commit a gitlink")
    assert git(tmp_path, "rev-parse", "HEAD") == head
    assert (tmp_path / ".git" / "index").read_bytes() == before


def test_checkpoint_detects_git_metadata_in_other_case_on_case_insensitive_disks(tmp_path):
    history = GitHistory(tmp_path)
    history.initialize()
    identify(tmp_path)
    with MemoryStore(tmp_path) as store:
        store.add("Ada", "drink", "tea", topic="nested")
    nested = tmp_path / "memory" / "nested"
    git(nested, "init", "--quiet")
    (nested / ".git").rename(nested / ".GIT")
    if not (nested / ".git").exists():
        pytest.skip("The filesystem is case sensitive")
    with pytest.raises(ValueError, match="nested Git repositories"):
        history.checkpoint("Must not miss an embedded repository")
    assert not git(tmp_path, "ls-files", "--", "memory")
