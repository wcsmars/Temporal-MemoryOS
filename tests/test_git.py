import shutil
import subprocess

import pytest

from temporal_memory_os.git_history import GitHistory


pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="Git unavailable")


def git(root, *arguments):
    return subprocess.run(["git", "-C", str(root), *arguments],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_checkpoint_only_contains_memory_and_preserves_other_staging(tmp_path):
    history = GitHistory(tmp_path)
    assert history.initialize()["initialized"]
    assert not history.initialize()["initialized"]
    git(tmp_path, "config", "user.name", "MemoryOS Test")
    git(tmp_path, "config", "user.email", "memoryos-test@example.invalid")
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


def test_git_checkpoint_refuses_symlinks(tmp_path):
    history = GitHistory(tmp_path)
    history.initialize()
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "outside").symlink_to(tmp_path / "outside-file")
    with pytest.raises(ValueError, match="symbolic links"):
        history.checkpoint("Cannot include links")
