"""Explicit local Git checkpoints for the editable memory and approval audit."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


_EXPLICIT_IDENTITY = ("-c", "user.useConfigOnly=true")


class GitHistory:
    """Never stages a parent repository, pushes, or changes global Git settings."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def _git(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
        if not shutil.which("git"):
            raise RuntimeError("Git is not installed; install Git to use checkpoints.")
        result = subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            text=True, capture_output=True, timeout=30,
        )
        if check and result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Git failed.")
        return result

    def _require_own_repository(self) -> None:
        result = self._git("rev-parse", "--show-toplevel", check=False)
        if result.returncode or Path(result.stdout.strip()).resolve() != self.root:
            raise ValueError("This memory directory needs its own repository. Run git-init first.")

    def initialize(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        existing = self._git("rev-parse", "--show-toplevel", check=False)
        if not existing.returncode:
            if Path(existing.stdout.strip()).resolve() != self.root:
                raise ValueError(
                    "The memory directory is inside another Git repository. Choose a separate "
                    "directory outside it before enabling memory checkpoints."
                )
            return {"initialized": False, "root": str(self.root)}
        self._git("init", "--quiet")
        # These defaults are repository-local and contain no identity information.
        exclude = self.root / ".git" / "info" / "exclude"
        with exclude.open("a", encoding="utf-8") as handle:
            handle.write("\n# Temporal MemoryOS derived/local files\n.memoryos/\n*.sqlite3*\n.env\nproposals/.lock\n")
        return {"initialized": True, "root": str(self.root)}

    def checkpoint(self, message: str) -> dict:
        if not message.strip():
            raise ValueError("A non-empty checkpoint message is required.")
        self._require_own_repository()
        targets = []
        for name in ("memory", "proposals"):
            path = self.root / name
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise ValueError(f"Checkpoint target {name!r} must be a directory.")
            if path.exists():
                if any(entry.is_symlink() for entry in path.rglob("*")):
                    raise ValueError("Checkpoint directories cannot contain symbolic links.")
            targets.append(name)
        # Check the effective identity before altering the index. Environment identities
        # are supported; missing identity is never guessed or written to Git config.
        # useConfigOnly stops Git from inventing an identity from the account
        # name and hostname, which would otherwise end up in commit history.
        identity = self._git(*_EXPLICIT_IDENTITY, "var", "GIT_AUTHOR_IDENT", check=False)
        committer = self._git(*_EXPLICIT_IDENTITY, "var", "GIT_COMMITTER_IDENT", check=False)
        if identity.returncode or committer.returncode:
            raise ValueError(
                "Configure user.name and user.email in this memory repository before "
                "checkpointing, or provide Git author/committer environment variables."
            )
        # Enumerate tracked and unignored files so transient proposal files stay
        # excluded even in a repository that predates our local ignore patterns.
        candidates = self._git(
            "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *targets
        ).stdout.split("\x00")
        stage_paths = sorted({name for name in candidates if name
                              and name != "proposals/.lock"
                              and not (name.startswith("proposals/") and name.endswith(".tmp"))})
        if stage_paths:
            self._git("add", "--all", "--", *[":(literal)" + name for name in stage_paths])
        changes = self._git("diff", "--cached", "--quiet", "--", *targets, check=False)
        if changes.returncode == 0:
            return {"committed": False, "reason": "No memory changes."}
        if changes.returncode != 1:
            raise RuntimeError(changes.stderr.strip() or "Unable to inspect staged memory changes.")
        changed_paths = [name for name in self._git(
            "diff", "--cached", "--name-only", "-z", "--", *targets
        ).stdout.split("\x00") if name and name != "proposals/.lock"
            and not (name.startswith("proposals/") and name.endswith(".tmp"))]
        if not changed_paths:
            return {"committed": False, "reason": "No memory changes."}
        # --only keeps any unrelated, already-staged files out of this commit.
        self._git(*_EXPLICIT_IDENTITY, "commit", "--only", "--quiet", "-m", message, "--",
                  *[":(literal)" + name for name in changed_paths])
        commit = self._git("rev-parse", "HEAD").stdout.strip()
        return {"committed": True, "commit": commit, "message": message}

    def log(self, limit: int = 20) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("Log limit must be between 1 and 1000.")
        self._require_own_repository()
        if self._git("rev-parse", "--verify", "HEAD", check=False).returncode:
            return []
        # JSON-escaped subjects from Git are not supported; NUL separates arbitrary
        # commit messages safely without interpreting them as shell commands.
        result = self._git(
            "log", f"-{limit}", "--format=%H%x00%aI%x00%s%x00", "--", "memory", "proposals"
        )
        parts = result.stdout.split("\x00")
        output = []
        for index in range(0, len(parts) - 2, 3):
            output.append({
                "commit": parts[index].strip(),
                "authored_at": parts[index + 1],
                "message": parts[index + 2],
            })
        return output
