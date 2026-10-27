"""Explicit local Git checkpoints for the editable memory and approval audit."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Git must use an identity someone configured (in Git's config files or the
# standard GIT_AUTHOR_*/GIT_COMMITTER_* variables), never one it guesses from
# the account name and hostname.
_EXPLICIT_IDENTITY = ("-c", "user.useConfigOnly=true")
# Variables that point Git at a different repository, index or object store.
# A caller's shell (a Git hook, an IDE terminal) may export them; they must
# never redirect private memory into another repository.
_LOCATION_VARIABLES = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE", "GIT_PREFIX",
    # Pathspec modes change how "memory" and "proposals" are matched.
    "GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS",
)
_MARKER_KEY = "memoryos.checkpoints"
_EXCLUDE_HEADER = "# Temporal MemoryOS derived/local files"
# Dot-named files are editor locks, swap files and interrupted writes; the
# store ignores them, so checkpoints do too. Temporary files and editor backups
# ("notes.md~", "#notes.md#") are left out as well. "!*.sqlite3*/" and "!*.tmp/"
# keep topic folders named like "app.sqlite3" or "notes.tmp" in: the file rules
# alone would hide their records.
_EXCLUDES = (".memoryos/", "*.sqlite3*", "!*.sqlite3*/", ".env", "proposals/.lock", "*.tmp",
             "!*.tmp/", "/memory/**/.*", "/proposals/.*", "/memory/**/*~", "/memory/**/#*#")
# Your personal ignore file (core.excludesFile, or ~/.config/git/ignore) often
# lists names such as build/ or logs/. Staging must not let it hide records in
# topic folders with those names, so only the rules above apply.
_OWN_EXCLUDES_ONLY = ("-c", "core.excludesFile=")
_TARGETS = ("memory", "proposals")
_TIMEOUT = 600


class GitHistory:
    """Never stages a parent repository, pushes, or changes global Git settings."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def _git(self, *arguments: str, check: bool = True, stdin: Optional[str] = None
             ) -> subprocess.CompletedProcess:
        if not shutil.which("git"):
            raise RuntimeError("Git is not installed; install Git to use checkpoints.")
        environment: Dict[str, str] = {key: value for key, value in os.environ.items()
                                       if key not in _LOCATION_VARIABLES}
        try:
            result = subprocess.run(
                ["git", "-C", str(self.root), *arguments], input=stdin, env=environment,
                text=True, capture_output=True, timeout=_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            # A killed Git command can leave .git/index.lock behind. It is not
            # removed here: the lock could belong to a Git command you started.
            raise RuntimeError(
                "Git did not finish within {} seconds. If no other Git command is running in "
                "this memory directory, delete .git/index.lock there before the next "
                "checkpoint.".format(_TIMEOUT)) from exc
        if check and result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Git failed.")
        return result

    def _toplevel(self) -> Optional[Path]:
        result = self._git("rev-parse", "--show-toplevel", check=False)
        return None if result.returncode else Path(result.stdout.strip()).resolve()

    def _owned(self) -> bool:
        """True for a repository MemoryOS created, or a clone of one."""
        if self._git("config", "--local", "--get", _MARKER_KEY, check=False).stdout.strip() == "true":
            return True
        exclude = self._exclude_path()
        if exclude.is_file() and _EXCLUDE_HEADER in exclude.read_text(encoding="utf-8"):
            return True  # Created before the marker existed.
        tracked = [name for name in self._git("ls-files", "-z").stdout.split("\x00") if name]
        return bool(tracked) and all(name.split("/", 1)[0] in _TARGETS for name in tracked)

    def _exclude_path(self) -> Path:
        location = self._git("rev-parse", "--git-path", "info/exclude").stdout.strip()
        path = Path(location)
        return path if path.is_absolute() else self.root / path

    def _adopt(self) -> None:
        """Mark the repository as MemoryOS's and make sure local files stay out of it.

        The MemoryOS block of the exclude file always holds every rule in the
        order above, because Git applies the last rule that matches: a rule
        added by an upgrade must not land after the dot-file and backup rules.
        Lines before the block and your own lines after it are kept. Nothing is
        written when both are already in place, so concurrent checkpoints do
        not compete for Git's config lock.
        """
        if self._git("config", "--local", "--get", _MARKER_KEY, check=False).stdout.strip() != "true":
            self._git("config", "--local", _MARKER_KEY, "true")
        exclude = self._exclude_path()
        exclude.parent.mkdir(parents=True, exist_ok=True)
        present = exclude.read_text(encoding="utf-8").splitlines() if exclude.is_file() else []
        if _EXCLUDE_HEADER in present:
            cut = present.index(_EXCLUDE_HEADER)
            before = present[:cut]
            own = [line for line in present[cut + 1:]
                   if line not in _EXCLUDES and line != _EXCLUDE_HEADER]
        else:
            before, own = present + [""], []
        lines = before + [_EXCLUDE_HEADER, *_EXCLUDES] + own
        if lines != present:
            exclude.write_text("\n".join(lines).strip("\n") + "\n", encoding="utf-8")

    def _is_root(self, path: Optional[Path]) -> bool:
        # samefile: on case-insensitive file systems Git may spell the folder differently.
        try:
            return path is not None and os.path.samefile(path, self.root)
        except OSError:
            return False

    def _require_own_repository(self) -> None:
        toplevel = self._toplevel()
        if not self._is_root(toplevel):
            raise ValueError("This memory directory needs its own repository. Run git-init first.")
        if not self._owned():
            raise ValueError(
                "This repository was not created by `memoryos git-init`. If it holds only this "
                "memory directory and you want checkpoints committed to it, mark it with: git -C "
                "<memory directory> config memoryos.checkpoints true")

    def initialize(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        toplevel = self._toplevel()
        if toplevel is not None:
            if not self._is_root(toplevel):
                raise ValueError(
                    "The memory directory is inside another Git repository. Choose a separate "
                    "directory outside it before enabling memory checkpoints.")
            if not self._owned():
                raise ValueError(
                    "This directory is already a Git repository that MemoryOS did not create "
                    "(for example a project repository). Use a separate memory directory, or, if "
                    "this repository holds only memory, mark it with: git -C <memory directory> "
                    "config memoryos.checkpoints true")
            self._adopt()
            return {"initialized": False, "root": str(self.root)}
        self._git("init", "--quiet")
        # These settings are repository-local and contain no identity information.
        self._adopt()
        return {"initialized": True, "root": str(self.root)}

    @staticmethod
    def _publishable(name: str) -> bool:
        # Dot files are editor locks, swap files or interrupted writes; the
        # store ignores them too.
        return bool(name) and not any(part.startswith(".") for part in name.split("/"))

    def checkpoint(self, message: str) -> dict:
        if not message.strip():
            raise ValueError("A non-empty checkpoint message is required.")
        self._require_own_repository()
        targets: List[str] = []
        for name in _TARGETS:
            path = self.root / name
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise ValueError(f"Checkpoint target {name!r} must be a directory.")
            if path.exists():
                if any(entry.is_symlink() for entry in path.rglob("*")
                       if self._publishable(entry.relative_to(self.root).as_posix())):
                    raise ValueError("Checkpoint directories cannot contain symbolic links.")
            targets.append(name)
        # Check the effective identity before altering the index. Environment identities
        # are supported; missing identity is never guessed or written to Git config.
        identity = self._git(*_EXPLICIT_IDENTITY, "var", "GIT_AUTHOR_IDENT", check=False)
        committer = self._git(*_EXPLICIT_IDENTITY, "var", "GIT_COMMITTER_IDENT", check=False)
        if identity.returncode or committer.returncode:
            raise ValueError(
                "Configure user.name and user.email in Git (globally or in this memory "
                "repository) before checkpointing, or provide Git author/committer "
                "environment variables.")
        # Hold the memory write lock and the proposal lock while staging, so a
        # checkpoint never captures an approved proposal without its fact. Reads
        # and searches go on meanwhile; writes and approvals wait for it.
        from .extraction import ProposalQueue
        from .store import MemoryStore
        with MemoryStore(self.root) as store:
            queue = ProposalQueue(store)
            with queue._lock(), store._transaction():
                self._adopt()  # Inside the locks, so checkpoints never race on it.
                return self._commit(message, targets)

    def _commit(self, message: str, targets: List[str]) -> dict:
        # Stage whole folders: the repository's exclude rules keep locks, temporary
        # and dot files out. Per-file pathspecs would cost time quadratic in the
        # number of records.
        def known(name: str) -> bool:
            return bool(self._git("ls-files", "--", name).stdout.strip())

        # A folder that is absent and was never committed has nothing to stage.
        tracked = {name for name in targets if known(name)}
        present = [name for name in targets if (self.root / name).is_dir() or name in tracked]
        if present:
            self._git(*_OWN_EXCLUDES_ONLY, "add", "--all", "--", *present)
        # Commit folders that had or now have tracked files (a deletion counts).
        targets = [name for name in present if name in tracked or known(name)]
        if not targets:
            return {"committed": False, "reason": "No memory changes."}
        changes = self._git("diff", "--cached", "--quiet", "--", *targets, check=False)
        if changes.returncode == 0:
            return {"committed": False, "reason": "No memory changes."}
        if changes.returncode != 1:
            raise RuntimeError(changes.stderr.strip() or "Unable to inspect staged memory changes.")
        # --only keeps any unrelated, already-staged files out of this commit.
        self._git(*_EXPLICIT_IDENTITY, "commit", "--only", "--quiet", "-m", message, "--",
                  *targets)
        commit = self._git("rev-parse", "HEAD").stdout.strip()
        return {"committed": True, "commit": commit, "message": message}

    def log(self, limit: int = 20) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("Log limit must be between 1 and 1000.")
        self._require_own_repository()
        if self._git("rev-parse", "--verify", "HEAD", check=False).returncode:
            return []
        # NUL separates arbitrary commit messages safely without interpreting them.
        result = self._git(
            "log", f"-{limit}", "--format=%H%x00%aI%x00%s%x00", "--", *_TARGETS
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
