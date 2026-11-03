"""The shell examples in the documentation hold no comments and no placeholders in angle
brackets, which stop them when pasted into zsh, and the 0.1 baseline example stops until
its folder is set.

zsh, the default shell on macOS, does not treat ``#`` as the start of a comment when it
is used interactively (unless INTERACTIVE_COMMENTS is set), so a note after a command
reaches the command as more words, and a parenthesis or backquote in a note line can
stop a whole pasted block. Every shell reads a placeholder in angle brackets as a
redirection. Notes therefore go in the text around each block, and placeholders are
words in capitals, such as FACT_ID, or a variable the reader sets first.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DOCUMENTS = ["README.md", "CHANGELOG.md", "benchmarks/README.md", "docs/architecture.md",
             "docs/validation.md"]
SHELLS = {"sh", "bash", "shell", "zsh"}
# An opening or closing code fence, at any indentation, and the first word after it.
FENCE = re.compile(r"\s*(`{3,}|~{3,})\s*([^\s`]*)")
# Quoted text is replaced by a stand-in letter, so a # inside quotes, or one that a
# closing quote touches, stays part of a word.
QUOTED = re.compile(r"'[^']*'|" + r'"(?:[^"\\]|\\.)*"')
# Where bash starts a comment: at the start of a word, after a space, an operator, a
# redirection or a backquote.
COMMENT = re.compile(r"(?:^|[\s;&|()<>`])#")
# <memory directory>, <id>; not a redirection such as <in.json >out.json or <<EOF.
PLACEHOLDER = re.compile(r"(?<![<\w])<[A-Za-z][\w ./-]*(?<!\s)>")


def shell_lines(text):
    """(block number, line number, line) for each line inside a shell code block in
    ``text``."""
    fence = None  # The marks that opened the block the line is in, if any.
    shell = False
    block = 0
    for number, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if match:
            marks, word = match.groups()
            if fence is None:
                fence, shell = marks, word.lower() in SHELLS
                block += 1
                continue
            if marks[0] == fence[0] and len(marks) >= len(fence) and not word:
                fence, shell = None, False
                continue
        if shell:
            yield block, number, line


def problems(text):
    """(line number, line) for each line of a shell code block in ``text`` that holds a
    comment or a placeholder in angle brackets outside quotes."""
    found = []
    for _, number, line in shell_lines(text):
        bare = QUOTED.sub("Q", line)
        if COMMENT.search(bare) or PLACEHOLDER.search(bare):
            found.append((number, line))
    return found


def shell_blocks(text):
    """The text of each shell code block in ``text``."""
    blocks = {}
    for block, _, line in shell_lines(text):
        blocks.setdefault(block, []).append(line)
    return ["\n".join(lines) for lines in blocks.values()]


@pytest.mark.parametrize("name", DOCUMENTS)
def test_shell_examples_hold_no_comments_or_placeholders(name):
    assert problems((ROOT / name).read_text(encoding="utf-8")) == []


def test_the_check_finds_comments_and_placeholders_outside_quotes():
    text = "\n".join([
        "Prose may hold # and <tags>.",
        "```sh",
        "memoryos list    # every fact",
        "# a note",
        "memoryos search 'a # b' --entity \"<x>\" --source issue#4",
        "memoryos fetch <id>",
        "echo a;# then b",
        "echo 'a'# one word, as in every shell",
        "echo ${#name} $# <in.json >out.json 2>errors.txt",
        "sort <in.txt >#sorted.txt",
        "echo `#note`",
        "```",
        "```text",
        "# output, not a command",
        "```",
        "~~~bash title=example",
        "true &&# next",
        "~~~",
        "1. A step:",
        "   ```zsh",
        "   memoryos list  # indented in a list item",
        "   ```",
        "```Bash",
        "ls  # a tag in capitals",
        "```",
    ])
    assert problems(text) == [(3, "memoryos list    # every fact"), (4, "# a note"),
                              (6, "memoryos fetch <id>"), (7, "echo a;# then b"),
                              (10, "sort <in.txt >#sorted.txt"), (11, "echo `#note`"),
                              (17, "true &&# next"),
                              (21, "   memoryos list  # indented in a list item"),
                              (24, "ls  # a tag in capitals")]


@pytest.mark.parametrize("shell", ["sh", "bash", "zsh"])
def test_the_0_1_baseline_example_stops_until_its_folder_is_set(shell, tmp_path):
    """Pasted without OLD_PACKAGE_FOLDER, or with it empty, the example that measures the
    0.1 baseline stops with a message before Python starts, so it cannot measure the
    current code and write that over the stored 0.1 report. It is one command, so a shell
    that reads a paste line by line drops the Python call along with the guard. With the
    folder set, it runs with that folder as the import path."""
    if shutil.which(shell) is None:
        pytest.skip(shell + " is not installed")
    blocks = [block for block in shell_blocks((ROOT / "benchmarks/README.md").read_text(
        encoding="utf-8")) if "scale-v0.1.json" in block]
    assert len(blocks) == 1
    # A shell that reads a paste line by line (the bash that ships with macOS, dash)
    # drops only the command that failed and runs the next line, which -c does not
    # show: a guard on a line of its own would let Python start with no import path.
    assert all(line.endswith("\\") for line in blocks[0].splitlines()[:-1])
    tools = tmp_path / "bin"
    tools.mkdir()
    calls = tmp_path / "calls.txt"
    python = tools / "python"  # Stands in for Python and records how it was started.
    python.write_text('#!/bin/sh\nprintf "%s|%s\\n" "$PYTHONPATH" "$*" >> "$CALLS"\n',
                      encoding="utf-8")
    python.chmod(0o755)
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("OLD_PACKAGE_FOLDER", "PYTHONPATH", "ENV", "BASH_ENV")}
    environment.update(PATH=str(tools) + os.pathsep + environment.get("PATH", ""),
                       CALLS=str(calls))
    # -f: zsh reads no start-up files except the system-wide zshenv.
    command = [shell, "-f", "-c", blocks[0]] if shell == "zsh" else [shell, "-c", blocks[0]]

    # Unset, then empty, as `export OLD_PACKAGE_FOLDER=` or a copy of an unset variable
    # leaves it.
    for folder in ({}, {"OLD_PACKAGE_FOLDER": ""}):
        stopped = subprocess.run(command, cwd=tmp_path, env={**environment, **folder},
                                 capture_output=True, text=True, timeout=60)
        assert stopped.returncode != 0
        assert "OLD_PACKAGE_FOLDER: set it to the folder that holds the 0.1 package" in (
            stopped.stderr)
        assert not calls.exists()

    environment["OLD_PACKAGE_FOLDER"] = str(tmp_path / "old")
    ran = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True,
                         text=True, timeout=60)
    assert ran.returncode == 0, ran.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "{}|benchmarks/scale.py --use-installed --label v0.1 --output "
        "benchmarks/results/scale-v0.1.json".format(tmp_path / "old")]
