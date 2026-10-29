#!/usr/bin/env python3
"""Summarize a finished ``mutmut run``: score, and which survivors matter.

A survivor is counted as text-only, and kept out of the review list, only
when its diff changes string literals and nothing else, and every changed
literal is one of these:

* Message text. The literal is in the first argument of ``raise X(...)``
  where the name X ends in Error, Exception or Warning, in the first argument
  of ``warnings.warn(...)``, in ``sys.stderr.write(...)``, or in the
  arguments of a ``print(...)`` statement that passes ``file=sys.stderr``.
  Inside that argument it may sit directly, in grouping parentheses or among
  the arguments of ``.format(...)``. Its ``{}`` and ``%`` placeholders must
  be unchanged, so the message is still built the same way.
* SQL keyword case. The literal is in the first argument (the statement) of
  ``.execute(...)``, ``.executemany(...)`` or ``.executescript(...)``,
  directly, in grouping parentheses or through ``.format(...)`` or
  ``.join(...)``. Only letter case changes, every word whose case changes is
  an SQL keyword or column type name, and quoted SQL strings are unchanged.

The statement around a change is read from the diff's context lines, so a
message continued over several lines counts when the line that opens the
call is in the diff. Every other literal change counts as behavioral and is
listed by function for review: dictionary keys, file and folder names,
default argument values, compared values, format strings whose result is
returned or stored, and anything the rule cannot place. Run from the
repository root after:

    HYPOTHESIS_PROFILE=mutation mutmut run

With no results, or a run that left mutants unchecked, it prints no score
and exits 1.
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from typing import List, Optional, Pattern, Sequence, Tuple

# An open bracket while scanning a statement: (bracket, code before it, comma seen).
Frame = Tuple[str, str, bool]
# A piece of source: (kind, text, line, open brackets), kind "code", "string" or "comment".
Piece = Tuple[str, str, int, Tuple[Frame, ...]]

_RAISE = re.compile(r"(?:^|:)\s*raise\s+[\w.]*(?:Error|Exception|Warning)\s*$")
_WARN = re.compile(r"(?:^|[^\w.])warnings\.warn\s*$")
_STDERR_WRITE = re.compile(r"(?:^|[^\w.])sys\.stderr\.write\s*$")
_PRINT = re.compile(r"(?:^|:)\s*print\s*$")
_TO_STDERR = re.compile(r"\bfile\s*=\s*sys\.stderr\b")
_SQL_CALL = re.compile(r"\.(?:execute|executemany|executescript)\s*$")
_FORMAT = re.compile(r"\.format\s*$")
_JOIN = re.compile(r"\.join\s*$")
# "(" after an operator, a comma or another bracket groups; after a name it calls.
_GROUPING = re.compile(r"[^\w)\]}\"'\s]\s*$")
_PLACEHOLDER = re.compile(
    r"\{[^{}]*\}|%(?:\([^)]*\))?[-#0 +]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[A-Za-z%]")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SQL_STRING = re.compile(r"'(?:[^']|'')*'")
SQL_KEYWORDS = frozenset("""
    ABORT ACTION ADD AFTER ALL ALTER ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE BEGIN BETWEEN
    BY CASCADE CASE CAST CHECK COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS CURRENT_DATE
    CURRENT_TIME CURRENT_TIMESTAMP DATABASE DEFAULT DEFERRABLE DEFERRED DELETE DESC DETACH DISTINCT
    DO DROP EACH ELSE END ESCAPE EXCEPT EXCLUSIVE EXISTS EXPLAIN FAIL FILTER FOR FOREIGN FROM FULL
    GLOB GROUP HAVING IF IGNORE IMMEDIATE IN INDEX INDEXED INITIALLY INNER INSERT INSTEAD INTERSECT
    INTO IS ISNULL JOIN KEY LEFT LIKE LIMIT MATCH NATURAL NO NOT NOTHING NOTNULL NULL OF OFFSET ON
    OR ORDER OUTER PLAN PRAGMA PRIMARY QUERY RECURSIVE REFERENCES REGEXP REINDEX RELEASE RENAME
    REPLACE RESTRICT RETURNING RIGHT ROLLBACK ROW SAVEPOINT SELECT SET TABLE TEMP TEMPORARY
    THEN TO TRANSACTION TRIGGER UNION UNIQUE UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE
    WITH WITHOUT
    BLOB INTEGER NUMERIC REAL TEXT
""".split())


def mutmut(*arguments: str) -> str:
    """Run ``python -m mutmut`` with ``arguments`` and return its output.

    Exits with mutmut's own error message if it fails, for example when it is not
    installed in this interpreter.
    """
    result = subprocess.run([sys.executable, "-m", "mutmut", *arguments], capture_output=True,
                            text=True)
    if result.returncode:
        raise SystemExit("mutmut {} failed:\n{}\nIf mutmut is not installed here, run: "
                         "python -m pip install -e '.[mutation]'".format(
                             arguments[0], (result.stderr or result.stdout).strip()))
    return result.stdout


def _aligned(diff: str) -> Optional[Tuple[List[str], List[str]]]:
    """The original and mutated lines of a one-hunk diff, line for line.

    None when there is more than one hunk or a change adds or removes lines.
    Lines before the hunk header (``mutmut show``'s title, file names) are skipped.
    """
    lines = diff.splitlines()
    headers = [index for index, line in enumerate(lines) if line.startswith("@@")]
    if len(headers) > 1:
        return None
    body = lines[headers[0] + 1:] if headers else [
        line for line in lines if not line.startswith(("--- ", "+++ ", "# "))]
    old: List[str] = []
    new: List[str] = []
    removed: List[str] = []
    added: List[str] = []
    for line in body + [" "]:
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
        elif not line.startswith("\\"):  # "\ No newline at end of file"
            if len(removed) != len(added):
                return None
            old.extend(removed)
            new.extend(added)
            removed, added = [], []
            old.append(line[1:])
            new.append(line[1:])
    return old[:-1], new[:-1]


def _scan(lines: Sequence[str], inside: str = "") -> Tuple[List[Piece], List[Optional[int]]]:
    """Split source lines into code, string and comment pieces.

    Each string piece carries the brackets open around it; the code before
    each bracket shows strings as ``""``. The second result is the bracket
    depth at the start of every line relative to the first line, None for a
    line that starts inside a string, plus one entry for the end (None if the
    text ends inside a string). ``inside`` is the quote of a string that the
    first line starts in, such as the end of a docstring.
    """
    text = inside + "\n".join(lines)
    pieces: List[Piece] = []
    frames: List[List] = []
    flat: List[str] = []
    depths: List[Optional[int]] = [0]
    base = line = index = start = start_line = 0
    open_string = False

    def finish_code(end: int) -> None:
        if end > start:
            pieces.append(("code", text[start:end], start_line, ()))

    while index < len(text):
        char = text[index]
        if char in "\"'":
            finish_code(index)
            quote = text[index:index + 3] if text[index:index + 3] in ('"""', "'''") else char
            end = index + len(quote)
            open_string = True
            while end < len(text):
                if text.startswith(quote, end):
                    end += len(quote)
                    open_string = False
                    break
                if text[end] == "\n" and len(quote) == 1:
                    open_string = False  # Not valid Python; end the string at the line.
                    break
                end += 2 if text[end] == "\\" else 1
            end = min(end, len(text))
            snapshot = tuple((bracket, prefix, comma) for bracket, prefix, comma in frames)
            pieces.append(("string", text[index:end], line, snapshot))
            flat.append('""')
            for _ in range(text.count("\n", index, end)):
                line += 1
                depths.append(None)
            index = start = end
            start_line = line
            continue
        if char == "#":
            finish_code(index)
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            pieces.append(("comment", text[index:end], line, ()))
            index = start = end
            start_line = line
            continue
        if char in "([{":
            frames.append([char, "".join(flat), False])
        elif char in ")]}":
            if frames:
                frames.pop()
            else:
                base -= 1
        elif char == "," and frames:
            frames[-1][2] = True
        elif char == "\n":
            line += 1
            depths.append(base + len(frames))
        flat.append(char)
        index += 1
    finish_code(len(text))
    depths.append(None if open_string else base + len(frames))
    return pieces, depths


def _body(literal: str) -> str:
    """A literal's text without its quotes."""
    quote = literal[:3] if literal[:3] in ('"""', "'''") else literal[:1]
    closed = len(literal) >= 2 * len(quote) and literal.endswith(quote)
    return literal[len(quote):len(literal) - len(quote) if closed else None]


def _argument_of(frames: Sequence[Frame], call: Pattern[str],
                 nested: Sequence[Pattern[str]]) -> Optional[Frame]:
    """The innermost open call matching ``call``, if every bracket opened
    inside it is a "(" whose preceding code matches one of ``nested``."""
    for position in range(len(frames) - 1, -1, -1):
        bracket, prefix, _ = frames[position]
        if bracket == "(" and call.search(prefix):
            inner = frames[position + 1:]
            if all(item[0] == "(" and any(rule.search(item[1]) for rule in nested)
                   for item in inner):
                return frames[position]
            return None
    return None


def _in_message(frames: Sequence[Frame], statement: Optional[str]) -> bool:
    for call in (_RAISE, _WARN):
        found = _argument_of(frames, call, (_FORMAT, _GROUPING))
        if found is not None:
            return not found[2]  # The first argument is the message.
    if _argument_of(frames, _STDERR_WRITE, (_FORMAT, _GROUPING)) is not None:
        return True
    return (_argument_of(frames, _PRINT, (_FORMAT, _GROUPING)) is not None
            and statement is not None and bool(_TO_STDERR.search(statement)))


def _in_sql(frames: Sequence[Frame]) -> bool:
    found = _argument_of(frames, _SQL_CALL, (_FORMAT, _JOIN, _GROUPING))
    return found is not None and not found[2]  # The first argument is the statement.


def _keyword_case(old: str, new: str) -> bool:
    if old == new or old.lower() != new.lower():
        return False
    if _SQL_STRING.findall(old) != _SQL_STRING.findall(new):
        return False
    old_words, new_words = _WORD.findall(old), _WORD.findall(new)
    return len(old_words) == len(new_words) and all(
        before == after or before.upper() in SQL_KEYWORDS
        for before, after in zip(old_words, new_words))


def text_only(diff: str) -> bool:
    """True if a mutant's diff only changes message text or SQL keyword case.

    ``diff`` is the unified diff that ``mutmut show`` prints. The exact rule
    is in the module docstring.
    """
    sides = _aligned(diff)
    if sides is None:
        return False
    old, new = sides
    changed = [index for index, (before, after) in enumerate(zip(old, new)) if before != after]
    if not changed:
        return False
    # A hunk may start inside a docstring; take the reading that ends outside a string.
    for inside in ("", '"""', "'''"):
        _, depths = _scan(old, inside)
        if depths[-1] is not None:
            break
    else:
        return False
    known = [index for index in range(changed[0] + 1) if depths[index] is not None]
    if not known:
        return False
    # The changed line's statement starts on the last line at the lowest depth
    # seen so far (a hunk can start inside an earlier statement's brackets).
    lowest = min(depths[index] or 0 for index in known)
    first = max(index for index in known if depths[index] == lowest)
    old_pieces, statement_depths = _scan(old[first:])
    new_pieces, _ = _scan(new[first:])
    if len(old_pieces) != len(new_pieces):
        return False
    last = changed[-1] - first
    end = next((index for index in range(last + 1, len(statement_depths))
                if statement_depths[index] == 0), None)
    statement = None
    if end is not None:
        statement = "".join('""' if kind == "string" else text
                            for kind, text, line, _ in _scan(old[first:first + end])[0]
                            if kind != "comment")
    found = False
    for (kind, before, line, frames), (new_kind, after, _, _) in zip(old_pieces, new_pieces):
        if kind != new_kind:
            return False
        if before == after:
            continue
        if kind != "string" or (end is not None and line >= end):
            return False
        old_text, new_text = _body(before), _body(after)
        if _PLACEHOLDER.findall(old_text) != _PLACEHOLDER.findall(new_text):
            return False
        if not (_in_message(frames, statement)
                or (_in_sql(frames) and _keyword_case(old_text, new_text))):
            return False
        found = True
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Print each behavioral survivor.")
    args = parser.parse_args()
    statuses = collections.Counter()
    survivors = []
    interrupted = []
    for line in mutmut("results", "--all", "true").splitlines():
        name, _, status = line.strip().rpartition(": ")
        if name:
            statuses[status] += 1
            if status == "survived":
                survivors.append(name)
            elif status == "check was interrupted by user":
                interrupted.append(name)
    total = sum(statuses.values())
    if not total:
        # mutmut lists nothing, and exits 0, when no run has made mutants in this folder.
        print("No mutmut results here. Run HYPOTHESIS_PROFILE=mutation mutmut run first.",
              file=sys.stderr)
        return 1
    # After a run that stopped early, for example because the tests failed before
    # any mutant was tried, mutmut lists the mutants it did not check and exits 0.
    unchecked = (statuses["not checked"] + statuses["no tests"] + statuses["skipped"]
                 + len(interrupted))
    if unchecked:
        print("The mutmut run did not finish: {} of {} mutants were not checked, so there is "
              "no score.\nCheck that the tests pass, then rerun: "
              "HYPOTHESIS_PROFILE=mutation mutmut run".format(unchecked, total), file=sys.stderr)
        if interrupted:
            print("Interrupted mutants are checked again only when named: "
                  "HYPOTHESIS_PROFILE=mutation mutmut run {}".format(" ".join(interrupted)),
                  file=sys.stderr)
        if statuses["skipped"]:
            print("Skipped mutants were not executed; explicitly rerun or document their "
                  "exclusion before using a score.", file=sys.stderr)
        if statuses["no tests"]:
            print("Some mutants had no associated tests. Recollect test mappings and explicitly "
                  "rerun those mutants; a cached resume can preserve the untested statuses.",
                  file=sys.stderr)
        return 1
    caught = statuses["killed"] + statuses["timeout"]
    behavioral = [name for name in survivors if not text_only(mutmut("show", name))]
    print("mutants: {}  caught: {} ({:.1%})  survived: {}".format(
        total, caught, caught / total, len(survivors)))
    print("survivors that only change message text or SQL keyword case: {}".format(
        len(survivors) - len(behavioral)))
    print("other survivors to review: {}".format(len(behavioral)))
    by_function = collections.Counter(re.sub(r"__mutmut_\d+$", "", name) for name in behavioral)
    for function, count in by_function.most_common():
        print("  {:4d}  {}".format(count, function))
    if args.list:
        for name in behavioral:
            print("\n" + mutmut("show", name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
