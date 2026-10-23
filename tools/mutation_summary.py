#!/usr/bin/env python3
"""Summarize a finished ``mutmut run``: score, and which survivors matter.

A surviving mutant that only changes the text or letter case of a string
literal (an error message, an SQL keyword) cannot change behavior that the
tests are meant to pin, so it is counted separately. Every other survivor is
listed by function for review. Run from the repository root after:

    HYPOTHESIS_PROFILE=mutation mutmut run
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys

LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')


def mutmut(*arguments: str) -> str:
    return subprocess.run([sys.executable, "-m", "mutmut", *arguments], capture_output=True,
                          text=True, check=True).stdout


def text_only(diff: str) -> bool:
    removed = [line[2:] for line in diff.splitlines() if line.startswith("- ")]
    added = [line[2:] for line in diff.splitlines() if line.startswith("+ ")]
    if len(removed) != 1 or len(added) != 1:
        return False
    if LITERAL.sub('""', removed[0]) != LITERAL.sub('""', added[0]):
        return False
    pairs = zip(LITERAL.findall(removed[0]), LITERAL.findall(added[0]))
    return all(old.lower() == new.lower() or new.strip("\"'").startswith("XX")
               for old, new in pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print each behavioral survivor.")
    args = parser.parse_args()
    statuses = collections.Counter()
    survivors = []
    for line in mutmut("results", "--all", "true").splitlines():
        name, _, status = line.strip().rpartition(": ")
        if name:
            statuses[status] += 1
            if status == "survived":
                survivors.append(name)
    total = sum(statuses.values())
    caught = statuses["killed"] + statuses["timeout"]
    behavioral = [name for name in survivors if not text_only(mutmut("show", name))]
    print("mutants: {}  caught: {} ({:.1%})  survived: {}".format(
        total, caught, caught / total if total else 0, len(survivors)))
    print("survivors that only change string text or case: {}".format(
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
