"""A completed process must not turn missing test mappings into a valid score."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("status", ["no tests", "skipped"])
def test_no_test_mutants_block_summary_even_when_process_finished(monkeypatch, capsys, status):
    path = Path(__file__).resolve().parents[1] / "tools" / "mutation_summary.py"
    spec = importlib.util.spec_from_file_location("mutation_summary_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def results(*args):
        calls.append(args)
        return f"core.one: killed\ncore.two: {status}\ncore.three: survived\n"

    monkeypatch.setattr(module, "mutmut", results)
    monkeypatch.setattr(sys, "argv", ["mutation_summary.py"])
    assert module.main() == 1
    output = capsys.readouterr()
    assert not output.out
    assert "1 of 3 mutants were not checked" in output.err
    assert ("Recollect test mappings" if status == "no tests" else "Skipped mutants") in output.err
    assert calls == [("results", "--all", "true")]
