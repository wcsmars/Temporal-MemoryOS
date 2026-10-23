import json
import os
import subprocess
import sys
from pathlib import Path


def invoke(root, *args):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-m", "temporal_memory_os", "--root", str(root), *args],
        capture_output=True, text=True, env=environment, timeout=30,
    )


def test_demo_is_repeatable_and_historical_query_works(tmp_path):
    first = invoke(tmp_path, "demo")
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["added"] == 7
    second = invoke(tmp_path, "demo")
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["added"] == 0
    historical = invoke(tmp_path, "search", "Acme application status",
                        "--as-of", "2026-09-10", "--entity", "Acme application")
    assert historical.returncode == 0, historical.stderr
    result = json.loads(historical.stdout)
    assert result[0]["fact"]["value"] == "pending"
    assert result[0]["fact"]["source"].startswith("synthetic:")


def test_cli_invalid_input_fails_without_traceback(tmp_path):
    result = invoke(tmp_path, "add", "entity", "attribute", "value", "--from", "bad-date")
    assert result.returncode == 1
    assert "error" in json.loads(result.stderr)
    assert not result.stdout


def test_cli_extraction_requires_approval(tmp_path):
    result = invoke(tmp_path, "extract", "--text", "Alex | favorite color | blue",
                    "--source", "synthetic:test", "--observed-at", "2026-01-01")
    assert result.returncode == 0, result.stderr
    proposals = json.loads(result.stdout)
    assert len(proposals) == 1
    assert json.loads(invoke(tmp_path, "init").stdout)["facts"] == 0
    approval = invoke(tmp_path, "approve", proposals[0]["id"])
    assert approval.returncode == 0, approval.stderr
    assert json.loads(approval.stdout)["value"] == "blue"
    assert json.loads(invoke(tmp_path, "init").stdout)["facts"] == 1
