"""Tests for the benchmark scripts in benchmarks/.

The scripts are not a package, so they are loaded from their files. Every test
here is fast: corpora have a handful of records and nothing sleeps.
"""

import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
BENCHMARKS = REPOSITORY / "benchmarks"
RESULTS = BENCHMARKS / "results"


def _load(name):
    spec = importlib.util.spec_from_file_location("benchmark_" + name, BENCHMARKS / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = _load("run")
scale = _load("scale")
plot = _load("plot")


# run.py


def _committed_accuracy_report():
    return json.loads((RESULTS / "synthetic-50.json").read_text(encoding="utf-8"))


def test_run_check_passes_the_committed_report():
    assert run.check(_committed_accuracy_report()) == []


@pytest.mark.parametrize("change, expected", [
    (lambda r: r["results"]["late_corrections"]["temporal_hybrid"].update(correct=99),
     "late_corrections: temporal_hybrid answered 99 of 100"),
    (lambda r: r["results"]["state_changes"]["all_records_keyword"].update(correct=51),
     "state_changes: all_records_keyword answered 51 of 150"),
    (lambda r: r["results"]["late_corrections"]["valid_time_only"].update(correct=100),
     "late_corrections: valid_time_only answered 100 of 100"),
    (lambda r: r["results"]["state_changes"]["temporal_hybrid"].update(stale_hit_rate=0.01),
     "state_changes: temporal_hybrid stale-hit rate 0.01"),
    (lambda r: r["results"]["state_changes"]["temporal_hybrid"].update(retrieval_coverage=0.99),
     "coverage 0.99"),
    (lambda r: r["results"]["late_corrections"].pop("latest_state_keyword"),
     "late_corrections: no result for latest_state_keyword"),
])
def test_run_check_flags_a_changed_report(change, expected):
    report = _committed_accuracy_report()
    change(report)
    problems = run.check(report)
    assert len(problems) == 1
    assert expected in problems[0]


def test_run_check_passes_a_fresh_small_run():
    report = run.run(entities=2)
    assert run.check(report) == []
    assert report["corpus"]["state_changes"]["queries"] == 6
    assert report["corpus"]["late_corrections"]["queries"] == 4


def test_run_main_check_exits_zero_and_writes_the_report(tmp_path, capsys):
    output = tmp_path / "report.json"
    run.main(["--entities", "2", "--check", "--output", str(output)])
    assert run.check(json.loads(output.read_text(encoding="utf-8"))) == []
    assert "Benchmark check passed." in capsys.readouterr().err


def test_nearest_rank_p95_of_100_values_is_the_95th():
    values = list(range(1, 101))
    random.Random(7).shuffle(values)
    assert run.nearest_rank(values, 95) == 95


@pytest.mark.parametrize("count, rank", [(1, 1), (2, 2), (19, 19), (20, 19), (21, 20), (250, 238)])
def test_nearest_rank_is_the_smallest_value_covering_the_share(count, rank):
    assert run.nearest_rank(range(1, count + 1), 95) == rank
    assert run.nearest_rank(range(1, count + 1), 100) == count


@pytest.mark.parametrize("values, percent", [([], 95), ([1.0], 0), ([1.0], 101)])
def test_nearest_rank_rejects_bad_input(values, percent):
    with pytest.raises(ValueError):
        run.nearest_rank(values, percent)


# scale.py


def test_scale_target_maps_search_index_to_entity_and_record():
    # 300 records hold 100 entities; search 1 asks for entity 7919 % 100 = 19, whose
    # records are numbers 57, 58 and 59 and whose second state has the ID 59.
    assert scale._target(300, 1) == ("candidate00019", "{:032x}".format(59))
    assert scale._target(300, 0) == ("candidate00000", "{:032x}".format(2))


def _hit(entity, value, record):
    return SimpleNamespace(fact=SimpleNamespace(entity=entity, value=value, id=record))


def test_scale_check_accepts_only_the_expected_entity_value_and_record():
    record = "{:032x}".format(59)
    scale._check([_hit("candidate00019", scale.EXPECTED_VALUE, record)], "candidate00019", record)
    wrong = [
        [],
        [_hit("candidate00000", scale.EXPECTED_VALUE, record)],
        [_hit("candidate00019", "pending", record)],
        [_hit("candidate00019", scale.EXPECTED_VALUE, "{:032x}".format(2))],
    ]
    for hits in wrong:
        with pytest.raises(AssertionError, match="candidate00019"):
            scale._check(hits, "candidate00019", record)


@pytest.mark.parametrize("argv, expected", [
    (["--use-installed"], True),
    (["--use-inst"], True),
    (["--use"], True),
    (["--sizes", "9", "--use-i", "--label", "v0.1"], True),
    ([], False),
    (["--sizes", "9", "--label", "use-installed"], False),
])
def test_scale_early_parse_accepts_abbreviations(argv, expected):
    assert scale._wants_installed(argv) is expected


# mutmut runs tests from a copied folder where "the importable package" is that
# copy, so this check of an installed package only means something outside it.
@pytest.mark.skipif("MUTANT_UNDER_TEST" in os.environ, reason="not meaningful during mutation runs")
def test_scale_script_with_an_abbreviated_flag_measures_the_importable_package(tmp_path):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPOSITORY / "src"), environment.get("PYTHONPATH")]))
    completed = subprocess.run(
        [sys.executable, str(BENCHMARKS / "scale.py"), "--use-inst", "--sizes", "3",
         "--writes", "1", "--searches", "1", "--settle", "0"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["package"]["use_installed"] is True
    assert report["results"][0]["facts"] == 3


def test_scale_tiny_run_completes(tmp_path, capsys):
    output = tmp_path / "scale.json"
    scale.main(["--sizes", "9", "--writes", "1", "--searches", "2", "--settle", "0",
                "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["parameters"] == {"sizes": [9], "writes": 1, "searches": 2,
                                    "settle_seconds": 0.0}
    assert report["package"]["use_installed"] is False
    (row,) = report["results"]
    assert row["facts"] == 9
    for metric in plot.SCALE_METRICS:
        assert row[metric] > 0
    assert "excluding interpreter start-up" in report["metrics"]["cold_search_ms_median"]


@pytest.mark.parametrize("broken", ["warm", "cold"])
def test_scale_rejects_an_answer_for_another_entity(monkeypatch, broken):
    """The warm retriever is created first; every cold search creates a new one."""
    created = []

    class OtherEntity(scale.HybridRetriever):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)
            self.broken = (len(created) == 1) == (broken == "warm")

        def search(self, query, **kwargs):
            if self.broken:
                query = "candidate00000 application_status"
            return super().search(query, **kwargs)

    monkeypatch.setattr(scale, "HybridRetriever", OtherEntity)
    # Search 0 asks for candidate00000, so the first wrong answer is for search 1.
    with pytest.raises(AssertionError, match="searched for candidate00002 and retrieved "
                                             "candidate00000"):
        scale.measure(9, 1, 2, 0)
    assert len(created) == (1 if broken == "warm" else 3)


# plot.py


def test_plot_reproduces_the_committed_charts_byte_for_byte():
    rendered = plot.charts(RESULTS)
    assert set(rendered) == {path.name for path in RESULTS.glob("*.svg")}
    for name, svg in rendered.items():
        assert (RESULTS / name).read_bytes() == svg.encode("utf-8"), name


def test_plot_check_passes_on_the_committed_results(capsys):
    plot.main(["--check"])
    assert capsys.readouterr().err == ""


@pytest.fixture
def results_copy(tmp_path):
    folder = tmp_path / "results"
    shutil.copytree(RESULTS, folder)
    return folder


def _edit(folder, name, change):
    path = folder / name
    report = json.loads(path.read_text(encoding="utf-8"))
    change(report)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_plot_check_flags_charts_that_no_longer_match_the_reports(results_copy, capsys):
    _edit(results_copy, "scale-current.json", lambda r: r["parameters"].update(searches=30))
    with pytest.raises(SystemExit) as raised:
        plot.main(["--results", str(results_copy), "--check"])
    assert raised.value.code == 1
    assert "scale-search.svg, scale-search-cold.svg" in capsys.readouterr().err


@pytest.mark.parametrize("name, change, message", [
    ("synthetic-50.json", lambda r: r.pop("corpus"), "it has no 'corpus' section"),
    ("synthetic-50.json", lambda r: r["corpus"]["state_changes"].pop("entities"),
     "'corpus' has no 'entities' count for 'state_changes'"),
    ("synthetic-50.json", lambda r: r["results"]["late_corrections"].pop("temporal_hybrid"),
     "'late_corrections' has no 'correct' and 'queries' counts for 'temporal_hybrid'"),
    ("scale-current.json", lambda r: r.pop("parameters"),
     "it does not record its 'writes' parameter"),
    ("scale-v0.1.json", lambda r: r["results"][0].pop("cold_search_ms_median"),
     "the result for 300 records has no positive 'cold_search_ms_median'"),
    ("scale-v0.1.json", lambda r: r["results"][1].update(write_ms_median=0),
     "the result for 1000 records has no positive 'write_ms_median'"),
    ("scale-current.json", lambda r: r.update(results=[]), "'results' is not a non-empty list"),
])
def test_plot_names_the_file_and_field_of_a_malformed_report(results_copy, name, change,
                                                              message):
    _edit(results_copy, name, change)
    with pytest.raises(ValueError) as raised:
        plot.charts(results_copy)
    assert str(raised.value).startswith(name + " cannot be charted: " + message)


def test_plot_reports_a_missing_or_unreadable_report_without_a_traceback(results_copy, capsys):
    (results_copy / "scale-v0.1.json").unlink()
    with pytest.raises(SystemExit) as raised:
        plot.main(["--results", str(results_copy), "--check"])
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "error: scale-v0.1.json was not found" in error
    assert "Traceback" not in error

    (results_copy / "scale-v0.1.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="scale-v0.1.json cannot be read as JSON"):
        plot.charts(results_copy)
    (results_copy / "scale-v0.1.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="scale-v0.1.json cannot be charted: it is not a JSON"):
        plot.charts(results_copy)


def test_plot_spread_keeps_order_spacing_and_bounds():
    assert plot._spread([100.0, 101.0, 102.0], 32, 76, 300) == [100.0, 132.0, 164.0]
    # Labels pushed past the bottom move back up, still 32 apart, in their original order.
    assert plot._spread([299.0, 290.0], 32, 76, 300) == [300.0, 268.0]
    assert plot._spread([], 32, 76, 300) == []


def test_evaluate_reports_the_nearest_rank_p95(monkeypatch):
    # The helper alone is not enough: evaluate() must use it for the report.
    latencies = list(range(1, 101))
    random.Random(3).shuffle(latencies)
    clock = iter(value for ms in latencies for value in (0.0, ms / 1000))
    monkeypatch.setattr(run.time, "perf_counter", lambda: next(clock))
    questions = [{"query": "q", "entity": "e", "attribute": "a", "as_of": "2025-01-01",
                  "expected_id": "x", "expected_value": "v"}] * 100
    result = run.evaluate(questions, lambda question: None)
    assert result["latency_ms_p95"] == 95.0
    assert result["latency_ms_median"] == 50.5


def test_plot_charts_values_that_are_all_the_same_power_of_ten(results_copy):
    """A chart whose values all equal one power of ten still renders."""
    def flatten(report):
        for row in report["results"]:
            row.update({metric: 10.0 for metric in plot.SCALE_METRICS})

    _edit(results_copy, "scale-current.json", flatten)
    _edit(results_copy, "scale-v0.1.json", flatten)
    rendered = plot.charts(results_copy)
    assert set(rendered) == {path.name for path in RESULTS.glob("*.svg")}
    assert "nan" not in rendered["scale-write.svg"]
