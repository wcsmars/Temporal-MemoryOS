"""``memoryos list`` and ``MemoryStore.records`` (0.3.2): what memory holds, as a whole.

``list`` shows the snapshot ``GET /api/facts`` returns: every fact true at one time,
as known at another. ``list --all`` shows every stored record, each one Markdown file:
each entity and attribute pair, every spelling of its names included, is one group,
placed as ``list`` with the same known_at places its facts (without known_at, records
recorded after now count too), and within a group the records come by start, each
followed by the revisions that superseded it.
conftest.py clears every MEMORYOS_* variable before each test.
"""

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from temporal_memory_os import cli
from temporal_memory_os.models import Fact, name_key, normalize_time
from temporal_memory_os.store import MemoryStore

from helpers import MARKER

MISSING = ("Memory directory is missing (moved, renamed or deleted?); move it back, "
           "or run memoryos init with the same --root or MEMORYOS_HOME to start a new, "
           "empty memory")
# The order list --all gives the records made by the memory fixture.
ALL = ("matcha", "Paris", "London", "tea", "coffee", "student", "popcorn", "no snack", "Tokyo")


def run(capsys, *arguments):
    """Call the CLI in this process; return (exit code, parsed stdout, raw stderr)."""
    code = cli.main(list(arguments))
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out.strip() else None, captured.err


def invoke(root, *arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-m", "temporal_memory_os", "--root", str(root), *arguments],
        capture_output=True, text=True, env=environment, timeout=30,
    )


def names(listed):
    return [(fact["entity"], fact["attribute"], fact["value"]) for fact in listed]


def written(root, *facts):
    """Write records by hand, as a person might, into the memory folder at ``root``."""
    folder = root / "memory" / "general"
    folder.mkdir(parents=True, exist_ok=True)
    for fact in facts:
        (folder / (fact.id + ".md")).write_text(
            MARKER + "\n```json\n" + json.dumps(fact.to_dict(), indent=2) + "\n```\n",
            encoding="utf-8")


@pytest.fixture
def memory(tmp_path):
    """Every kind of record, each recorded at a fixed time; return (root, records by name)."""
    root = tmp_path / "TemporalMemory"
    with MemoryStore(root) as store:
        made = {"tea": store.add("user", "drink", "tea", valid_from="2025-01-01",
                                 recorded_at="2025-01-01T01:00:00Z")}
        made["coffee"] = store.add("user", "drink", "coffee", valid_from="2025-06-01",
                                   recorded_at="2025-06-01")
        made["Paris"] = store.add("user", "city", "Paris", valid_from="2025-01-01",
                                  recorded_at="2025-01-01T02:00:00Z")
        made["London"] = store.correct(made["Paris"].id, "London", recorded_at="2025-03-01")
        made["popcorn"] = store.add("user", "snack", "popcorn", valid_from="2025-01-01",
                                    recorded_at="2025-01-01T03:00:00Z")
        made["no snack"] = store.retract(made["popcorn"].id, recorded_at="2025-04-01")
        made["student"] = store.add("user", "job", "student", valid_from="2025-01-01",
                                    valid_to="2025-02-01", recorded_at="2025-01-01T04:00:00Z")
        made["Tokyo"] = store.add("user", "trip", "Tokyo", valid_from="2099-01-01",
                                  recorded_at="2025-01-01T05:00:00Z")
        made["matcha"] = store.add("Alex", "drink", "matcha", valid_from="2025-01-01",
                                   recorded_at="2025-01-01T06:00:00Z")
    return root, made


# list: the facts true at one time ----------------------------------------------------


def test_list_shows_the_facts_true_now(memory, capsys):
    root, made = memory
    code, listed, error = run(capsys, "--root", str(root), "list")
    assert (code, error) == (0, "")
    # Not the tea that coffee replaced, the Paris that London corrected, the retracted
    # popcorn, the job that ended or the trip that starts later.
    assert names(listed) == [("Alex", "drink", "matcha"), ("user", "city", "London"),
                             ("user", "drink", "coffee")]
    with MemoryStore(root, create=False) as store:
        assert listed == [fact.to_dict() for fact in store.snapshot()]


def test_list_gives_what_the_http_api_gives(memory, capsys):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from temporal_memory_os.api import create_app
    root, _ = memory
    # Each option alone too: a cutoff before the job ended tells "true now, as known
    # then" from "true then".
    every = [{}, {"known_at": "2025-01-15"}, {"as_of": "2025-01-15"}, {"entity": "alex"},
             {"as_of": "2025-01-15", "known_at": "2025-02-15", "entity": "USER"}]
    with TestClient(create_app(root), base_url="http://localhost") as client:
        for parameters in every:
            response = client.get("/api/facts", params=parameters)
            assert response.status_code == 200, response.text
            options = [part for key, value in parameters.items()
                       for part in ("--" + key.replace("_", "-"), value)]
            code, listed, _ = run(capsys, "--root", str(root), "list", *options)
            assert code == 0
            assert listed == response.json()["facts"]
            assert listed


def test_list_as_of_shows_the_facts_true_then_with_the_ends_later_records_give(memory, capsys):
    root, _ = memory
    code, listed, _ = run(capsys, "--root", str(root), "list", "--as-of", "2025-01-15")
    assert code == 0
    assert names(listed) == [("Alex", "drink", "matcha"), ("user", "city", "London"),
                             ("user", "drink", "tea"), ("user", "job", "student")]
    assert listed[2]["valid_to"] == normalize_time("2025-06-01")  # Where coffee starts.
    assert listed[3]["valid_to"] == normalize_time("2025-02-01")  # Its own end.


def test_list_known_at_shows_what_memory_knew_then(memory, capsys):
    root, _ = memory
    code, listed, _ = run(capsys, "--root", str(root), "list", "--as-of", "2025-01-15",
                          "--known-at", "2025-02-15")
    assert code == 0
    # Before the correction to London, the retraction and the move to coffee were known.
    assert names(listed) == [("Alex", "drink", "matcha"), ("user", "city", "Paris"),
                             ("user", "drink", "tea"), ("user", "job", "student"),
                             ("user", "snack", "popcorn")]
    assert listed[2]["valid_to"] is None


def test_list_known_at_alone_shows_the_facts_true_now_as_known_then(memory, capsys):
    root, _ = memory
    code, listed, _ = run(capsys, "--root", str(root), "list", "--known-at", "2025-01-15")
    assert code == 0
    # Not the job that had ended by now, though it was still true on 15 January.
    assert names(listed) == [("Alex", "drink", "matcha"), ("user", "city", "Paris"),
                             ("user", "drink", "tea"), ("user", "snack", "popcorn")]


@pytest.mark.parametrize("entity, expected", [
    (" USER ", [("user", "city", "London"), ("user", "drink", "coffee")]),
    ("alex", [("Alex", "drink", "matcha")]),
    ("nobody", []),
])
def test_list_entity_keeps_one_entity_in_any_spelling(memory, capsys, entity, expected):
    root, _ = memory
    code, listed, _ = run(capsys, "--root", str(root), "list", "--entity", entity)
    assert (code, names(listed)) == (0, expected)


# list --all: every stored record -----------------------------------------------------


def test_list_all_shows_every_record_once_each_revision_after_its_target(memory, capsys):
    root, made = memory
    files = [path.stem for path in (root / "memory").rglob("*.md")]
    # A note in the memory folder is not a record (it has no record block).
    (root / "memory" / "general" / "shopping.md").write_text("# Shopping\n\n- milk\n",
                                                             encoding="utf-8")
    code, listed, error = run(capsys, "--root", str(root), "list", "--all")
    assert (code, error) == (0, "")
    # Records as stored: ends are not filled in from later records.
    assert listed == [made[name].to_dict() for name in ALL]
    assert sorted(fact["id"] for fact in listed) == sorted(files)
    assert len(files) == 9


@pytest.mark.parametrize("known_at, left_out", [
    ("2025-03-15", {"coffee", "no snack"}),
    ("2025-03-01", {"coffee", "no snack"}),  # A record recorded at the cutoff is known.
    ("2025-02-28T23:59:59.999999Z", {"coffee", "no snack", "London"}),
    ("2024-12-31", set(ALL)),
])
def test_list_all_known_at_keeps_the_records_recorded_by_then(memory, capsys, known_at,
                                                             left_out):
    root, made = memory
    code, listed, _ = run(capsys, "--root", str(root), "list", "--all", "--known-at", known_at)
    assert code == 0
    assert [fact["id"] for fact in listed] == [made[name].id for name in ALL
                                               if name not in left_out]


def test_list_all_entity_keeps_one_entity_in_any_spelling(memory, capsys):
    root, made = memory
    code, listed, _ = run(capsys, "--root", str(root), "list", "--all", "--entity", "ALEX")
    assert (code, [fact["id"] for fact in listed]) == (0, [made["matcha"].id])
    code, listed, _ = run(capsys, "--root", str(root), "list", "--all", "--entity", "nobody")
    assert (code, listed) == (0, [])


def test_list_all_includes_a_record_imported_with_a_recording_time_ahead_of_the_clock(memory,
                                                                                    capsys):
    root, _ = memory
    with MemoryStore(root, create=False) as store:
        future = store.add("user", "mood", "calm", valid_from="2025-01-01",
                           recorded_at="2999-01-01")
    code, listed, _ = run(capsys, "--root", str(root), "list")
    assert ("user", "mood", "calm") not in names(listed)  # Not known yet.
    code, listed, _ = run(capsys, "--root", str(root), "list", "--all")
    assert future.id in [fact["id"] for fact in listed]
    code, listed, _ = run(capsys, "--root", str(root), "list", "--all", "--known-at",
                          "2026-01-01")
    assert future.id not in [fact["id"] for fact in listed]


def test_list_all_places_revisions_recorded_at_the_same_instant_after_their_target(tmp_path):
    """IDs that sort the other way round must not put a revision before its target."""
    root = tmp_path / "TemporalMemory"
    MemoryStore(root).close()
    first = Fact(id="f" * 32, entity="user", attribute="drink", value="tea",
                 valid_from="2025-01-01", recorded_at="2025-01-01")
    second = Fact(id="5" * 32, entity="user", attribute="drink", value="green tea",
                  valid_from="2025-01-01", recorded_at="2025-01-01", supersedes=first.id)
    third = Fact(id="0" * 32, entity="user", attribute="drink", value="", deleted=True,
                 valid_from="2025-01-01", recorded_at="2025-01-01", supersedes=second.id)
    before = Fact(id="e" * 32, entity="user", attribute="drink", value="water",
                  valid_from="2024-01-01", recorded_at="2025-01-01")
    written(root, first, second, third, before)
    with MemoryStore(root, create=False) as store:
        assert store.records() == [before, first, second, third]
        assert store.records(known_at="2025-01-01") == [before, first, second, third]
        assert store.records(known_at="2024-12-31") == []


def test_list_all_keeps_every_spelling_of_a_name_in_one_history(tmp_path, capsys):
    """Stores written before 0.3.1, by hand, or by an older Python version can spell one
    name several ways. The records of each entity and attribute pair stay together, in
    the order of their starts, as in its history; the pairs of one entity can be apart,
    as in list."""
    root = tmp_path / "TemporalMemory"
    MemoryStore(root).close()
    upper = Fact(id="a" * 32, entity="User", attribute="drink", value="tea",
                 valid_from="2025-01-01", recorded_at="2025-01-01")
    lower = Fact(id="b" * 32, entity="user", attribute="Drink", value="coffee",
                 valid_from="2025-06-01", recorded_at="2025-06-01")
    other = Fact(id="c" * 32, entity="alex", attribute="drink", value="matcha",
                 valid_from="2025-01-01", recorded_at="2025-01-01")
    city = Fact(id="d" * 32, entity="user", attribute="city", value="Paris",
                valid_from="2025-01-01", recorded_at="2025-01-01")
    written(root, lower, city, other, upper)
    code, listed, _ = run(capsys, "--root", str(root), "list", "--all")
    # Each group sits where its smallest spelling sorts (A to Z, then a to z), as list
    # puts it,
    # so the user's drink and city are apart; --entity lists them together.
    assert (code, listed) == (0, [upper.to_dict(), lower.to_dict(), other.to_dict(),
                                  city.to_dict()])
    code, listed, _ = run(capsys, "--root", str(root), "list")
    assert names(listed) == [("user", "Drink", "coffee"), ("alex", "drink", "matcha"),
                             ("user", "city", "Paris")]
    code, listed, _ = run(capsys, "--root", str(root), "list", "--all", "--entity", "uSeR")
    assert (code, listed) == (0, [upper.to_dict(), lower.to_dict(), city.to_dict()])


def test_list_all_places_a_group_by_its_smallest_pair_of_names(tmp_path, capsys):
    """A group sits where its smallest spelling of both names together sorts, as in list,
    not where its smallest entity spelling and smallest attribute spelling taken apart
    would put it: User/drink sorts after User/city, though Drink sorts before city."""
    root = tmp_path / "TemporalMemory"
    MemoryStore(root).close()
    upper = Fact(id="a" * 32, entity="User", attribute="drink", value="tea",
                 valid_from="2025-01-01", recorded_at="2025-01-01")
    lower = Fact(id="b" * 32, entity="user", attribute="Drink", value="coffee",
                 valid_from="2025-06-01", recorded_at="2025-06-01")
    city = Fact(id="c" * 32, entity="User", attribute="city", value="Paris",
                valid_from="2025-01-01", recorded_at="2025-01-01")
    written(root, upper, lower, city)
    code, listed, _ = run(capsys, "--root", str(root), "list", "--all")
    assert (code, listed) == (0, [city.to_dict(), upper.to_dict(), lower.to_dict()])
    code, listed, _ = run(capsys, "--root", str(root), "list")
    assert names(listed) == [("User", "city", "Paris"), ("user", "Drink", "coffee")]


def test_list_all_puts_spellings_that_share_a_start_in_the_order_they_were_recorded(
        tmp_path):
    root = tmp_path / "TemporalMemory"
    MemoryStore(root).close()
    same = dict(attribute="drink", valid_from="2025-01-01")
    late = Fact(id="1" * 32, entity="User", value="cocoa", recorded_at="2025-03-01", **same)
    early = Fact(id="9" * 32, entity="user", value="tea", recorded_at="2025-02-01", **same)
    # Its ID sorts after that of early, its spelling before: only the ID can order them.
    tied = Fact(id="a" * 32, entity="USER", value="milk", recorded_at="2025-02-01", **same)
    fixed = Fact(id="0" * 32, entity="user", value="green tea", recorded_at="2025-02-01",
                 supersedes=early.id, **same)
    written(root, late, early, tied, fixed)
    with MemoryStore(root, create=False) as store:
        # By recording time; of two recorded at the same instant the smaller ID comes
        # first; each correction right after what it corrects.
        assert store.records() == [early, fixed, tied, late]
        # The history shows only the one begun last; records lists every one.
        assert [fact.value for fact in store.history("user")] == ["cocoa"]


def test_list_all_joins_spellings_that_differ_in_width_spacing_or_case_folding(tmp_path):
    """Names compare by name_key, in list --all as in list: full-width letters, spaces
    at the ends or repeated inside, and case folding (the German sharp s matches SS, the
    Turkish dotted capital I matches I, and a Greek letter that folding splits into a
    letter and accents is put back together) do not count, in either name."""
    root = tmp_path / "TemporalMemory"
    MemoryStore(root).close()
    plain = Fact(id="a" * 32, entity="user", attribute="drink", value="tea",
                 valid_from="2025-01-01", recorded_at="2025-01-01")
    wide = Fact(id="b" * 32, entity="\uff55\uff53\uff45\uff52",
                attribute="\uff44\uff52\uff49\uff4e\uff4b", value="coffee",
                valid_from="2025-02-01", recorded_at="2025-02-01")
    spaced = Fact(id="c" * 32, entity=" user", attribute="drink  ", value="cocoa",
                  valid_from="2025-03-01", recorded_at="2025-03-01")
    folded = Fact(id="d" * 32, entity="Strau\u00df", attribute="Stra\u00dfe",
                  value="Main Street", valid_from="2025-01-01", recorded_at="2025-01-01")
    shouted = Fact(id="e" * 32, entity="STRAUSS", attribute="STRASSE", value="High Street",
                   valid_from="2025-02-01", recorded_at="2025-02-01")
    other = Fact(id="f" * 32, entity="alex", attribute="drink", value="milk",
                 valid_from="2025-01-01", recorded_at="2025-01-01")
    dotted = Fact(id="1" * 32, entity="\u0130stanbul", attribute="home  city",
                  value="Old Town", valid_from="2025-01-01", recorded_at="2025-01-01")
    capital = Fact(id="2" * 32, entity="ISTANBUL", attribute="Home City", value="Harbour",
                   valid_from="2025-02-01", recorded_at="2025-02-01")
    # Case folding turns U+0390 into iota and two accents; folded U+03AA U+0301 is a
    # different sequence until normalization composes both into U+0390 again.
    iota = Fact(id="3" * 32, entity="\u0390", attribute="drink", value="ouzo",
                valid_from="2025-02-01", recorded_at="2025-02-01")
    parts = Fact(id="4" * 32, entity="\u03aa\u0301", attribute="drink", value="water",
                 valid_from="2025-01-01", recorded_at="2025-01-01")
    written(root, plain, wide, spaced, folded, shouted, other, dotted, capital, iota, parts)
    with MemoryStore(root, create=False) as store:
        # Each pair sits where its smallest spelling sorts: " user" (a space sorts
        # first), "ISTANBUL", "STRAUSS", "alex", then the Greek letters; "user",
        # "\u0130stanbul" and the full-width form join pairs placed before them. In
        # each pair the smallest spelling starts last, so a split pair would come out
        # in another order.
        assert store.records() == [plain, wide, spaced, dotted, capital, folded, shouted,
                                   other, parts, iota]
        assert [fact.value for fact in store.snapshot()] == [
            "cocoa", "Harbour", "High Street", "milk", "ouzo"]
        assert store.records(entity="\uff35\uff33\uff25\uff32") == [plain, wide, spaced]
        assert store.records(entity="strauss") == [folded, shouted]
        assert store.records(entity="istanbul") == [dotted, capital]
        assert store.records(entity="\u0390") == [parts, iota]


def test_list_sorts_names_in_code_point_order(tmp_path, capsys):
    """A to Z come first, then a to z, then accented and other letters, as README.md
    says; list --all places its groups the same way."""
    root = tmp_path / "TemporalMemory"
    order = ["Zed", "alex", "zoe", "\u00c9mile", "\u0107ma", "\u0141ukasz", "\u03a9mega"]
    with MemoryStore(root) as store:
        for name in reversed(order):
            store.add(name, "drink", "tea", valid_from="2025-01-01")
    for command in (["list"], ["list", "--all"]):
        code, listed, _ = run(capsys, "--root", str(root), *command)
        assert (code, [fact["entity"] for fact in listed]) == (0, order)


# Errors and set-up -------------------------------------------------------------------


def test_list_on_a_new_memory_is_empty(tmp_path, capsys):
    root = tmp_path / "TemporalMemory"
    assert run(capsys, "--root", str(root), "init")[0] == 0
    assert run(capsys, "--root", str(root), "list") == (0, [], "")
    assert run(capsys, "--root", str(root), "list", "--all") == (0, [], "")


@pytest.mark.parametrize("command", [["list"], ["list", "--all"]], ids=["now", "all"])
def test_list_reports_a_missing_memory_and_creates_nothing(tmp_path, capsys, command):
    root = tmp_path / "TemporalMemory"
    code, output, error = run(capsys, "--root", str(root), *command)
    assert (code, output) == (1, None)
    assert json.loads(error) == {"error": MISSING}
    assert not root.exists()


@pytest.mark.parametrize("option", ["--as-of", "--known-at"])
def test_list_rejects_a_bad_time_without_a_traceback(memory, capsys, option):
    root, _ = memory
    code, output, error = run(capsys, "--root", str(root), "list", option, "yesterday")
    assert (code, output) == (1, None)
    assert "error" in json.loads(error)
    code, output, error = run(capsys, "--root", str(root), "list", "--all", "--known-at", "soon")
    assert (code, output) == (1, None)
    assert "error" in json.loads(error)


def test_list_all_does_not_take_a_date_the_facts_were_true(memory, capsys):
    root, _ = memory
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--root", str(root), "list", "--all", "--as-of", "2025-01-15"])
    assert stopped.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_list_runs_as_a_command(tmp_path):
    root = tmp_path / "TemporalMemory"
    assert invoke(root, "demo").returncode == 0
    now = invoke(root, "list")
    assert now.returncode == 0, now.stderr
    assert [fact["value"] for fact in json.loads(now.stdout)] == [
        "robotics research", "interview completed", "Riverton", "decaf coffee"]
    everything = invoke(root, "list", "--all")
    assert everything.returncode == 0, everything.stderr
    assert len(json.loads(everything.stdout)) == 7
    mixed = invoke(root, "list", "--all", "--as-of", "2026-09-10")
    assert mixed.returncode == 2
    assert "not allowed with argument" in mixed.stderr
    assert not mixed.stdout


# Output and its reader ---------------------------------------------------------------

# Starts a command as a shell starts one in the foreground. A background job, as a test
# runner may be, ignores Ctrl-C, and so would every command it starts.
FOREGROUND = ("import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); "
              "os.execv(sys.executable, [sys.executable] + sys.argv[1:])")
# The pipe holds 64 KiB (set on Linux, where it holds 1 MiB if memory pages are 64 KiB),
# and Python 3.14 keeps up to 128 KiB of output before it writes any: the "held" listing
# reaches the pipe only when it is flushed.
SIZES = pytest.mark.parametrize("notes, smallest, largest", [
    (13, 64 * 1024, 128 * 1024), (40, 128 * 1024, 1024 * 1024)], ids=["held", "written"])


def start_listing(tmp_path, notes):
    """Start ``list --all`` on a store of ``notes`` long facts, writing into a pipe that
    holds 64 KiB; return the process, the pipe's reading end, the file that receives
    standard error, and the size of the listing."""
    root = tmp_path / "TemporalMemory"
    with MemoryStore(root) as store:
        for number in range(notes):
            store.add("user", "note {}".format(number), "x" * 8000, valid_from="2025-01-01")
        size = len(json.dumps([fact.to_dict() for fact in store.records()],
                              ensure_ascii=False, indent=2).encode("utf-8"))
    reading, writing = os.pipe()
    if sys.platform.startswith("linux"):
        import fcntl
        fcntl.fcntl(writing, fcntl.F_SETPIPE_SZ, 64 * 1024)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    errors = tmp_path / "errors.txt"
    with errors.open("wb") as error_file:
        listing = subprocess.Popen(
            [sys.executable, "-c", FOREGROUND, "-m", "temporal_memory_os", "--root", str(root),
             "list", "--all"], stdout=writing, stderr=error_file, env=environment)
    os.close(writing)
    return listing, open(reading, "rb", buffering=0), errors, size


def finish(listing, reader):
    reader.close()
    if listing.poll() is None:
        listing.kill()
        listing.wait()


@SIZES
def test_list_into_a_reader_that_stops_early_reports_no_error(tmp_path, notes, smallest,
                                                               largest):
    """Piped into head, or into a pager closed before the end, list stops quietly with
    the status a shell gives a command stopped that way, as git log or seq do, instead of
    reporting the closed pipe as an error of its own."""
    listing, reader, errors, size = start_listing(tmp_path, notes)
    try:
        assert smallest < size < largest
        assert reader.read(1) == b"["  # Then the reader stops.
        reader.close()
        assert listing.wait(timeout=60) == 141
    finally:
        finish(listing, reader)
    assert errors.read_bytes() == b""


@SIZES
def test_list_interrupted_while_its_reader_waits_ends_at_once(tmp_path, notes, smallest,
                                                              largest):
    """Ctrl-C while the output fills a pipe that its reader has stopped taking from, as a
    pager does after the first page, ends list at once with status 130 and nothing on
    standard error. Before, on Python 3.14 with part of the output still held back, Python
    printed its own warning about the interrupt at once, then waited for the pager to
    close, and the command ended with status 120."""
    listing, reader, errors, size = start_listing(tmp_path, notes)
    try:
        assert smallest < size < largest
        assert reader.read(1) == b"["  # The reader takes no more, and the pipe stays full.
        listing.send_signal(signal.SIGINT)
        assert listing.wait(timeout=30) == 130
    finally:
        finish(listing, reader)
    assert errors.read_bytes() == b""


def test_a_command_started_with_its_output_closed_does_its_work_quietly(tmp_path):
    """Started with standard output closed (>&-), when Python has no sys.stdout, add
    still saves its record, and add and list end with status 0 and nothing on standard
    error, as in 0.3.1."""
    root = tmp_path / "TemporalMemory"
    with MemoryStore(root):
        pass
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    for arguments in (["add", "user", "drink", "tea", "--from", "2025-01-01"],
                      ["list", "--all"]):
        closed = subprocess.run(
            ["sh", "-c", 'exec "$@" >&-', "sh", sys.executable, "-m", "temporal_memory_os",
             "--root", str(root), *arguments], capture_output=True, env=environment,
            timeout=60)
        assert (closed.returncode, closed.stderr) == (0, b"")
    with MemoryStore(root) as store:
        assert [fact.value for fact in store.records()] == ["tea"]


@pytest.mark.parametrize("stop, status", [(BrokenPipeError, 141), (KeyboardInterrupt, 130)],
                         ids=["closed pipe", "ctrl-c"])
def test_list_ends_quietly_when_its_output_stops_and_has_no_file(memory, capsys,
                                                                monkeypatch, stop, status):
    """Run by a program that gives the CLI a standard output with no file behind it, list
    still ends with the status for a stopped reader or for Ctrl-C, and reports nothing."""
    root, _ = memory

    class Stopping(io.StringIO):
        def write(self, text):
            raise stop()

    monkeypatch.setattr(sys, "stdout", Stopping())
    assert cli.main(["--root", str(root), "list"]) == status
    assert capsys.readouterr().err == ""


# Every history -----------------------------------------------------------------------

DAYS = ["2025-01-0{}".format(day) for day in range(1, 5)]
STEPS = st.lists(st.tuples(
    st.sampled_from(["add", "correct", "retract"]),
    st.sampled_from(["user", "User", "alex"]),
    st.sampled_from(["drink", "city"]),
    st.sampled_from(DAYS),  # Start of a new fact.
    st.integers(min_value=0, max_value=11),  # The record a revision revises.
    st.sampled_from(DAYS),  # When it is recorded; few days make ties likely.
), min_size=1, max_size=12)


@given(STEPS)
def test_records_lists_every_record_once_grouped_and_each_revision_after_its_target(steps):
    with tempfile.TemporaryDirectory() as folder:
        with MemoryStore(Path(folder) / "memory-data") as store:
            made = []
            for kind, entity, attribute, start, pick, recorded in steps:
                try:
                    if kind == "add" or not made:
                        made.append(store.add(entity, attribute, "v", valid_from=start,
                                              recorded_at=recorded))
                    elif kind == "correct":
                        made.append(store.correct(made[pick % len(made)].id, "w",
                                                  recorded_at=recorded))
                    else:
                        made.append(store.retract(made[pick % len(made)].id,
                                                  recorded_at=recorded))
                except ValueError:  # Refused writes, MemoryConflictError included.
                    continue
            listed = store.records()
            assert sorted(fact.id for fact in listed) == sorted(fact.id for fact in made)
            keys = [(fact.entity, fact.attribute, fact.valid_from) for fact in listed]
            assert keys == sorted(keys)
            place = {fact.id: index for index, fact in enumerate(listed)}
            for fact in listed:
                if fact.supersedes is not None:
                    assert place[fact.supersedes] == place[fact.id] - 1
            for cutoff in ["2024-12-31", *DAYS, "2025-01-02T12:00:00Z"]:
                known = normalize_time(cutoff)
                assert store.records(known_at=cutoff) == [
                    fact for fact in listed if fact.recorded_at <= known]
            for entity in ("USER", "alex", "nobody"):
                assert store.records(entity=entity) == [
                    fact for fact in listed if name_key(fact.entity) == name_key(entity)]


# Groups of spellings that name_key joins, one rule or more apart: case, width, spaces
# at the ends or repeated inside, case folding (sharp s and SS, the dotted capital I) and
# composition after it (Greek iota with accents), in the entity, the attribute or both.
# A group key weaker than name_key splits one of them.
JOINED = [
    [("user", "drink"), ("User", "drink"), ("USER", "Drink"), ("\uff55\uff53\uff45\uff52", "drink"),
     (" user", "drink  "), ("USER", "\uff44\uff52\uff49\uff4e\uff4b")],
    [("Strau\u00df", "Stra\u00dfe"), ("STRAUSS", "STRASSE")],
    [("\u0130stanbul", "drink"), ("ISTANBUL", "drink")],
    [("user", "home  city"), ("User", "Home City")],
    [("\u0390", "drink"), ("\u03aa\u0301", "drink")],
]
OTHERS = [("alex", "drink"), ("user", "city"), ("USER", "job")]
# Each example draws its chains from every spelling, or from one group and three other
# names, where a split group or a misordered shared start shows more often. USER/job
# sorts after USER/drink and before USER with a full-width drink, so a group placed by
# its smallest entity and attribute spellings taken apart, not by its smallest pair of
# names, comes out in another order.
POOLS = [[names for group in JOINED for names in group] + OTHERS] + [
    group + OTHERS for group in JOINED]


def chains_from(spellings):
    return st.lists(st.tuples(
        st.sampled_from(spellings),  # Names, as this chain spells them.
        st.integers(min_value=1, max_value=4),  # Start day.
        st.integers(min_value=1, max_value=4),  # Day its first record was recorded.
        st.lists(st.integers(min_value=0, max_value=1), max_size=3),  # Days to each revision.
        st.booleans(),  # Whether the last revision is a retraction.
        st.booleans(),  # Whether it was recorded in 2099, after now, as an import can be.
    ), max_size=8, unique_by=lambda chain: (chain[0], chain[1]))


def stated_order(facts):
    """The order records() states: groups of names compared by name_key, each where its
    smallest spelling sorts; in a group, first records by start, recording time and ID,
    each followed by its revisions."""
    following = {fact.supersedes: fact for fact in facts if fact.supersedes is not None}
    pairs = []
    for fact in sorted(facts, key=lambda fact: (fact.entity, fact.attribute)):
        pair = (name_key(fact.entity), name_key(fact.attribute))
        if pair not in pairs:
            pairs.append(pair)
    ordered = []
    for pair in pairs:
        firsts = sorted((fact for fact in facts if fact.supersedes is None
                         and (name_key(fact.entity), name_key(fact.attribute)) == pair),
                        key=lambda fact: (fact.valid_from, fact.recorded_at, fact.id))
        for fact in firsts:
            ordered.append(fact)
            while fact.id in following:
                fact = following[fact.id]
                ordered.append(fact)
    return ordered


@given(st.data())
def test_records_keeps_each_history_together_in_a_hand_written_store(data):
    """Several spellings of a name, some sharing a start, revisions recorded at their
    target's instant, records recorded after now and IDs in any order, as a store
    written by hand or imported can hold."""
    chains = data.draw(chains_from(data.draw(st.sampled_from(POOLS))))
    count = sum(len(chain[3]) + 1 for chain in chains)
    ids = ["{:032x}".format(number) for number in data.draw(st.permutations(range(count)))]
    facts = []
    for (entity, attribute), start, recorded, steps, retracted, later in chains:
        previous = None
        for index in range(len(steps) + 1):
            deleted = retracted and index == len(steps) and index > 0
            fact = Fact(id=ids[len(facts)], entity=entity, attribute=attribute,
                        value="" if deleted else "v{}".format(index), deleted=deleted,
                        valid_from="2025-01-0{}".format(start),
                        recorded_at="{}-01-0{}".format(2099 if later else 2025,
                                                       recorded + sum(steps[:index])),
                        supersedes=previous)
            facts.append(fact)
            previous = fact.id
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "memory-data"
        MemoryStore(root).close()
        written(root, *facts)
        with MemoryStore(root, create=False) as store:
            stored = store.all_facts()
            assert len(stored) == len(facts)
            for cutoff in [None, "2025-01-01", "2025-01-03", "2025-01-05", "2099-01-03"]:
                known = None if cutoff is None else normalize_time(cutoff)
                kept = [fact for fact in stored if known is None or fact.recorded_at <= known]
                listed = store.records(known_at=cutoff)
                assert listed == stated_order(kept)
                # Groups come in the order snapshot lists its facts with the same cutoff;
                # without one, records() counts the records of 2099 too, so snapshot is
                # given a cutoff after them (its own default, now, leaves them out).
                pairs = list(dict.fromkeys((name_key(fact.entity), name_key(fact.attribute))
                                           for fact in listed))
                shown = [(name_key(fact.entity), name_key(fact.attribute))
                         for fact in store.snapshot(known_at=cutoff or "2100-01-01")]
                assert shown == [pair for pair in pairs if pair in shown]
                for entity in ("USER", "alex", "strauss", "\u0130STANBUL", "\u03aa\u0301"):
                    assert store.records(known_at=cutoff, entity=entity) == [
                        fact for fact in listed if name_key(fact.entity) == name_key(entity)]
