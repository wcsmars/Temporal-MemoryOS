"""Contracts that callers and files on disk rely on.

Surviving mutants showed that these could change without any other test
failing: the JSON shape of search results, the key order of Markdown records
and proposal files, default arguments, folder and file names, and the guards
of the proposal queue. One test checks that conftest.py keeps the Git tests
working when the temporary folder is inside a Git checkout. The last tests
check the rule that tools/mutation_summary.py uses to set aside survivors
that only change the text of a message or the letter case of SQL keywords,
and that it gives no score for an unfinished run.
"""

import difflib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from temporal_memory_os import cli, extraction
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.models import Fact
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.store import MemoryConflictError, MemoryStore

FACT_KEYS = ["id", "entity", "attribute", "value", "valid_from", "valid_to", "recorded_at",
             "source", "evidence", "confidence", "tags", "links", "topic", "supersedes",
             "deleted"]
RESULT_KEYS = ["fact", "score", "channels", "citation"]
PROPOSAL_KEYS = ["id", "entity", "attribute", "value", "valid_from", "source", "evidence",
                 "topic", "created_at", "status", "kind", "confidence", "target_id",
                 "target_fingerprint", "prior_value", "result_fact_id", "warnings", "dated",
                 "observed"]
CHANNELS = {"keyword", "lexical_embedding", "semantic", "graph"}


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memory-root") as memory:
        yield memory


def add_application(store):
    return store.add("Acme application", "status", "interview invited",
                     valid_from="2026-09-15", source="synthetic:test",
                     topic="career/applications")


def check_result(result, fact):
    """One search result in JSON form: these keys in this order, with these types."""
    assert list(result) == RESULT_KEYS
    assert result["fact"] == fact.to_dict()
    assert type(result["score"]) is float and result["score"] > 0
    assert result["channels"] and set(result["channels"]) <= CHANNELS
    assert all(type(score) is float for score in result["channels"].values())
    assert result["citation"] == "memory/career/applications/{}.md".format(fact.id)


# Search output

def test_search_result_to_dict_has_the_documented_shape(store):
    fact = add_application(store)
    results = HybridRetriever(store).search("Acme application status")
    assert [result.fact.id for result in results] == [fact.id]
    data = results[0].to_dict()
    check_result(data, fact)
    assert data["score"] == results[0].score and data["channels"] == results[0].channels
    data["channels"].clear()  # A copy: changing the output never changes the result.
    assert results[0].channels


def test_cli_search_prints_the_result_shape(store, capsys):
    fact = add_application(store)
    assert cli.main(["--root", str(store.root), "search", "Acme application status"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert len(printed) == 1
    check_result(printed[0], fact)


def test_api_search_returns_the_result_shape(store):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from temporal_memory_os.api import create_app

    fact = add_application(store)
    with TestClient(create_app(store.root), base_url="http://localhost") as client:
        response = client.get("/api/search", params={"q": "Acme application status"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body) == ["results"] and len(body["results"]) == 1
    check_result(body["results"][0], fact)


# Records

def test_markdown_records_keep_the_fact_key_order(store):
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01", tags=["b", "a"],
                     links=["Acme"])
    assert list(fact.to_dict()) == FACT_KEYS
    text = store.path_for(fact.id).read_text(encoding="utf-8")
    written = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert list(written) == FACT_KEYS
    assert written == fact.to_dict()
    assert (written["tags"], written["links"]) == (["b", "a"], ["Acme"])


@pytest.mark.parametrize("valid_to", ["2025-02-01", "2025-02-01T08:00+08:00", " 2025-02-01 ",
                                      date(2025, 2, 1)])
def test_fact_normalizes_valid_to(valid_to):
    fact = Fact(id="a" * 32, entity="person", attribute="city", value="Riverton",
                valid_from="2025-01-01", valid_to=valid_to, recorded_at="2025-01-01")
    assert fact.valid_to == "2025-02-01T00:00:00.000000Z"
    assert fact.to_dict()["valid_to"] == "2025-02-01T00:00:00.000000Z"


def test_fact_deduplicates_tags_and_links_in_order():
    fact = Fact(id="a" * 32, entity="person", attribute="city", value="Riverton",
                valid_from="2025-01-01", recorded_at="2025-01-01",
                tags=["b", "a", "b"], links=["Acme", "Riverton", "Acme"])
    assert (fact.tags, fact.links) == (("b", "a"), ("Acme", "Riverton"))
    assert (fact.to_dict()["tags"], fact.to_dict()["links"]) == (["b", "a"], ["Acme", "Riverton"])


def test_write_defaults_are_recorded_in_markdown(tmp_path):
    root = tmp_path / "memory-root"
    with MemoryStore(root) as store:
        plain = store.add("person", "drink", "tea", valid_from="2025-01-01")
        assert (plain.source, plain.evidence, plain.topic) == ("manual", "", "general")
        assert (plain.confidence, plain.tags, plain.links) == (1.0, (), ())
        imported = store.add("Acme", "status", "pending", valid_from="2025-01-01",
                             source="synthetic:import", evidence="import line")
        corrected = store.correct(imported.id, "submitted")
        assert (corrected.source, corrected.evidence) == ("manual", "")
        retracted = store.retract(corrected.id)
        assert (retracted.source, retracted.evidence) == ("manual", "Retracted assertion")
        assert (retracted.value, retracted.deleted) == ("", True)
    with MemoryStore(root) as reopened:  # Read back from the Markdown files.
        for fact in (plain, corrected, retracted):
            assert reopened.get(fact.id) == fact
        assert reopened.path_for(plain.id).relative_to(reopened.root).as_posix() == (
            "memory/general/{}.md".format(plain.id))


# Folder and file names

def test_store_and_queue_folder_and_file_names(tmp_path):
    root = tmp_path / "memory-root"
    with MemoryStore(root) as store:
        # os.listdir reports the names' real letter case, also on case-insensitive disks.
        assert sorted(os.listdir(root)) == [".memoryos", "memory"]
        assert "index.sqlite3" in os.listdir(root / ".memoryos")
        assert store.db_path.relative_to(store.root).as_posix() == ".memoryos/index.sqlite3"
        queue = ProposalQueue(store)
        assert sorted(os.listdir(root)) == [".memoryos", "memory", "proposals"]
        assert queue.directory == store.root / "proposals"
        proposal, = queue.extract("Alex | city | Riverton", "synthetic:chat",
                                  observed_at="2025-01-01")
        assert sorted(os.listdir(root / "proposals")) == [".lock", proposal.id + ".json"]


def test_extract_defaults_to_the_general_topic(store):
    queue = ProposalQueue(store)
    proposal, = queue.extract("Alex | city | Riverton", "synthetic:chat",
                              observed_at="2025-01-01")
    assert proposal.topic == "general"
    fact = queue.approve(proposal.id)
    assert fact.topic == "general"
    assert os.listdir(store.root / "memory") == ["general"]
    assert os.listdir(store.root / "memory" / "general") == [fact.id + ".md"]


def test_proposal_files_hold_the_documented_keys(store):
    queue = ProposalQueue(store)
    proposal, = queue.extract("Alex | city | Zürich", "synthetic:chat",
                              observed_at="2025-01-01")
    text = (store.root / "proposals" / (proposal.id + ".json")).read_text(encoding="utf-8")
    data = json.loads(text)
    assert list(data) == PROPOSAL_KEYS
    assert data == proposal.to_dict()
    assert '"value": "Zürich"' in text and text.endswith("}\n")  # Readable, one key per line.
    assert queue.get(proposal.id) == proposal


# Proposal queue guards

def test_proposal_directory_must_not_be_a_symbolic_link(store, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (store.root / "proposals").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Proposal directory must not be a symbolic link"):
        ProposalQueue(store)
    assert not list(outside.iterdir())


def test_proposal_lock_must_not_be_a_symbolic_link(store, tmp_path_factory):
    queue = ProposalQueue(store)
    target = tmp_path_factory.mktemp("outside") / "lock"
    (queue.directory / ".lock").symlink_to(target)
    with pytest.raises(ValueError, match="Proposal lock must not be a symbolic link"):
        queue.extract("Alex | city | Riverton", "synthetic:chat")
    assert not target.exists()
    assert not list(queue.directory.glob("*.json"))


def test_proposal_files_must_not_be_symbolic_links(store, tmp_path_factory):
    queue = ProposalQueue(store)
    proposal, = queue.extract("Alex | city | Riverton", "synthetic:chat",
                              observed_at="2025-01-01")
    path = queue.directory / (proposal.id + ".json")
    target = tmp_path_factory.mktemp("outside") / "proposal.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    for action in (queue.get, queue.approve, queue.reject):
        with pytest.raises(ValueError, match="Proposal files must not be symbolic links"):
            action(proposal.id)
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "pending"
    assert store.all_facts() == []
    assert queue.list(status="all") == []
    assert queue.problems == [
        "Proposal files must not be symbolic links: proposals/{}.json".format(proposal.id)]


def test_list_filters_by_status_and_rejects_unknown_statuses(store):
    queue = ProposalQueue(store)
    city, drink, food = queue.extract("Alex | city | Riverton\nAlex | drink | tea\n"
                                      "Alex | food | rice", "synthetic:chat",
                                      observed_at="2025-01-01")
    queue.approve(city.id)
    queue.reject(drink.id)

    def listed(*status):
        return [proposal.id for proposal in queue.list(*status)]

    assert listed("approved") == [city.id]
    assert listed("rejected") == [drink.id]
    assert listed("pending") == listed() == [food.id]
    assert listed("all") == listed(None) == [city.id, drink.id, food.id]
    for status in ("APPROVED", "done", ""):
        with pytest.raises(ValueError, match="status must be pending, approved, rejected, or all"):
            queue.list(status=status)
    with pytest.raises(MemoryConflictError, match="Approved proposals cannot be rejected"):
        queue.reject(city.id)
    assert queue.get(city.id).status == "approved"


class _FailingOs:
    """The os module as extraction sees it, with one function failing."""

    def __init__(self, failing):
        self.failing = failing

    def __getattr__(self, name):
        if name == self.failing:
            def fail(*_args, **_kwargs):
                raise OSError("simulated {} failure".format(name))
            return fail
        return getattr(os, name)


@pytest.mark.parametrize("failing", ["fsync", "replace"])
def test_failed_proposal_save_leaves_no_temporary_file(store, monkeypatch, failing):
    queue = ProposalQueue(store)
    pending, = queue.extract("Alex | city | Riverton", "synthetic:chat",
                             observed_at="2025-01-01")
    with monkeypatch.context() as patch:
        patch.setattr(extraction, "os", _FailingOs(failing))
        with pytest.raises(OSError, match="simulated"):
            queue.extract("Alex | drink | tea", "synthetic:chat", observed_at="2025-01-01")
        with pytest.raises(OSError, match="simulated"):
            queue.approve(pending.id)  # The fact is written; saving the proposal fails.
    assert sorted(os.listdir(queue.directory)) == [".lock", pending.id + ".json"]
    assert queue.get(pending.id).status == "pending"


# tests/conftest.py

@pytest.mark.skipif(not shutil.which("git"), reason="Git unavailable")
def test_git_tests_pass_with_the_temp_folder_inside_a_git_checkout(tmp_path):
    """Git in a test never finds a repository around pytest's base temp folder,
    so the Git tests pass when TMPDIR is inside a Git checkout."""
    repository = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True, capture_output=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    environment["HYPOTHESIS_STORAGE_DIRECTORY"] = str(tmp_path / "hypothesis")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
         "--basetemp", str(checkout / "tmp"),
         "tests/test_git.py::test_git_init_refuses_parent_repository"],
        cwd=repository, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


# tools/mutation_summary.py

def load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "mutation_summary.py"
    spec = importlib.util.spec_from_file_location("mutation_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mutant_diff(function, old, new):
    """The diff that ``mutmut show`` prints for one changed literal of ``function``."""
    original = function.strip()
    assert original.count(old) == 1, old
    lines = difflib.unified_diff(original.split("\n"), original.replace(old, new).split("\n"),
                                 fromfile="src/module.py", tofile="src/module.py", lineterm="")
    return "# module.x_function__mutmut_1: survived\n" + "\n".join(lines)


LIST = '''
def list(self, status="pending"):
    """Proposals in creation order."""
    if status not in (None, "all", "pending", "approved", "rejected"):
        raise ValueError("status must be pending, approved, rejected, or all")
    return [p for p in self.everything() if status in (None, "all") or p.status == status]
'''
PARSE = '''
def parse(text):
    match = PATTERN.fullmatch(text)
    if not match:
        raise ValueError(
            "Time must be YYYY-MM-DD or YYYY-MM-DDTHH:MM with an optional "
            "Z or +HH:MM offset; got {!r}".format(text))
    return match
'''
DOCUMENTED = '''
def create(model_name=None):
    """Create the configured embedder.

    An explicit model name overrides the environment.
    """
    if model_name is not None and not isinstance(model_name, str):
        raise ValueError("model_name must be a string or None")
    return model_name
'''
TO_DICT = '''
def to_dict(self):
    return {
        "fact": self.fact.to_dict(),
        "score": self.score,
        "channels": dict(self.channels),
        "citation": self.citation,
    }
'''
QUEUE = '''
def __init__(self, store):
    self.store = store
    self.directory = Path(store.root) / "proposals"
    if self.directory.is_symlink():
        raise ValueError("Proposal directory must not be a symbolic link")
    self.directory.mkdir(parents=True, exist_ok=True)
'''
ADD = '''
def add(self, entity, attribute, value, valid_from=None,
        valid_to=None, source="manual", evidence="",
        topic="general"):
    with self.transaction():
        return self.write(entity, attribute, value, source, evidence, topic)
'''
SQL = '''
def group(self, entity, attribute):
    rows = self.connection.execute(
        "SELECT id, payload FROM facts WHERE entity = ? AND attribute = ?", (entity, attribute))
    count = self.connection.execute("SELECT count(*) FROM facts WHERE kind = ?", ("Add",))
    return [self.fact(row) for row in rows], count
'''
SCRIPT = '''
def initialize(self, connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY, payload TEXT NOT NULL
        );
    """)
'''
KEYWORD = '''
def keyword(self, terms):
    expression = " OR ".join('"' + term + '"' for term in terms)
    return self.connection.execute("SELECT id FROM selected WHERE selected MATCH ?",
                                   (expression,))
'''
RENDER = '''
def render(fact):
    title = "{} · {}".format(fact.entity, fact.attribute)
    if not title:
        raise ValueError(MESSAGES["missing"])
    if fact.kind not in KINDS:
        raise ConfigError("Unknown kind", code="E_KIND")
    return "# " + title + "\\n"
'''
SERVE = '''
def serve(host):
    if host not in LOCAL:
        print("Warning: serving on {} without a token.".format(host), file=sys.stderr)
        warnings.warn("Serving without a token", RuntimeWarning)
    print("Serving on {}".format(host))
    connect(host, timeout=30)
'''

TEXT_ONLY_CASES = [
    # Exception and warning messages, including continued lines and docstring context.
    pytest.param(LIST, '"status must be pending, approved, rejected, or all"',
                 '"XXstatus must be pending, approved, rejected, or allXX"', True,
                 id="raise-message"),
    pytest.param(PARSE, '"Z or +HH:MM offset; got {!r}"', '"z or +hh:mm offset; got {!r}"',
                 True, id="raise-message-continued"),
    pytest.param(PARSE, '"Time must be', '"XXTime must be', True, id="raise-message-next-line"),
    pytest.param(DOCUMENTED, '"model_name must be a string or None"',
                 '"MODEL_NAME MUST BE A STRING OR NONE"', True, id="raise-after-docstring"),
    pytest.param(QUEUE, '"Proposal directory must not be a symbolic link"',
                 '"proposal directory must not be a symbolic link"', True, id="raise-case"),
    pytest.param(SERVE, '"Warning: serving on {} without a token."',
                 '"XXWarning: serving on {} without a token.XX"', True, id="print-to-stderr"),
    pytest.param(SERVE, '"Serving without a token"', '"SERVING WITHOUT A TOKEN"', True,
                 id="warnings-warn"),
    # SQL keyword case, in one statement string or a multi-line script.
    pytest.param(SQL, '"SELECT id, payload FROM facts WHERE entity = ? AND attribute = ?"',
                 '"select id, payload from facts where entity = ? and attribute = ?"', True,
                 id="sql-keyword-case"),
    pytest.param(SCRIPT, "CREATE TABLE IF NOT EXISTS facts (\n            id TEXT PRIMARY KEY, "
                 "payload TEXT NOT NULL", "create table if not exists facts (\n            id "
                 "text primary key, payload text not null", True, id="sql-script-keyword-case"),
    pytest.param(KEYWORD, '"SELECT id FROM selected WHERE selected MATCH ?"',
                 '"select id from selected where selected match ?"', True, id="sql-match-case"),
    # A changed placeholder changes how the message is built.
    pytest.param(PARSE, '"Z or +HH:MM offset; got {!r}"', '"Z OR +HH:MM OFFSET; GOT {!R}"',
                 False, id="message-placeholder"),
    # Compared values, dictionary keys, names on disk and default arguments.
    pytest.param(LIST, '"approved", "rejected")', '"XXapprovedXX", "rejected")', False,
                 id="compared-value"),
    pytest.param(LIST, 'status="pending"', 'status="XXpendingXX"', False, id="default-on-def"),
    pytest.param(TO_DICT, '"score"', '"XXscoreXX"', False, id="dict-key"),
    pytest.param(TO_DICT, '"citation"', '"CITATION"', False, id="dict-key-case"),
    pytest.param(QUEUE, '"proposals"', '"PROPOSALS"', False, id="folder-name"),
    pytest.param(ADD, 'source="manual"', 'source="XXmanualXX"', False, id="default-continued"),
    pytest.param(ADD, 'topic="general"', 'topic="GENERAL"', False, id="default-case"),
    pytest.param(RENDER, '"missing"', '"XXmissingXX"', False, id="key-inside-raise"),
    pytest.param(RENDER, '"E_KIND"', '"XXE_KINDXX"', False, id="raise-second-argument"),
    pytest.param(RENDER, '"# "', '"XX# XX"', False, id="returned-text"),
    pytest.param(RENDER, '"{} · {}"', '"XX{} · {}XX"', False, id="format-string"),
    pytest.param(SERVE, '"Serving on {}"', '"XXServing on {}XX"', False, id="print-to-stdout"),
    # SQL: identifiers, values outside the statement, FTS operators, and non-case changes.
    pytest.param(SQL, '"SELECT id, payload FROM facts WHERE entity = ? AND attribute = ?"',
                 '"SELECT ID, PAYLOAD FROM FACTS WHERE ENTITY = ? AND ATTRIBUTE = ?"', False,
                 id="sql-identifier-case"),
    pytest.param(SQL, '("Add",)', '("ADD",)', False, id="sql-parameter-value"),
    pytest.param(SQL, '"SELECT count(*)', '"XXSELECT count(*)', False, id="sql-text"),
    pytest.param(KEYWORD, '" OR "', '" or "', False, id="fts-operator"),
    # Code, not a literal.
    pytest.param(SERVE, "timeout=30", "timeout=31", False, id="number"),
]


@pytest.mark.parametrize("function, old, new, expected", TEXT_ONLY_CASES)
def test_mutation_summary_text_only_rule(function, old, new, expected):
    assert load_tool().text_only(mutant_diff(function, old, new)) is expected


def test_mutation_summary_report(monkeypatch, capsys):
    tool = load_tool()
    shown = {
        "module.x_list__mutmut_1": mutant_diff(LIST, '"status must', '"XXstatus must'),
        "module.x_list__mutmut_2": mutant_diff(LIST, '"approved", "rejected")',
                                               '"XXapprovedXX", "rejected")'),
    }

    def mutmut(*arguments):
        if arguments[0] == "results":
            return ("    module.x_list__mutmut_1: survived\n"
                    "    module.x_list__mutmut_2: survived\n"
                    "    module.x_list__mutmut_3: killed\n"
                    "    module.x_add__mutmut_1: timeout\n")
        return shown[arguments[1]]

    monkeypatch.setattr(tool, "mutmut", mutmut)
    monkeypatch.setattr(sys, "argv", ["mutation_summary.py"])
    assert tool.main() == 0
    assert capsys.readouterr().out.splitlines() == [
        "mutants: 4  caught: 2 (50.0%)  survived: 2",
        "survivors that only change message text or SQL keyword case: 1",
        "other survivors to review: 1",
        "     1  module.x_list",
    ]


def test_mutation_summary_without_results_reports_no_score(monkeypatch, capsys):
    """With no finished run to read, the summary fails instead of printing a zero score."""
    tool = load_tool()
    monkeypatch.setattr(tool, "mutmut", lambda *arguments: "")
    monkeypatch.setattr(sys, "argv", ["mutation_summary.py"])
    assert tool.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No mutmut results here" in captured.err


@pytest.mark.parametrize("results, hint", [
    # The baseline tests failed, so mutmut generated the mutants and checked none.
    ("    pkg.x_add__mutmut_1: not checked\n", "mutmut run\n"),
    # Interrupted part-way; mutmut checks an interrupted mutant again only by name.
    ("    pkg.x_add__mutmut_1: killed\n"
     "    pkg.x_add__mutmut_2: check was interrupted by user\n"
     "    pkg.x_add__mutmut_3: not checked\n",
     "mutmut run pkg.x_add__mutmut_2\n"),
])
def test_mutation_summary_of_an_unfinished_run_reports_no_score(monkeypatch, capsys, results,
                                                                 hint):
    """Mutants that were never checked fail the summary instead of lowering the score."""
    tool = load_tool()
    monkeypatch.setattr(tool, "mutmut", lambda *arguments: results)
    monkeypatch.setattr(sys, "argv", ["mutation_summary.py"])
    assert tool.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "The mutmut run did not finish" in captured.err
    assert captured.err.endswith(hint)


def test_mutation_summary_shows_why_mutmut_failed(monkeypatch):
    """A failing mutmut command exits with mutmut's own error and how to install it."""
    tool = load_tool()

    def run(command, **options):
        return tool.subprocess.CompletedProcess(command, 1, "", "No module named mutmut\n")

    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(SystemExit) as raised:
        tool.mutmut("results", "--all", "true")
    message = str(raised.value.code)
    assert "mutmut results failed" in message and "No module named mutmut" in message
    assert "python -m pip install -e '.[mutation]'" in message
