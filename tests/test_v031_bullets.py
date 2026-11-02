"""List markers in extracted text (0.3.1).

A line from a bulleted or numbered list reads like the same line without its
marker, and its evidence leaves the marker out, so "- X" and "X" are one
statement. Before, a marker became part of the entity ("- user"), which started
a second history, and bulleted or numbered prose was skipped without a word.
"""

import json

import pytest

from temporal_memory_os import MemoryStore, cli
from temporal_memory_os.extraction import ProposalQueue

BOM = "\ufeff"
BULLETS = ["-", "*", "+", "\u2022", "\u25e6", "\u2023", "\u25aa", "\u25cf", "\u25cb", "\u25a0",
           "\u00b7", "\u30fb", "\N{EN DASH}", "\N{EM DASH}"]
NUMBERS = ["1.", "1)", "0.", "42.", "999)", "007."]
PARIS = ("user", "city", "Paris", "2026-06-01", "user | city | Paris")


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path) as memory:
        yield memory


@pytest.fixture
def queue(store):
    return ProposalQueue(store)


def parsed(queue, line):
    """(entity, attribute, value, date, evidence) of what ``line`` proposes, read beside
    a plain line that always proposes something, so a skipped line is not an error."""
    *proposals, last = queue.extract(line + "\nAlex | pet | cat", source="chat",
                                     observed_at="2026-06-01")
    assert last.evidence == "Alex | pet | cat"
    return [(p.entity, p.attribute, p.value, p.valid_from[:10], p.evidence) for p in proposals]


def test_a_bulleted_pipe_line_and_numbered_dated_prose_are_read(queue):
    """A bulleted pipe line names "user", not "- user"; numbered dated prose is read."""
    assert parsed(queue, "- user | snack | popcorn | 2026-11-01") == [
        ("user", "snack", "popcorn", "2026-11-01", "user | snack | popcorn | 2026-11-01")]
    assert parsed(queue, "1. 2026-11-01: My favourite drink is matcha.") == [
        ("user", "favourite drink", "matcha", "2026-11-01",
         "2026-11-01: My favourite drink is matcha.")]


@pytest.mark.parametrize("marker", BULLETS + NUMBERS)
@pytest.mark.parametrize("line, expected", [
    ("user | city | Paris", PARIS),
    ("My city is Paris.", ("user", "city", "Paris", "2026-06-01", "My city is Paris.")),
    ("2026-01-02: I prefer tea.",
     ("user", "preference", "tea", "2026-01-02", "2026-01-02: I prefer tea.")),
    ("Application Acme is pending.", ("Acme application", "status", "pending", "2026-06-01",
                                      "Application Acme is pending.")),
], ids=["pipe", "prose", "dated-prose", "application"])
def test_a_list_marker_is_not_part_of_the_line(queue, marker, line, expected):
    assert parsed(queue, marker + " " + line) == [expected]


@pytest.mark.parametrize("line", [
    "-\tuser | city | Paris",
    "-    user | city | Paris",
    "1.\u00a0user | city | Paris",
    "\u2022\u3000user | city | Paris",
    "  - user | city | Paris",
    "\t\t* user | city | Paris",
    "      12) user | city | Paris",
    BOM + "- user | city | Paris",
    BOM + "  + user | city | Paris  ",
], ids=["tab", "spaces", "no-break-space", "ideographic-space", "nested", "nested-tabs",
        "nested-number", "byte-order-mark", "byte-order-mark-and-indent"])
def test_any_whitespace_may_follow_or_indent_a_marker(queue, line):
    assert parsed(queue, line) == [PARIS]


def test_a_bulleted_file_with_windows_line_ends_and_a_byte_order_mark_on_each_line(store, queue):
    """Files joined with cat keep each one's byte-order mark, before the marker."""
    store.add("user", "city", "London", valid_from="2026-01-01")
    text = (BOM + "- user | city | Paris | 2026-02-01\r\n" + BOM + "  * My drink is tea.\r\n"
            + "1. # notes\r\n" + BOM + "2) I prefer jazz.\r\n")
    proposals = queue.extract(text, source="notes", observed_at="2026-03-01")
    assert [(p.entity, p.attribute, p.value, p.kind, p.evidence) for p in proposals] == [
        ("user", "city", "Paris", "update", "user | city | Paris | 2026-02-01"),
        ("user", "drink", "tea", "add", "My drink is tea."),
        ("user", "preference", "jazz", "add", "I prefer jazz.")]


@pytest.mark.parametrize("line, entity", [
    ("1.5 | ratio | x", "1.5"),
    ("-5 | t | v", "-5"),
    ("+44 | code | UK", "+44"),
    ("**bold** | x | y", "**bold**"),
    ("*italic* | x | y", "*italic*"),
    ("-user | x | y", "-user"),
    ("1.user | x | y", "1.user"),
    ("1000. user | x | y", "1000. user"),
    ("(1) user | x | y", "(1) user"),
    ("(555) 123-4567 | owner | Alex", "(555) 123-4567"),
    ("a. user | x | y", "a. user"),
    ("b) user | x | y", "b) user"),
    ("J. Smith | x | y", "J. Smith"),
    ("\u0661. user | x | y", "\u0661. user"),
    ("-- user | x | y", "-- user"),
    ("> user | x | y", "> user"),
], ids=["decimal", "negative", "plus-sign", "bold", "italic", "dash-without-space",
        "number-without-space", "four-digits", "bracketed-number", "phone-number", "letter",
        "letter-and-bracket", "initial", "arabic-indic-digit", "double-dash", "quote"])
def test_text_that_only_looks_like_a_marker_stays_in_the_name(queue, line, entity):
    """Markdown numbers lists with ASCII digits only. Letters and bracketed numbers also
    start names and numbers, so they stay, as before."""
    [(found, *_)] = parsed(queue, line)
    assert found == entity


@pytest.mark.parametrize("line", [
    "(1) My city is Paris.", "a) My city is Paris.", "1000. My city is Paris.",
    "-My city is Paris.", "1.My city is Paris."])
def test_prose_after_something_that_is_not_a_marker_is_still_skipped(queue, line):
    assert parsed(queue, line) == []


@pytest.mark.parametrize("line, entity", [
    ("1. FC Köln | league | Bundesliga", "FC Köln"),
    ("- 1. FC Köln | league | Bundesliga", "1. FC Köln"),
    ("- - user | x | y", "- user"),
    ("1. 2. user | x | y", "2. user"),
], ids=["read-as-a-numbered-item", "a-marker-first-keeps-the-name", "two-bullets",
        "two-numbers"])
def test_only_one_marker_is_removed(queue, line, entity):
    """A name that starts like a marker can be written after one."""
    [(found, *_)] = parsed(queue, line)
    assert found == entity


@pytest.mark.parametrize("line", [
    "-", "- ", "*", "1.", "2)", "\u2022", "  -  ", "- - -", "* * *", "---",
    "- # comment", "- # a | b | c", "1. #heading | x | y", "* #",
    "- 2025-01-01: # a dated comment",
])
def test_a_bare_marker_or_a_listed_comment_proposes_nothing(queue, line):
    assert parsed(queue, line) == []


def test_text_of_bare_markers_is_unsupported_not_empty(queue):
    """As before, a marker alone is content no rule understands, not empty text."""
    assert queue.extract("-\n1.\n\u2022\n", source="chat") == []
    assert queue.list(status="all") == []


@pytest.mark.parametrize("line", [
    "- [ ] My application at Acme is submitted.",
    "- [x] My application at Acme is submitted.",
    "* [X] user | city | Paris",
    "1. [ ] user | city | Paris | 2026-01-01",
    "  - [x]\tMy city is Paris.",
    "[ ] user | city | Paris",
    "- [ ]",
    "+ [x]",
    "- [x] user | city | Paris",
    "- [ ]\tuser | city | Paris",
], ids=["unticked", "ticked", "capital-x", "numbered", "nested-with-a-tab", "no-marker",
        "empty-unticked", "empty-ticked", "ticked-pipe-line", "tab-after-the-box"])
def test_a_to_do_item_is_skipped(queue, line):
    """A to-do item is a task rather than a fact, and an unticked one is not done yet."""
    assert parsed(queue, line) == []


@pytest.mark.parametrize("line, entity", [
    ("- [?] user | x | y", "[?] user"),
    ("- [ ]user | x | y", "[ ]user"),
    ("- [] user | x | y", "[] user"),
    ("- [xx] user | x | y", "[xx] user"),
    ("- [1] user | x | y", "[1] user"),
], ids=["question-mark", "no-space-after", "empty-brackets", "two-letters", "footnote"])
def test_only_a_space_or_an_x_in_brackets_is_a_task_box(queue, line, entity):
    [(found, *_)] = parsed(queue, line)
    assert found == entity


def test_a_line_with_and_without_its_marker_is_one_statement(queue):
    """The evidence leaves the marker out, so the same dated line in another list style
    returns the pending proposal instead of a second one for the same time."""
    [first] = queue.extract("- user | city | Paris | 2026-01-01", source="chat")
    for line in ("user | city | Paris | 2026-01-01", "3) user | city | Paris | 2026-01-01",
                 "  \u2022 user | city | Paris | 2026-01-01"):
        assert [p.id for p in queue.extract(line, source="chat")] == [first.id]
    assert len(queue.list(status="all")) == 1


def test_an_undated_line_restated_in_another_list_style_is_the_pending_one(queue,
                                                                           ticking_clock):
    [first] = queue.extract("* My city is Paris.", source="chat")
    [again] = queue.extract("My city is Paris.", source="chat")
    assert again.id == first.id and not first.dated


@pytest.mark.parametrize("line", ["user | city | Paris", "My city is Paris."])
def test_one_line_in_several_list_styles_in_one_text_is_one_proposal(queue, line):
    """Without its marker the line is the same each time, so there is no same-time warning."""
    proposals = queue.extract("- {0}\n{0}\n1. {0}".format(line), source="chat",
                              observed_at="2026-10-01")
    assert [(p.entity, p.value, p.evidence, p.warnings) for p in proposals] == [
        ("user", "Paris", line, [])]
    assert len(queue.list()) == 1


def test_two_values_in_one_list_still_get_the_same_time_warning(queue):
    paris, london = queue.extract("- My city is Paris.\n- My city is London.", source="chat",
                                  observed_at="2026-10-01")
    assert all("same time" in warning for p in (paris, london) for warning in p.warnings)
    assert paris.warnings and london.warnings


def test_a_bulleted_line_continues_the_plain_history(store, queue):
    """The entity no longer starts a second history whose value would also look current."""
    london = store.add("user", "city", "London", valid_from="2026-01-01")
    [update] = queue.extract("- user | city | Paris | 2026-02-01", source="chat")
    assert (update.entity, update.kind, update.target_id, update.prior_value) == (
        "user", "update", london.id, "London")
    fact = queue.approve(update.id)
    assert (fact.entity, fact.evidence) == ("user", "user | city | Paris | 2026-02-01")
    assert [f.value for f in store.history("user", "city")] == ["London", "Paris"]
    assert [(f.entity, f.value) for f in store.snapshot(as_of="2026-03-01")] == [
        ("user", "Paris")]


@pytest.mark.parametrize("line, expected", [
    ("1. My city is Paris. I moved here.", []),
    ("- My doctor is Dr. Smith.", [("user", "doctor", "Dr. Smith")]),
    ("2) J. P. Morgan | city | New York", [("J. P. Morgan", "city", "New York")]),
    ("\N{EM DASH} My application at Acme is pending.", [("Acme application", "status", "pending")]),
], ids=["two-sentences", "abbreviation", "initials", "dash"])
def test_sentence_rules_read_the_line_after_its_marker(queue, line, expected):
    assert [found[:3] for found in parsed(queue, line)] == expected


def test_the_size_limit_counts_the_text_as_given(queue):
    """Markers count toward the 1 MB limit; a text of them at the limit is read."""
    assert queue.extract("- #\n" * 250_000, source="chat") == []  # Exactly 1,000,000.
    with pytest.raises(ValueError, match="1 MB extraction limit"):
        queue.extract("- #\n" * 250_000 + "-", source="chat")


def test_the_cli_reads_a_bulleted_notes_file(tmp_path, capsys):
    root = str(tmp_path / "store")
    assert cli.main(["--root", root, "init"]) == 0
    notes = tmp_path / "notes.md"
    notes.write_text("# Notes\n\n- user | snack | popcorn | 2026-11-01\n"
                     "1. 2026-11-01: My favourite drink is matcha.\n"
                     "- [ ] My favourite drink is coffee.\n", encoding="utf-8")
    capsys.readouterr()
    assert cli.main(["--root", root, "extract", "--file", str(notes), "--source", "notes"]) == 0
    proposals = json.loads(capsys.readouterr().out)
    assert [(p["entity"], p["attribute"], p["value"], p["evidence"]) for p in proposals] == [
        ("user", "snack", "popcorn", "user | snack | popcorn | 2026-11-01"),
        ("user", "favourite drink", "matcha", "2026-11-01: My favourite drink is matcha.")]
