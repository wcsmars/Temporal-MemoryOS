"""0.3.1: entity and attribute names compare by name_key everywhere.

"User" and "user", or "Favourite drink" and "favourite drink", used to start
separate histories in the store, so both values looked current, while the
search filter already ignored letter case. Now writes take the spelling a name
already has, reads join every spelling of it, extraction proposes the stored
spelling, and stores written before 0.3.1 with several spellings still load.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest
from hypothesis import given, strategies as st

from temporal_memory_os import cli, retrieval
from temporal_memory_os.demo import seed_demo
from temporal_memory_os.embeddings import HashingEmbedder
from temporal_memory_os.extraction import ProposalQueue, _fingerprint
from temporal_memory_os.models import Fact, name_key, normalize_time
from temporal_memory_os.retrieval import HybridRetriever
from temporal_memory_os.store import MemoryConflictError, MemoryStore, StoreIntegrityError

JAN, FEB, MAR, APR = "2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"


@pytest.fixture
def root(tmp_path):
    return tmp_path / "store"


@pytest.fixture
def store(root):
    with MemoryStore(root) as memory:
        yield memory


@pytest.fixture
def queue(store):
    return ProposalQueue(store)


def write_record(root, number, entity, attribute, value, valid_from, recorded_at):
    """A record file as 0.3.0 or a hand edit leaves it, in any spelling."""
    fact = Fact(id="{:032x}".format(number), entity=entity, attribute=attribute, value=value,
                valid_from=valid_from, recorded_at=recorded_at)
    folder = Path(root) / "memory" / fact.topic
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (fact.id + ".md")).write_text(MemoryStore._render(fact), encoding="utf-8")
    return fact


def legacy_store(root, same_start=False):
    """Two spellings of one history: "User / City" first, "user / city" later."""
    paris = write_record(root, 1, "User", "City", "Paris", JAN, JAN)
    london = write_record(root, 2, "user", "city", "London", JAN if same_start else MAR, FEB)
    return paris, london


def values(facts):
    return [fact.value for fact in facts]


# ==========================================================================
# The key
# ==========================================================================


@pytest.mark.parametrize("first, second", [
    ("User", "user"),
    ("Favourite drink", "favourite DRINK"),
    ("Stra\u00dfe", "STRASSE"),                       # German sharp s folds to "ss".
    ("\u00df", "SS"),
    ("\u0130stanbul", "Istanbul"),                    # Turkish dotted capital I.
    ("\u0130ZM\u0130R", "izmir"),
    ("\u0130stanbul", "i\u0307stanbul"),              # The dot written as a combining mark.
    ("\uff35\uff53\uff45\uff52", "user"),             # Full-width letters (NFKC).
    ("\ufb01le", "FILE"),                             # The "fi" ligature (NFKC).
    ("\u212a", "k"),                                  # Kelvin sign.
    ("caf\u00e9", "cafe\u0301"),                      # Composed and decomposed accents.
    (" user ", "user"),                               # Surrounding spaces.
    ("\u00a0user\u3000", "user"),                     # No-break and ideographic spaces.
    ("home  city", "home city"),                      # A doubled space.
    ("home\tcity", "home city"),
])
def test_names_that_differ_only_in_case_form_or_spacing_share_a_key(first, second):
    assert name_key(first) == name_key(second)


@pytest.mark.parametrize("first, second", [
    ("\u0131l\u0131k", "ilik"),     # Turkish dotless i is another letter.
    ("caf\u00e9", "cafe"),          # Accents count.
    ("user", "users"),
    ("New York", "NewYork"),
    ("home_city", "home city"),
])
def test_names_with_other_letters_keep_their_own_key(first, second):
    assert name_key(first) != name_key(second)


@given(st.text(max_size=12))
def test_the_key_of_a_key_is_itself(text):
    assert name_key(name_key(text)) == name_key(text)


@pytest.mark.parametrize("text", ["\u0130\u0307", "i\u0307\u0307", "I\u0307\u0307",
                                  "\u0130\u0307stanbul"])
def test_every_dot_above_after_an_i_goes_so_the_key_is_stable(text):
    key = name_key(text)
    assert "\u0307" not in key
    assert name_key(key) == key


def test_search_compares_entity_names_with_the_same_key():
    assert retrieval._name is name_key


# ==========================================================================
# Writes join the stored spelling
# ==========================================================================


def test_add_in_other_capitals_joins_the_existing_history(store):
    tea = store.add("user", "Favourite drink", "tea", valid_from=JAN)
    matcha = store.add("User", "favourite DRINK", "matcha", valid_from=MAR)
    assert (matcha.entity, matcha.attribute) == ("user", "Favourite drink")
    saved = MemoryStore._parse(Path("record.md"), store.path_for(matcha.id).read_bytes())
    assert saved == matcha  # The Markdown record has the stored spelling too.
    history = store.history("USER", "favourite drink")
    assert values(history) == ["tea", "matcha"]
    assert history[0].valid_to == normalize_time(MAR)
    # Only one value is current, whatever the capitals.
    assert values(store.snapshot(as_of=APR)) == ["matcha"]
    assert values(store.snapshot(as_of=FEB)) == ["tea"]
    assert store.get(tea.id) == tea


@pytest.mark.parametrize("stored, given_name", [
    ("Stra\u00dfe", "STRASSE"),
    ("\u0130stanbul office", "ISTANBUL OFFICE"),
    ("user", "\uff35\uff33\uff25\uff32"),
    ("user", "  user "),
    ("home city", "Home  City"),
])
def test_unicode_variants_of_a_name_join_its_history(store, stored, given_name):
    store.add(stored, "status", "open", valid_from=JAN)
    later = store.add(given_name, "STATUS", "closed", valid_from=MAR)
    assert (later.entity, later.attribute) == (stored, "status")
    assert values(store.history(given_name, "Status")) == ["open", "closed"]


def test_a_dotless_i_still_names_another_entity(store):
    store.add("\u0131l\u0131k", "taste", "warm", valid_from=JAN)
    other = store.add("ilik", "taste", "rich", valid_from=JAN)
    assert other.entity == "ilik"
    assert values(store.history("\u0131l\u0131k")) == ["warm"]
    assert values(store.history("ilik")) == ["rich"]


def test_a_record_joins_a_stored_pair_of_spellings_rather_than_a_new_mix(root):
    write_record(root, 1, "User", "City", "Paris", JAN, JAN)
    write_record(root, 2, "user", "drink", "tea", JAN, FEB)
    with MemoryStore(root) as store:
        # "user" is a stored spelling, but "city" is stored only as "User / City".
        fact = store.add("user", "CITY", "Rome", valid_from=MAR)
        assert (fact.entity, fact.attribute) == ("User", "City")
        fact = store.add("USER", "DRINK", "coffee", valid_from=MAR)
        assert (fact.entity, fact.attribute) == ("user", "drink")
        assert values(store.history("user", "city")) == ["Paris", "Rome"]
        assert values(store.history("user", "drink")) == ["tea", "coffee"]
        # A new attribute keeps its spelling. A stored entity spelling is kept;
        # another takes the spelling of the entity's latest record (coffee's).
        assert store.add("User", "Pet", "cat").entity == "User"
        fact = store.add("USER", "Food", "rice")
        assert (fact.entity, fact.attribute) == ("User", "Food")  # The cat was recorded last.


def test_a_spelling_in_use_wins_over_a_later_retracted_one(root):
    write_record(root, 1, "User", "city", "Paris", JAN, JAN)
    write_record(root, 2, "USER", "food", "rice", JAN, FEB)  # A typo, retracted below.
    with MemoryStore(root) as store:
        store.retract("{:032x}".format(2))
        fact = store.add("user", "drink", "tea", valid_from=MAR)
        assert (fact.entity, fact.attribute) == ("User", "drink")
        # With every record of a pair retracted, the latest one still names it.
        fact = store.add("user", "FOOD", "noodles", valid_from=MAR)
        assert (fact.entity, fact.attribute) == ("USER", "food")


def spelled(word, capitals):
    return "".join(letter.upper() if capital else letter for letter, capital in zip(word, capitals))


@given(writes=st.lists(st.tuples(st.lists(st.booleans(), min_size=4, max_size=4),
                                 st.lists(st.booleans(), min_size=4, max_size=4),
                                 st.integers(min_value=1, max_value=28)),
                       min_size=1, max_size=5, unique_by=lambda write: write[2]))
def test_any_mix_of_capitals_keeps_the_first_spelling_and_one_history(tmp_path_factory, writes):
    with MemoryStore(tmp_path_factory.mktemp("store")) as store:
        for entity_capitals, attribute_capitals, day in writes:
            store.add(spelled("user", entity_capitals), spelled("city", attribute_capitals),
                      str(day), valid_from="2026-01-{:02d}".format(day))
        first = (spelled("user", writes[0][0]), spelled("city", writes[0][1]))
        assert {(fact.entity, fact.attribute) for fact in store.all_facts()} == {first}
        days = sorted(day for *_, day in writes)
        assert values(store.history("USER", "CITY")) == [str(day) for day in days]
        assert values(store.snapshot()) == [str(days[-1])]


def test_corrections_and_retractions_keep_the_names_of_their_target(store):
    paris = store.add("user", "city", "Paris", valid_from=JAN)
    correction = store.correct(paris.id, "Lyon")
    tombstone = store.retract(correction.id)
    assert {(fact.entity, fact.attribute) for fact in (correction, tombstone)} == {
        ("user", "city")}


# ==========================================================================
# Reads agree on every spelling
# ==========================================================================


@pytest.mark.parametrize("asked", ["user", "User", "USER", " user ", "\uff55\uff53\uff45\uff52"])
def test_history_snapshot_audit_trail_and_search_agree_on_every_spelling(root, asked):
    paris, london = legacy_store(root)
    with MemoryStore(root) as store:
        assert [f.id for f in store.history(asked, "CITY")] == [paris.id, london.id]
        assert [f.id for f in store.history(asked)] == [paris.id, london.id]
        assert [f.id for f in store.audit_trail(asked, "city")] == [paris.id, london.id]
        assert [f.id for f in store.snapshot(as_of=FEB, entity=asked)] == [paris.id]
        assert [f.id for f in store.snapshot(as_of=APR, entity=asked)] == [london.id]
        found = HybridRetriever(store, embedder=HashingEmbedder()).search(
            "city", as_of=APR, entity=asked)
        assert [result.fact.id for result in found] == [london.id]


def test_names_of_other_entities_and_attributes_are_not_joined(store):
    store.add("user", "city", "Paris", valid_from=JAN)
    store.add("users", "city", "Oslo", valid_from=JAN)
    store.add("user", "city of birth", "Rome", valid_from=JAN)
    assert values(store.history("User", "City")) == ["Paris"]
    assert values(store.history("USERS")) == ["Oslo"]
    assert store.history("nobody") == [] and store.audit_trail("user", "town") == []


def test_blank_names_match_nothing(store):
    store.add("user", "city", "Paris", valid_from=JAN)
    assert store.snapshot(entity="") == store.snapshot(entity=" ") == []
    assert store.history("") == store.history(" ") == store.history("user", "") == []
    assert store.audit_trail("", "city") == store.audit_trail("user", " ") == []


def test_graph_search_finds_a_name_whatever_the_dot_on_its_capital_i(store):
    """name_key drops the dot of a Turkish capital "İ"; graph search must too."""
    store.add("\u0130stanbul office", "rent", "high", valid_from=JAN)
    store.add("Istanbul office", "status", "open", valid_from=JAN)  # Filed as "İstanbul office".
    retriever = HybridRetriever(store)
    for query in ("Istanbul office", "\u0130stanbul office", "istanbul OFFICE"):
        found = retriever.search(query, mode="graph", limit=5)
        assert sorted(result.fact.value for result in found) == ["high", "open"], query


def test_snapshot_order_is_unchanged_for_names_without_other_spellings(store):
    for entity in ("Zed", "alex", "Bob"):
        store.add(entity, "city", "Paris", valid_from=JAN)
    assert [fact.entity for fact in store.snapshot()] == ["Bob", "Zed", "alex"]


def test_reads_by_attribute_alone_or_by_names_that_are_not_text_still_answer(store):
    # Outside the typed API, but reachable from Python: no TypeError from name_key.
    store.add("user", "role", "analyst", valid_from=JAN)
    store.add("team", "Role", "research", valid_from=JAN)
    assert sorted(values(store.history(None, attribute=" ROLE "))) == ["analyst", "research"]
    assert store.history(42) == [] and store.snapshot(entity=42) == []
    assert store.history("user", attribute=42) == []


# ==========================================================================
# Stores written before 0.3.1 with several spellings
# ==========================================================================


def test_a_store_with_two_spellings_loads_and_shows_one_timeline(root):
    paris, london = legacy_store(root)
    with MemoryStore(root) as store:
        assert store.reindex() == 2  # The 0.3.0 rules still accept it.
        history = store.history("user", "city")
        assert [f.id for f in history] == [paris.id, london.id]
        assert history[0].valid_to == normalize_time(MAR)  # Ended by the other spelling.
        assert history[1].valid_to is None
        # The records themselves keep their own spelling.
        assert [(f.entity, f.attribute) for f in history] == [("User", "City"), ("user", "city")]
        assert values(store.history("user", "city", known_at=JAN)) == ["Paris"]


def test_spellings_apart_in_sorted_order_still_share_one_timeline(root):
    """Other names can sort between the spellings ("User" < "Zed" < "user"), so
    grouping only neighbouring rows would show two current values."""
    write_record(root, 1, "User", "City", "Paris", JAN, JAN)
    write_record(root, 2, "Zed", "city", "Oslo", JAN, JAN)
    write_record(root, 3, "user", "Zip", "75001", JAN, JAN)
    write_record(root, 4, "user", "city", "London", MAR, FEB)
    with MemoryStore(root) as store:
        current = store.snapshot(as_of=APR)
        assert sorted((name_key(f.entity), name_key(f.attribute), f.value) for f in current) == [
            ("user", "city", "London"), ("user", "zip", "75001"), ("zed", "city", "Oslo")]
        for name in ("User", "user", "USER", "Zed"):
            assert store.snapshot(as_of=APR, entity=name) == [
                fact for fact in current if name_key(fact.entity) == name_key(name)]
        assert values(store.history("USER", "CITY")) == ["Paris", "London"]


def test_spellings_sharing_a_start_show_the_record_learned_last(root):
    paris, london = legacy_store(root, same_start=True)
    write_record(root, 3, "USER", "CITY", "Rome", JAN, "2026-01-15")
    with MemoryStore(root) as store:
        assert [f.id for f in store.history("user", "city")] == [london.id]
        assert values(store.history("user", "city", known_at="2026-01-20")) == ["Rome"]
        assert values(store.history("user", "city", known_at=JAN)) == ["Paris"]
        assert values(store.snapshot(as_of=FEB)) == ["London"]
        assert values(store.audit_trail("User", "City")) == ["Paris", "Rome", "London"]
        # No interval of zero length is shown for the hidden records.
        assert all(f.valid_to is None or f.valid_to > f.valid_from
                   for f in store.history("user", known_at=MAR))


def test_either_spelling_can_be_corrected_or_retracted(root):
    paris, london = legacy_store(root)
    with MemoryStore(root) as store:
        rome = store.correct(paris.id, "Rome")
        assert (rome.entity, rome.attribute) == ("User", "City")
        assert values(store.history("USER", "CITY")) == ["Rome", "London"]
        tombstone = store.retract(london.id)
        assert (tombstone.entity, tombstone.attribute) == ("user", "city")
        assert store.snapshot(as_of=APR) == []
        assert values(store.snapshot(as_of=FEB)) == ["Rome"]
        assert [f.deleted for f in store.history("user", "city")] == [False, True]


def test_revising_either_record_of_a_shared_start_never_changes_which_is_shown(root):
    paris, london = legacy_store(root, same_start=True)
    with MemoryStore(root) as store:
        assert values(store.history("user", "city")) == ["London"]
        # The hidden record's revisions stay hidden: tidying it up changes nothing.
        rome = store.correct(paris.id, "Rome")
        assert values(store.history("user", "city")) == ["London"]
        store.retract(rome.id)
        assert values(store.history("user", "city")) == ["London"]
        # The shown record's revisions are shown.
        berlin = store.correct(london.id, "Berlin")
        assert values(store.snapshot(as_of=FEB)) == ["Berlin"]
        store.retract(berlin.id)
        assert store.snapshot(as_of=FEB) == []
        assert [f.deleted for f in store.history("user", "city")] == [True]
        assert values(store.history("user", "city", known_at=MAR)) == ["London"]


@given(spellings=st.integers(min_value=2, max_value=3),
       days=st.lists(st.integers(min_value=1, max_value=9), min_size=3, max_size=3),
       revisions=st.lists(st.tuples(st.integers(min_value=0, max_value=2), st.booleans()),
                          max_size=6))
def test_a_shared_start_shows_the_chain_begun_last_whatever_is_revised(tmp_path_factory,
                                                                       spellings, days,
                                                                       revisions):
    root = tmp_path_factory.mktemp("store")
    names = [("User", "City"), ("user", "city"), ("USER", "CITY")][:spellings]
    origins = [write_record(root, number + 1, entity, attribute, "v{}".format(number), JAN,
                            "2026-01-0{}".format(days[number]))
               for number, (entity, attribute) in enumerate(names)]
    shown = max(origins, key=lambda fact: (fact.recorded_at, fact.id))
    with MemoryStore(root) as store:
        heads = list(origins)
        for number, (which, retract) in enumerate(revisions):
            head = heads[which % spellings]
            if head.deleted:
                continue  # A retraction can be corrected, but keep this simple.
            heads[which % spellings] = (store.retract(head.id) if retract
                                        else store.correct(head.id, "r{}".format(number)))
        head = heads[origins.index(shown)]
        history = store.history("user", "city")
        assert [(f.id, f.deleted) for f in history] == [(head.id, head.deleted)]
        assert values(store.snapshot(as_of=FEB)) == ([] if head.deleted else [head.value])


def test_a_new_record_cannot_take_a_start_held_by_another_spelling(root):
    legacy_store(root)
    with MemoryStore(root) as store:
        files = sorted(store.memory_dir.rglob("*.md"))
        # "user" and "city" are both stored as spelled, but "User / City" holds January.
        with pytest.raises(MemoryConflictError, match=r"Duplicate effective time .*correct\(\)"
                           ) as raised:
            store.add("user", "city", "Rome", valid_from=JAN)
        assert "User / City" in str(raised.value) and str(root) not in str(raised.value)
        # The spelling that holds it is refused by the rule for one spelling.
        with pytest.raises(MemoryConflictError, match="Duplicate effective time"):
            store.add("USER", "CITY", "Rome", valid_from=JAN)
        assert sorted(store.memory_dir.rglob("*.md")) == files
        # A retracted record still owns its start.
        store.retract(store.history("user", "city")[1].id)
        with pytest.raises(MemoryConflictError, match="Duplicate effective time"):
            store.add("User", "City", "Rome", valid_from=MAR)
        assert store.add("user", "city", "Rome", valid_from=APR).value == "Rome"


# ==========================================================================
# Extraction proposes, and approves into, the stored spelling
# ==========================================================================


def test_proposals_classify_against_the_stored_spelling_and_approve_into_it(store, queue):
    tea = store.add("user", "favourite drink", "tea", valid_from=JAN)
    [update] = queue.extract("User | Favourite Drink | matcha | 2026-03-01", source="chat")
    assert (update.entity, update.attribute) == ("user", "favourite drink")
    assert (update.kind, update.target_id, update.prior_value) == ("update", tea.id, "tea")
    approved = queue.approve(update.id)
    assert (approved.entity, approved.attribute) == ("user", "favourite drink")
    assert values(store.history("user", "favourite drink")) == ["tea", "matcha"]
    [same] = queue.extract("USER | FAVOURITE DRINK | matcha | 2026-03-01", source="chat")
    assert (same.kind, same.target_id) == ("duplicate", approved.id)


def test_prose_lines_take_the_stored_spelling_of_user(store, queue):
    store.add("User", "Favourite drink", "tea", valid_from=JAN)
    [proposal] = queue.extract("2026-03-01: My favourite drink is matcha.", source="chat")
    assert (proposal.entity, proposal.attribute, proposal.kind) == (
        "User", "Favourite drink", "update")


def test_a_retraction_is_restored_through_extraction_in_other_capitals(store, queue):
    paris = store.add("user", "city", "Paris", valid_from=JAN)
    tombstone = store.retract(paris.id)
    [restore] = queue.extract("User | City | Lyon | 2026-01-01", source="chat")
    assert (restore.entity, restore.attribute) == ("user", "city")
    assert (restore.kind, restore.target_id) == ("correct", tombstone.id)
    restored = queue.approve(restore.id)
    assert restored.supersedes == tombstone.id
    assert values(store.snapshot(as_of=FEB)) == ["Lyon"]


def test_unstored_names_take_the_queue_spelling_then_that_of_earlier_lines(store, queue):
    [first] = queue.extract("Alex | Home City | Paris | 2026-01-01", source="chat")
    [queued] = queue.extract("alex | home city | London | 2026-02-01", source="chat")
    assert (queued.entity, queued.attribute) == ("Alex", "Home City")
    lines = queue.extract("Sam | Job | baker | 2026-01-01\nSAM | JOB | pilot | 2026-02-01",
                          source="chat")
    assert [(p.entity, p.attribute) for p in lines] == [("Sam", "Job")] * 2
    for proposal in [first, queued, *lines]:
        queue.approve(proposal.id)
    assert values(store.history("alex", "home city")) == ["Paris", "London"]
    assert values(store.history("sam", "job")) == ["baker", "pilot"]


def test_a_rejected_proposal_does_not_set_the_spelling(queue):
    [shouted] = queue.extract("ALEX | CITY | Paris | 2026-01-01", source="chat")
    queue.reject(shouted.id)
    [proposal] = queue.extract("Alex | city | Paris | 2026-01-01", source="chat")
    assert (proposal.entity, proposal.attribute) == ("Alex", "city")


def test_lines_for_one_time_in_two_spellings_are_flagged(queue):
    paris, london = queue.extract("Alex | city | Paris | 2026-01-01\n"
                                  "alex | City | London | 2026-01-01", source="chat")
    assert (london.entity, london.attribute) == ("Alex", "city")
    assert any("same time" in warning for warning in london.warnings)


def test_a_queued_line_in_other_capitals_is_returned_not_proposed_again(store, queue):
    """A proposal that kept the chat's capitals (as 0.3.0 did) still matches its line."""
    line = "User | city | Paris | 2026-01-01"
    [first] = queue.extract(line, source="chat")
    assert first.entity == "User"
    store.add("user", "city", "London", valid_from=FEB, source="manual")
    # Rewritten to the stored spelling, the line is still the queued statement.
    [again] = queue.extract(line, source="chat")
    assert again.id == first.id and len(queue.list()) == 1
    approved = queue.approve(first.id)
    assert approved.entity == "user"  # Approved into the stored spelling.
    assert values(store.history("user", "city")) == ["Paris", "London"]


def test_a_proposal_reviewed_as_new_is_stale_once_its_name_matches_a_history(store, queue):
    """Approving it would change a value the reviewer never saw, so it is refused."""
    [proposal] = queue.extract("User | city | Paris | 2026-03-01", source="chat")
    assert proposal.kind == "add"
    store.add("user", "city", "London", valid_from=JAN, source="manual")
    with pytest.raises(MemoryConflictError, match="extract again"):
        queue.approve(proposal.id)
    queue.reject(proposal.id)
    [update] = queue.extract("User | city | Paris | 2026-03-01", source="chat")
    assert (update.entity, update.kind, update.prior_value) == ("user", "update", "London")


def test_a_line_finds_the_current_value_under_another_spelling_of_its_attribute(root):
    """In a store with two spellings, the line keeps its own stored spelling ("city"),
    while the value in effect is filed under the other one ("City")."""
    write_record(root, 1, "user", "city", "London", JAN, JAN)
    paris = write_record(root, 2, "user", "City", "Paris", MAR, MAR)
    with MemoryStore(root) as store:
        queue = ProposalQueue(store)
        [update] = queue.extract("user | city | Rome | 2026-04-01", source="chat")
        assert (update.attribute, update.kind, update.target_id) == ("city", "update", paris.id)


def test_an_approval_after_a_newer_record_checks_the_reviewed_one_under_any_spelling(root):
    """The record the reviewer saw is "City"; the proposal is spelled "city"."""
    write_record(root, 1, "user", "city", "Lisbon", "2025-12-01", "2025-12-01")
    london = write_record(root, 2, "user", "City", "London", JAN, JAN)
    with MemoryStore(root) as store:
        queue = ProposalQueue(store)
        [update] = queue.extract("user | city | Paris | 2026-03-01", source="chat")
        assert (update.attribute, update.target_id) == ("city", london.id)
        # A value approved from the same source after this proposal, effective before it.
        store.add("user", "city", "Oslo", valid_from=FEB, source="chat")
        assert queue.approve(update.id).value == "Paris"
        assert values(store.history("user", "city")) == ["Lisbon", "London", "Oslo", "Paris"]


# ==========================================================================
# Interfaces
# ==========================================================================


def run(capsys, *arguments):
    code = cli.main(list(arguments))
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out.strip() else None, captured.err


def test_cli_add_and_history_use_one_history_for_every_spelling(root, capsys):
    assert run(capsys, "--root", str(root), "init")[0] == 0
    code, first, error = run(capsys, "--root", str(root), "add", "User", "City", "Paris",
                             "--from", JAN)
    assert (code, error) == (0, "")  # Filed as given: nothing to say.
    code, second, error = run(capsys, "--root", str(root), "add", "user", "city", "London",
                              "--from", MAR)
    assert code == 0 and (second["entity"], second["attribute"]) == ("User", "City")
    assert json.loads(error) == {"note": "Recorded as User / City, the spelling memory already "
                                         "uses for these names (letter case, Unicode "
                                         "compatibility forms and spacing do not count)"}
    code, history, _ = run(capsys, "--root", str(root), "history", "USER", "--attribute", "city")
    assert code == 0 and [fact["value"] for fact in history] == ["Paris", "London"]
    code, _, error = run(capsys, "--root", str(root), "add", "user", "CITY", "Rome",
                         "--from", JAN)
    assert code == 1 and "Duplicate effective time" in json.loads(error)["error"]


def test_http_api_adds_and_reads_every_spelling_as_one_history(root):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from temporal_memory_os.api import create_app

    MemoryStore(root).close()  # The API never creates a memory directory.
    with TestClient(create_app(root), base_url="http://localhost") as client:
        first = client.post("/api/facts", json={"entity": "User", "attribute": "City",
                                                "value": "Paris", "valid_from": JAN})
        second = client.post("/api/facts", json={"entity": "user", "attribute": "city",
                                                 "value": "London", "valid_from": MAR})
        assert first.status_code == second.status_code == 201
        assert (second.json()["entity"], second.json()["attribute"]) == ("User", "City")
        history = client.get("/api/history", params={"entity": "USER", "attribute": "city"})
        assert [fact["value"] for fact in history.json()["facts"]] == ["Paris", "London"]
        current = client.get("/api/facts", params={"entity": "user", "as_of": APR})
        assert [fact["value"] for fact in current.json()["facts"]] == ["London"]
        clash = client.post("/api/facts", json={"entity": "USER", "attribute": "CITY",
                                                "value": "Rome", "valid_from": JAN})
        assert clash.status_code == 409


def test_mcp_history_and_search_find_every_spelling(root):
    pytest.importorskip("mcp")
    import asyncio
    from mcp.shared.memory import create_connected_server_and_client_session
    from temporal_memory_os.mcp_server import create_server

    paris, london = legacy_store(root)
    MemoryStore(root).close()

    async def scenario():
        async with create_connected_server_and_client_session(
                create_server(root), raise_exceptions=True) as session:
            history = await session.call_tool("memory_history",
                                              {"entity": "USER", "attribute": "City"})
            found = await session.call_tool("search", {"query": "city", "entity": "user",
                                                       "as_of": APR})
            return history, found

    history, found = asyncio.run(scenario())
    assert not history.isError and not found.isError
    facts = (history.structuredContent or json.loads(history.content[0].text))["facts"]
    results = (found.structuredContent or json.loads(found.content[0].text))["results"]
    assert [fact["id"] for fact in facts] == [paris.id, london.id]
    assert [result["id"] for result in results] == [london.id]


def test_demo_repeats_after_its_names_joined_a_history_spelled_otherwise(store):
    store.add("alex", "Preferred drink", "water", valid_from="2026-07-01")
    assert seed_demo(store)["added"] == 7
    assert values(store.history("Alex", "preferred drink")) == ["water", "tea", "decaf coffee"]
    again = seed_demo(store)
    assert again["added"] == 0 and "skipped_existing" not in again


# ==========================================================================
# The list of stored names follows every change to the index
# ==========================================================================


def test_names_written_through_another_connection_are_found(root):
    with MemoryStore(root) as reader, MemoryStore(root) as writer:
        assert reader.history("alex") == []  # Reads the list of names.
        writer.add("Alex", "city", "Paris", valid_from=JAN)
        assert values(reader.history("ALEX")) == ["Paris"]
        assert reader.add("alex", "city", "Rome", valid_from=MAR).entity == "Alex"


def test_names_written_by_this_store_and_by_hand_are_found(root):
    with MemoryStore(root) as store:
        assert store.history("bob") == []
        store.add("Bob", "city", "Paris", valid_from=JAN)  # The same connection.
        assert values(store.history("BOB")) == ["Paris"]
        write_record(root, 7, "Carol", "city", "Oslo", JAN, JAN)
        assert store.refresh() is True  # A rebuild by this connection.
        assert values(store.history("carol")) == ["Oslo"]


def test_names_rolled_back_with_a_failed_write_are_forgotten(store):
    with pytest.raises(RuntimeError):
        with store._transaction():
            store.add("Zed", "city", "Paris", valid_from=JAN)
            assert values(store.history("zed")) == ["Paris"]
            raise RuntimeError("stop")
    assert store.history("zed") == []
    assert store.add("zed", "city", "Oslo", valid_from=JAN).entity == "zed"
    # "Zed" was never stored, so it takes the spelling that is.
    assert store.add("Zed", "city", "Rome", valid_from=MAR).entity == "zed"


def test_a_spelling_whose_records_were_removed_by_hand_is_forgotten(root):
    paris, london = legacy_store(root)
    with MemoryStore(root) as store:
        assert len(store.history("user", "city")) == 2  # Reads the list of names.
        store.path_for(paris.id).unlink()
        assert store.refresh() is True
        fact = store.add("User", "City", "Rome", valid_from=APR)
        assert (fact.entity, fact.attribute) == ("user", "city")


# Guards for how extraction keys names: a line in other capitals, and the twin
# that a release before 0.3.1 queued for the same line.
DRINK = "User | Favourite drink | latte | 2026-11-01"


def approve_interrupted(queue, proposal_id, monkeypatch):
    """An approval stopped after its fact was written, before the proposal was marked."""
    def stop(self, proposal):
        raise RuntimeError("process stopped after the fact was written")

    with monkeypatch.context() as patch:
        patch.setattr(ProposalQueue, "_save", stop)
        with pytest.raises(RuntimeError):
            queue.approve(proposal_id)


def queue_as_before_031(queue, proposal, **fields):
    """The proposal a release before 0.3.1 queued for the same line, which compared
    names exactly: its names as written and the preview of that exact spelling."""
    data = proposal.to_dict()
    data.update(id=uuid4().hex, warnings=[], **fields)
    (queue.directory / (data["id"] + ".json")).write_text(json.dumps(data), encoding="utf-8")
    return queue.get(data["id"])


def twins(store, queue):
    """B, extracted now in the stored spelling, and A, its twin from before 0.3.1,
    previewed as new because that lookup missed "user / favourite drink"."""
    store.add("user", "favourite drink", "matcha", valid_from=JAN)
    [b] = queue.extract(DRINK, source="chat")
    assert (b.entity, b.attribute, b.kind) == ("user", "favourite drink", "update")
    a = queue_as_before_031(queue, b, entity="User", attribute="Favourite drink", kind="add",
                            target_id=None, target_fingerprint=None, prior_value=None)
    return b, a


def test_an_undated_line_in_capitals_extracted_again_returns_its_proposal(
        store, queue, ticking_clock):
    [first] = queue.extract("User | City | Paris", source="chat")
    [again] = queue.extract("User | City | Paris", source="chat")
    assert again.id == first.id
    store.add("user", "city", "Oslo", valid_from="2025-06-01")  # Stored in other capitals.
    [still] = queue.extract("User | City | Paris", source="chat")
    assert still.id == first.id and len(queue.list()) == 1


def test_a_twin_from_before_031_can_be_rejected_after_the_other_is_approved(store, queue):
    b, a = twins(store, queue)
    queue.approve(b.id)
    assert queue.reject(a.id).status == "rejected"


def test_a_twin_from_before_031_does_not_claim_the_others_fact(store, queue):
    b, a = twins(store, queue)
    written = queue.approve(b.id)
    with pytest.raises(MemoryConflictError):
        queue.approve(a.id)
    assert queue.get(a.id).status == "pending"
    assert [fact.value for fact in store.all_facts()] == ["matcha", written.value]


def test_an_interrupted_twin_can_be_rejected_while_the_other_claims_its_fact(
        store, queue, monkeypatch):
    b, a = twins(store, queue)
    approve_interrupted(queue, b.id, monkeypatch)
    assert queue.reject(b.id).status == "rejected"
    claimed = queue.approve(a.id)
    assert [fact.id for fact in store.all_facts() if fact.value == "latte"] == [claimed.id]


def test_the_stored_spelling_wins_over_a_queued_one(store, queue):
    queue.extract("Alex | City | Paris | 2026-01-01", source="chat")
    store.add("alex", "city", "Oslo", valid_from="2025-06-01")
    [rome] = queue.extract("ALEX | CITY | Rome | 2026-03-01", source="chat")
    assert (rome.entity, rome.attribute) == ("alex", "city")


def test_a_twin_from_before_031_names_the_record_its_preview_missed(store, queue):
    b, a = twins(store, queue)
    with pytest.raises(MemoryConflictError, match=(
            r"^Memory holds user / favourite drink, recorded before this proposal was made but "
            r"not shown in its preview \(a release before 0\.3\.1, which compared names "
            r"exactly, including letter case, may have made the proposal\); reject it and "
            r"extract again$")):
        queue.approve(a.id)
    assert queue.get(a.id).status == "pending"
    assert [fact.value for fact in store.all_facts()] == ["matcha"]
    assert queue.approve(b.id).value == "latte"  # Extracted again, it updates matcha.


@pytest.mark.parametrize("earlier, value", [(DRINK, "matcha"), (DRINK, "latte"),
                                            (DRINK.replace("2026-11-01", JAN), "latte")])
def test_a_proposal_from_before_031_names_a_same_source_record_its_preview_missed(
        store, queue, earlier, value):
    """A line of the same source approved before the proposal was made, in other
    capitals: at its date (a correction or a duplicate) or earlier with its value."""
    queue.approve(queue.extract(earlier, source="chat")[0].id)
    [now] = queue.extract("user | favourite drink | {} | 2026-11-01".format(value),
                          source="chat")
    before = queue_as_before_031(queue, now, entity="user", attribute="favourite drink",
                                 kind="add", target_id=None, target_fingerprint=None,
                                 prior_value=None)
    with pytest.raises(MemoryConflictError, match=(
            r"^Memory holds User / Favourite drink, recorded before this proposal was made")):
        queue.approve(before.id)
    assert queue.get(before.id).status == "pending"


def test_an_update_from_before_031_names_the_record_its_preview_missed(root):
    tea = write_record(root, 1, "User", "drink", "tea", JAN, JAN)
    write_record(root, 2, "user", "drink", "coffee", MAR, FEB)  # Before the proposal.
    with MemoryStore(root) as store:
        queue = ProposalQueue(store)
        [now] = queue.extract("User | drink | latte | 2026-11-01", source="chat")
        # "User / drink" is stored exactly as the line spells it, so it is kept.
        assert (now.entity, now.kind, now.prior_value) == ("User", "update", "coffee")
        # The preview of "User / drink" alone, as before 0.3.1: an update of tea.
        before = queue_as_before_031(
            queue, now, entity="User", target_id=tea.id, prior_value="tea",
            target_fingerprint=_fingerprint(store.get(tea.id)))
        with pytest.raises(MemoryConflictError, match="^Memory holds user / drink, recorded"):
            queue.approve(before.id)
        # After an edit of the record that preview showed, the cause is not known.
        edited = queue_as_before_031(queue, before, target_fingerprint="0" * 64)
        with pytest.raises(MemoryConflictError, match="^Memory changed since this proposal"):
            queue.approve(edited.id)


def test_a_record_that_arrives_with_an_earlier_recorded_time_is_named_too(store, queue):
    """A record restored from a backup, or added with an earlier recorded_at, also
    reaches approval unseen, so the refusal names it without blaming an older release."""
    [springfield] = queue.extract("alex | home city | Springfield | 2026-09-01", source="chat")
    store.add("Alex", "Home City", "Shelbyville", valid_from="2026-09-01", recorded_at=JAN)
    with pytest.raises(MemoryConflictError, match=(
            r"^Memory holds Alex / Home City, recorded before this proposal was made but not "
            r"shown in its preview \(a release before 0\.3\.1, .* may have made the proposal\)")):
        queue.approve(springfield.id)
    assert queue.get(springfield.id).status == "pending"


def test_a_record_in_another_spelling_written_after_the_proposal_is_a_change(store, queue):
    [paris] = queue.extract("alex | city | Paris | 2026-03-01", source="chat")
    store.add("Alex", "City", "Oslo", valid_from=JAN)  # Nothing stored yet: kept as given.
    with pytest.raises(MemoryConflictError, match="^Memory changed since this proposal"):
        queue.approve(paris.id)


def test_a_load_error_says_a_correction_must_spell_its_targets_names_exactly(store):
    """A correction edited by hand to spell its target's entity in other capitals stops
    the load with the full message: the names must match exactly, including letter case,
    followed by the correction's file and then its target's."""
    first = store.add("Ada", "employer", "Analytical Engines Ltd", valid_from=JAN,
                      recorded_at=JAN)
    second = store.correct(first.id, "Difference Works", recorded_at=FEB)
    files = [store.path_for(fact.id).relative_to(store.root).as_posix()
             for fact in (second, first)]
    path = store.path_for(second.id)
    path.write_text(path.read_text(encoding="utf-8").replace('"entity": "Ada"',
                                                             '"entity": "ada"'),
                    encoding="utf-8")
    with pytest.raises(StoreIntegrityError) as caught:
        store.reindex()
    assert str(caught.value) == (
        "Corrections must retain entity, attribute, and valid_from exactly, including "
        "letter case ({}, {})".format(*files))


def test_a_start_held_under_another_spelling_of_only_one_name_is_refused(root):
    """A second spelling of just the attribute, or of just the entity, holds the start:
    the new record would hide it all the same, so it is refused and nothing is filed."""
    write_record(root, 1, "user", "City", "Paris", JAN, JAN)
    write_record(root, 2, "user", "city", "London", MAR, FEB)
    write_record(root, 3, "Pet", "name", "Rex", JAN, JAN)
    write_record(root, 4, "pet", "name", "Max", MAR, FEB)
    with MemoryStore(root) as store:
        files = sorted(store.memory_dir.rglob("*.md"))
        # "user / city" is stored as spelled; only "City" differs, and it holds January.
        with pytest.raises(MemoryConflictError, match="Duplicate effective time .*user / City"):
            store.add("user", "city", "Rome", valid_from=JAN)
        # "pet / name" is stored as spelled; only "Pet" differs, and it holds January.
        with pytest.raises(MemoryConflictError, match="Duplicate effective time .*Pet / name"):
            store.add("pet", "name", "Bo", valid_from=JAN)
        assert sorted(store.memory_dir.rglob("*.md")) == files
        assert values(store.history("user", "city")) == ["Paris", "London"]
        assert values(store.history("pet", "name")) == ["Rex", "Max"]


def test_the_refusal_names_the_names_given_and_the_spelling_that_holds_the_start(root):
    """The message says which names were refused, how the record that holds the start is
    spelled, and to correct that record instead."""
    legacy_store(root)
    with MemoryStore(root) as store:
        with pytest.raises(MemoryConflictError) as raised:
            store.add("user", "city", "Rome", valid_from=JAN)
    assert str(raised.value) == ("Duplicate effective time for user / city (held by a record "
                                 "spelled User / City); use correct()")


def test_a_record_recorded_at_the_instant_the_proposal_was_made_is_a_change(store, queue):
    """A record in another spelling recorded at the very instant the proposal was made
    was not recorded before it, so approval says memory changed instead of naming it."""
    [paris] = queue.extract("alex | city | Paris | 2026-03-01", source="chat")
    store.add("Alex", "City", "Oslo", valid_from="2026-03-01", recorded_at=paris.created_at)
    with pytest.raises(MemoryConflictError, match="^Memory changed since this proposal"):
        queue.approve(paris.id)
    assert queue.get(paris.id).status == "pending"


@pytest.mark.parametrize("column", ["deleted", "recorded_at"])
def test_a_column_missing_when_choosing_a_stored_spelling_is_reported_as_index_damage(
        root, capsys, column):
    """Extracting a line whose names memory stores in another spelling reads index columns
    that opening the store does not. If the index lacks one (a damaged name, or one
    renamed by another program), the command stops with the damaged-index message, which
    names the missing column as the engine spells it and gives the repair command."""
    with MemoryStore(root) as store:
        store.add("User", "City", "Paris", valid_from=JAN)
    with closing(sqlite3.connect(str(root / ".memoryos" / "index.sqlite3"))) as connection:
        connection.execute("ALTER TABLE facts RENAME COLUMN {0} TO {0}_old".format(column))
        connection.commit()
    code, output, error = run(capsys, "--root", str(root), "extract", "--source", "chat",
                              "--text", "user | city | London | 2026-03-01")
    assert (code, output) == (1, None)
    assert json.loads(error) == {"error": (
        "The index .memoryos/index.sqlite3 is damaged (no such column: {}). Stop other "
        "programs using this memory directory, then rebuild the index from the Markdown "
        "records by running reindex with --root set to this memory directory (the same "
        "--root or MEMORYOS_HOME used to open it), for example "
        "`memoryos --root <memory directory> reindex`.".format(column))}


def test_a_repeated_lookup_does_not_read_every_stored_name_again(root):
    """A lookup by name first finds every stored spelling of it. The names are read
    once and kept until the index changes, so a repeated history, snapshot or audit
    trail does as much SQLite work with hundreds of other entities stored as with
    none, instead of slowing down as memory grows."""
    def work(store):
        steps = [0]

        def step():
            steps[0] += 1
            return 0  # Carry on.

        store.connection.set_progress_handler(step, 1)
        try:
            assert values(store.history("USER", "CITY")) == ["Paris"]
            assert values(store.snapshot(entity="uSeR")) == ["Paris"]
            assert values(store.audit_trail("User", "city")) == ["Paris"]
        finally:
            store.connection.set_progress_handler(None, 1)
        return steps[0]

    write_record(root, 1, "User", "City", "Paris", JAN, JAN)
    with MemoryStore(root) as store:
        work(store)  # The first lookups read the names.
        alone = work(store)
        for number in range(300):
            write_record(root, 2 + number, "Person {}".format(number), "city", "Oslo", JAN, JAN)
        assert store.refresh() is True
        work(store)
        crowded = work(store)
    assert crowded - alone < 100, (alone, crowded)  # Far fewer steps than names added.
