"""Validation behavior of the record model that mutation testing showed no
other test checked: UTC offset bounds, naive datetimes, non-string topics,
and errors that name the rejected input.
"""

import json
import os
import re
import time
from datetime import datetime

import pytest

from temporal_memory_os import cli
from temporal_memory_os.extraction import ProposalQueue
from temporal_memory_os.models import Fact, normalize_time, validate_topic
from temporal_memory_os.store import MemoryStore, StoreIntegrityError


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "store") as memory:
        yield memory


VALID = dict(id="a" * 32, entity="person", attribute="drink", value="tea",
             valid_from="2025-01-01", recorded_at="2025-01-01")


def rewrite(store, fact_id, **changes):
    """Edit a record's canonical JSON by hand, as a user with a text editor would."""
    path = store.path_for(fact_id)
    text = path.read_text(encoding="utf-8")
    block = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    data = json.loads(block.group(1))
    data.update(changes)
    path.write_text(text.replace(block.group(1), json.dumps(data, indent=2)), encoding="utf-8")


def cli_error(capsys, *arguments):
    """Run the CLI in this process and return (exit code, the JSON error message)."""
    code = cli.main(list(arguments))
    return code, json.loads(capsys.readouterr().err)["error"]


# Time strings: UTC offset bounds


@pytest.mark.parametrize("text, expected", [
    ("2025-01-02T00:00+23:59", "2025-01-01T00:01:00.000000Z"),
    ("2025-01-01T00:00-23:59", "2025-01-01T23:59:00.000000Z"),
    ("2025-01-01T12:00+23:00", "2024-12-31T13:00:00.000000Z"),
    ("2025-01-01T12:00+05:59", "2025-01-01T06:01:00.000000Z"),
    ("2025-01-01T12:00-00:59", "2025-01-01T12:59:00.000000Z"),
])
def test_offsets_up_to_23_hours_and_59_minutes_are_accepted(text, expected):
    """Offsets with 23 hours or 59 minutes are in range and normalize to the same UTC instant."""
    assert normalize_time(text) == expected


@pytest.mark.parametrize("text", [
    "2025-01-01T08:30+00:60",
    "2025-01-01T08:30-05:60",
    "2025-01-01T08:30+00:99",
    "2025-01-01T08:30+23:60",
])
def test_offset_minutes_above_59_are_rejected(text):
    """An offset whose minutes exceed 59 is invalid, even when the total is under 24 hours."""
    with pytest.raises(ValueError):
        normalize_time(text)
    with pytest.raises(ValueError):
        Fact(**{**VALID, "valid_from": text})


# Time strings: errors name the rejected input


@pytest.mark.parametrize("text", [
    "2025-1-1",                      # outside the grammar
    "2025-01-01T08:30+0800",         # compact offset, outside the grammar
    "2025-01-01T08:30+24:00",        # offset hours out of range
    "2025-01-01T08:30-24:30",        # offset hours out of range, negative
    "2025-01-01T08:30+00:60",        # offset minutes out of range
    "2025-02-30",                    # no such day
])
def test_time_errors_name_the_rejected_input(text):
    """A rejected time string is quoted back in the error, so the caller can tell which one failed."""
    with pytest.raises(ValueError) as caught:
        normalize_time(text)
    assert text in str(caught.value)


@pytest.mark.parametrize("bad", ["2025-1-1", "2025-01-01T08:30+24:00"])
def test_cli_error_names_the_malformed_one_of_two_time_arguments(tmp_path, capsys, bad):
    """With a valid --as-of and a malformed --known-at, the CLI JSON error quotes the malformed one."""
    code, error = cli_error(capsys, "--root", str(tmp_path), "search", "tea",
                            "--as-of", "2025-01-01", "--known-at", bad)
    assert code == 1
    assert bad in error


# Topics


@pytest.mark.parametrize("topic", [7, 3.5, ["general"], ("career", "jobs"), b"general"],
                         ids=["int", "float", "list", "tuple", "bytes"])
def test_non_string_topics_are_rejected_with_value_error(topic):
    """A truthy non-string topic is a ValueError from validate_topic and Fact, not a crash."""
    with pytest.raises(ValueError):
        validate_topic(topic)
    with pytest.raises(ValueError):
        Fact(**VALID, topic=topic)


def test_store_and_queue_reject_a_non_string_topic_without_writing(store):
    """store.add and ProposalQueue.extract reject a numeric topic as invalid data and write nothing."""
    queue = ProposalQueue(store)
    with pytest.raises(ValueError):
        store.add("person", "drink", "tea", valid_from="2025-01-01", topic=7)
    with pytest.raises(ValueError):
        queue.extract("person | drink | tea", source="chat", topic=7)
    assert store.all_facts() == []
    assert queue.list(status="all") == []


# Fact.from_dict


def test_from_dict_error_names_the_unknown_field():
    """An unexpected key in fact metadata is named in the error."""
    with pytest.raises(ValueError) as caught:
        Fact.from_dict({**VALID, "colour": "blue"})
    assert "colour" in str(caught.value)


def test_reindex_error_names_the_file_and_the_unknown_field(store):
    """A hand-edited record with an unknown key fails the rebuild, naming the file and the key."""
    fact = store.add("person", "drink", "tea", valid_from="2025-01-01")
    relative = store.path_for(fact.id).relative_to(store.root).as_posix()
    rewrite(store, fact.id, colour="blue")
    with pytest.raises(StoreIntegrityError) as caught:
        store.reindex()
    message = str(caught.value)
    assert relative in message and str(store.root) not in message
    assert "colour" in message


# ==========================================================================
# Naive datetimes mean UTC; out-of-range errors quote the input
# ==========================================================================


needs_tzset = pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is Unix-only")


# POSIX zone strings carry their own rules, so they work without a tz database installed.
@pytest.fixture(params=[
    pytest.param("HKT-8", id="east-of-utc"),
    pytest.param("EST5EDT,M3.2.0,M11.1.0", id="west-of-utc-with-dst"),
])
def local_zone(request):
    """Run the test with the process's local time zone set to a zone that is not UTC."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = request.param
    time.tzset()
    yield request.param
    if previous is None:
        del os.environ["TZ"]
    else:
        os.environ["TZ"] = previous
    time.tzset()


@needs_tzset
def test_naive_datetime_means_utc_whatever_the_local_zone(local_zone):
    """A datetime without a zone is read as UTC, not as the host's local time."""
    assert normalize_time(datetime(2025, 1, 1, 8, 30)) == "2025-01-01T08:30:00.000000Z"
    assert normalize_time(datetime(2025, 7, 1, 23, 59, 59, 5)) == "2025-07-01T23:59:59.000005Z"


@needs_tzset
def test_store_records_a_naive_datetime_as_utc_whatever_the_local_zone(tmp_path, local_zone):
    """A naive valid_from given to the store is written and read back as the same UTC wall time."""
    with MemoryStore(tmp_path / "store") as store:
        fact = store.add("person", "drink", "tea", valid_from=datetime(2025, 1, 1, 8, 30),
                         recorded_at=datetime(2025, 1, 2))
    assert fact.valid_from == "2025-01-01T08:30:00.000000Z"
    assert fact.recorded_at == "2025-01-02T00:00:00.000000Z"
    with MemoryStore(tmp_path / "store") as reopened:
        assert reopened.get(fact.id).valid_from == "2025-01-01T08:30:00.000000Z"


def test_naive_datetimes_at_the_ends_of_the_calendar_are_accepted():
    """Naive datetimes at the first and last representable instants normalize without error."""
    assert normalize_time(datetime(1, 1, 1)) == "0001-01-01T00:00:00.000000Z"
    assert normalize_time(datetime(9999, 12, 31, 23, 59, 59, 999999)) == (
        "9999-12-31T23:59:59.999999Z")
    fact = Fact(id="a" * 32, entity="person", attribute="drink", value="tea",
                valid_from=datetime(1, 1, 1), recorded_at=datetime(1, 1, 1, 0, 0, 1))
    assert (fact.valid_from, fact.recorded_at) == (
        "0001-01-01T00:00:00.000000Z", "0001-01-01T00:00:01.000000Z")


@pytest.mark.parametrize("text", [
    "0001-01-01T00:00:00+01:00",
    "9999-12-31T23:59:59-01:00",
    "0001-01-01T00:30+00:31",
])
def test_out_of_range_error_names_the_rejected_input(text):
    """A time that falls outside the calendar once converted to UTC is quoted back in the error."""
    with pytest.raises(ValueError) as caught:
        normalize_time(text)
    assert text in str(caught.value)
    with pytest.raises(ValueError) as caught:
        Fact(id="a" * 32, entity="person", attribute="drink", value="tea",
             valid_from=text, recorded_at="2025-01-01")
    assert text in str(caught.value)


def test_cli_error_names_an_out_of_range_known_at(tmp_path, capsys):
    """With a valid --as-of, the CLI error quotes the --known-at that overflows the calendar."""
    bad = "9999-12-31T23:30-01:00"
    code = cli.main(["--root", str(tmp_path), "search", "tea",
                     "--as-of", "2025-01-01", "--known-at", bad])
    assert code == 1
    assert bad in json.loads(capsys.readouterr().err)["error"]
