"""Shared test configuration.

Hypothesis profiles, chosen with the ``HYPOTHESIS_PROFILE`` environment variable:

* ``thorough`` (the default, also used in CI): 200 random examples per property.
  It is registered on top of Hypothesis's built-in default profile, so it stays
  random on CI services too. Hypothesis would otherwise load its ``ci`` profile
  there and replay the same fixed examples on every run. Failing examples are
  kept in Hypothesis's example database (``.hypothesis/``, or the directory
  named by ``HYPOTHESIS_STORAGE_DIRECTORY``) and replayed first next time, and
  a reproduction blob is printed with every failure.
* ``mutation``: 40 fixed examples, no example database and no shrinking, so
  that a mutation run (which repeats the suite once per mutant) stays fast and
  deterministic. A mutant only has to make a test fail; a minimal failing
  example would add minutes per mutant and tell the run nothing more.

Every test also runs isolated from the developer's environment: ``MEMORYOS_*``
variables are cleared, and Git reads no global or system configuration, no
global ignore or attribute files and no repository location or identity
variables. Git also stops looking for a repository at pytest's base temp
folder. A global ``commit.gpgsign``, hook directory, exported ``GIT_DIR`` or
a TMPDIR inside a Git checkout therefore cannot change a test's outcome.
Tests that need a Git identity set one themselves.

During a mutation run, each test session also deletes its own base temp folder
when it ends. mutmut 3.8 ends every worker process with ``os._exit``, so
pytest's exit-time cleanup never runs: each mutant would leave a locked
``pytest-N`` folder behind, which later pytest runs leave alone for three
days, and the folders left by runs of about 4,000 mutants filled the disk.
mutmut sets ``MUTANT_UNDER_TEST`` for every test run it starts (to an empty
value for its clean run), so a session counts as a mutation run when that
variable is set. A worker that mutmut kills, for example at its time limit,
still leaves its folder. Other runs keep pytest's default of the three
newest folders, and a folder given with ``--basetemp`` is left to pytest.
"""

import itertools
import os
import shutil
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, Phase, settings

import temporal_memory_os.extraction as extraction_module

_SUPPRESSED = [HealthCheck.too_slow, HealthCheck.filter_too_much]
_DEFAULT = settings.get_profile("default")

settings.register_profile(
    "thorough", parent=_DEFAULT, max_examples=200, deadline=None, derandomize=False,
    print_blob=True, suppress_health_check=_SUPPRESSED,
)
settings.register_profile(
    "mutation", parent=_DEFAULT, max_examples=40, deadline=None, derandomize=True,
    database=None, phases=(Phase.explicit, Phase.reuse, Phase.generate),
    suppress_health_check=_SUPPRESSED,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "thorough"))

# Git's own executable lookup is the only GIT_* variable a test run may inherit.
_KEPT_GIT_VARIABLES = frozenset({"GIT_EXEC_PATH"})


@pytest.fixture
def ticking_clock(monkeypatch):
    """Extraction "now" advances one second per reading, so undated lines extracted
    one after another get distinct, increasing observation times."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ticks = itertools.count(1)
    monkeypatch.setattr(extraction_module, "_now", lambda: (
        start + timedelta(seconds=next(ticks))).isoformat(timespec="microseconds"))


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path_factory):
    """Clear user configuration that would change what the code under test does."""
    for name in list(os.environ):
        if name.startswith("MEMORYOS_"):
            monkeypatch.delenv(name)
        elif name.startswith("GIT_") and name not in _KEPT_GIT_VARIABLES:
            monkeypatch.delenv(name)
    # Git falls back to EMAIL for an author address.
    monkeypatch.delenv("EMAIL", raising=False)
    home = tmp_path_factory.mktemp("git-home")
    global_config = home / "gitconfig"
    global_config.touch()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    # Git reads its global ignore and attribute files from here.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    # Git stops looking for a repository at pytest's base temp folder, so a
    # TMPDIR inside a Git checkout does not put every test inside that checkout.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path_factory.getbasetemp().resolve()))


@pytest.fixture(scope="session", autouse=True)
def mutation_run_temp_cleanup(request, tmp_path_factory):
    """Delete this session's base temp folder at the end of a session mutmut started."""
    mutation_run = "MUTANT_UNDER_TEST" in os.environ
    if not mutation_run or request.config.getoption("basetemp") is not None:
        yield
        return
    basetemp = tmp_path_factory.getbasetemp()
    yield
    shutil.rmtree(basetemp, ignore_errors=True)
