"""Shared test configuration.

``HYPOTHESIS_PROFILE=mutation`` runs fewer, fixed property examples so that a
mutation run (which repeats the suite once per mutant) stays fast and
deterministic. The default profile is the thorough one used in CI.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "thorough", max_examples=200, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.register_profile(
    "mutation", max_examples=40, deadline=None, derandomize=True, database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "thorough"))
