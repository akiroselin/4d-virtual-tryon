"""Skip tests whose local-only assets are absent from a clean clone.

Body meshes, mocap clips, prior run outputs and the CortexODE checkout are
gitignored (licensing and size), so a fresh checkout cannot run the tests that
read them. Those tests are reported as skipped with the missing path instead of
failing; every other FileNotFoundError still fails normally.
"""

from __future__ import annotations

import pytest

LOCAL_ONLY_MARKERS = (
    "assets/body_meshes/",
    "assets/mocap/",
    "outputs/",
    "work/",
    "CortexODE tca.py",
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    excinfo = outcome.excinfo
    if excinfo is None or not issubclass(excinfo[0], FileNotFoundError):
        return
    message = str(excinfo[1])
    if any(marker in message for marker in LOCAL_ONLY_MARKERS):
        outcome.force_exception(pytest.skip.Exception(f"local-only asset missing: {message}", _use_item_location=True))
