import pytest

from jomission.harness import seals
from jomission.harness.envfail import classify

seals.install()   # a bare `pytest tests/` must not be able to overwrite sealed evidence


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed:
        ids = classify(rep.longreprtext)
        if ids:
            rep.sections.append(("environment classification",
                                 f"ENVIRONMENT {ids}: see manifests/harness/environment_failures.json; not evidence about the model"))
