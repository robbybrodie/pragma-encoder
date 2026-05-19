"""Pytest configuration and fixtures for PRAGMA OpenShift integration tests.

All tests in tests/openshift/ skip unless ``RUN_OPENSHIFT_TESTS=1`` is set.
``PRAGMA_TEST_NAMESPACE`` is required when the suite is enabled.

Safety model:
  Argo CD deploys and owns the long-lived PRAGMA platform substrate.
  These tests verify that substrate and create only short-lived labelled
  runtime resources. Cleanup deletes only resources carrying both:
    pragma.redhat.com/test-run=true
    pragma.redhat.com/test-id=<test_id>

No namespace creation, no secret deletion, no Argo CD resource mutation.
"""

from __future__ import annotations

import datetime
import os
import uuid

import pytest

from tests.openshift.oc import oc, label_selector

# ---------------------------------------------------------------------------
# Suite-wide skip guard
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip every test in tests/openshift/ unless RUN_OPENSHIFT_TESTS=1."""
    if os.environ.get("RUN_OPENSHIFT_TESTS") == "1":
        return  # suite is opted-in — run normally

    skip_marker = pytest.mark.skip(
        reason=(
            "OpenShift integration tests are opt-in. "
            "Set RUN_OPENSHIFT_TESTS=1 and PRAGMA_TEST_NAMESPACE=<namespace> to enable."
        )
    )
    for item in items:
        # Only apply to tests inside tests/openshift/
        if "openshift" in str(item.fspath):
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_namespace() -> str:
    """Return the Argo CD-managed PRAGMA namespace from PRAGMA_TEST_NAMESPACE.

    Skips the test (with a clear message) if the env var is not set.
    Tests may read from this namespace and verify substrate resources,
    but must not create, patch, or delete namespace-level resources here.

    This is distinct from ``runtime_namespace`` where ephemeral test
    resources may be created.
    """
    ns = os.environ.get("PRAGMA_TEST_NAMESPACE", "").strip()
    if not ns:
        pytest.skip(
            "PRAGMA_TEST_NAMESPACE is not set. "
            "Export it to the namespace where the PRAGMA platform substrate is deployed, "
            "e.g.: export PRAGMA_TEST_NAMESPACE=pragma-encoder"
        )
    return ns


@pytest.fixture(scope="session")
def runtime_namespace(test_namespace: str) -> str:
    """Return the namespace where ephemeral test resources may be created.

    Reads PRAGMA_TEST_RUNTIME_NAMESPACE.
    Falls back to ``test_namespace`` (PRAGMA_TEST_NAMESPACE) if unset.

    Use this namespace for PipelineRun, PyTorchJob, and other short-lived
    test resources. Never use it for Argo-managed substrate resources.
    """
    return os.environ.get("PRAGMA_TEST_RUNTIME_NAMESPACE", "").strip() or test_namespace


@pytest.fixture(scope="function")
def test_id() -> str:
    """Return a unique identifier for this test run.

    Format: ``pragma-it-<YYYYMMDD-HHMMSS>-<shortuuid>``

    Used as the value of the ``pragma.redhat.com/test-id`` label on every
    resource created by the test. Cleanup is scoped to this ID so that
    parallel runs cannot interfere with each other.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"pragma-it-{timestamp}-{short}"


@pytest.fixture(scope="function")
def test_labels(test_id: str) -> dict[str, str]:
    """Return the label dict that must be applied to every test-created resource.

    Both labels must be present:
        pragma.redhat.com/test-run=true
        pragma.redhat.com/test-id=<test_id>

    Cleanup is scoped to resources carrying both labels.
    """
    return {
        "pragma.redhat.com/test-run": "true",
        "pragma.redhat.com/test-id": test_id,
    }


@pytest.fixture(scope="session")
def timeout_seconds() -> int:
    """Return the timeout in seconds for cluster operations.

    Reads PRAGMA_TEST_TIMEOUT_SECONDS (default: 300).
    Used for PipelineRun and PyTorchJob wait loops.
    """
    raw = os.environ.get("PRAGMA_TEST_TIMEOUT_SECONDS", "300").strip()
    try:
        return max(30, int(raw))
    except ValueError:
        return 300


@pytest.fixture(scope="function")
def cleanup_labelled_resources(
    test_id: str,
    runtime_namespace: str,
) -> "Generator[None, None, None]":
    """Yield, then clean up all labelled resources created during the test.

    After each test, deletes only resources in the safe cleanup list that
    carry both test labels, scoped to ``runtime_namespace``.

    Safe resource types (never includes secrets or serviceaccounts):
        pipelinerun, taskrun, pod, job, configmap, pytorchjob

    Cleanup failures are reported via warnings but do not hide the original
    test result.

    Yields:
        None — the test body runs between setup and teardown.
    """
    from typing import Generator  # local import avoids top-level cycle
    yield  # --- test body runs here ---

    selector = label_selector(test_id)

    # Resource types that are safe to cleanup when label-scoped.
    # PVCs excluded by default — they may hold valuable data and require
    # explicit opt-in.  Secrets and ServiceAccounts never included.
    _CLEANUP_KINDS = [
        "pipelinerun",
        "taskrun",
        "pod",
        "job",
        "configmap",
        "pytorchjob",
    ]

    for kind in _CLEANUP_KINDS:
        try:
            result = oc(
                ["delete", kind, "-l", selector, "--ignore-not-found"],
                namespace=runtime_namespace,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                # Report but do not re-raise — cleanup failure must not hide test result.
                import warnings
                warnings.warn(
                    f"Cleanup of {kind} with selector {selector!r} "
                    f"in namespace {runtime_namespace!r} failed "
                    f"(exit {result.returncode}): {result.stderr.strip()[:200]}",
                    stacklevel=2,
                )
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"Cleanup of {kind} raised: {exc}",
                stacklevel=2,
            )
