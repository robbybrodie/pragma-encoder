"""Level 0 — oc access and namespace reachability.

Verifies that:
  - the oc binary is available on PATH
  - the current session is authenticated (oc whoami succeeds)
  - PRAGMA_TEST_NAMESPACE exists and is accessible
  - the test runner can list pods in the test namespace (basic RBAC)
  - PRAGMA_TEST_RUNTIME_NAMESPACE exists (when different from test_namespace)

All tests are read-only. No resources are created or modified.

Prerequisites:
  - RUN_OPENSHIFT_TESTS=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - oc logged in (oc whoami must succeed)
"""

from __future__ import annotations

import pytest

from tests.openshift.oc import oc, resource_exists


class TestOcAccess:
    """Level 0: verify oc binary and authentication."""

    def test_oc_binary_available(self) -> None:
        """oc binary must be present on PATH.

        If oc is missing, all subsequent tests would fail with confusing errors.
        This test surfaces the problem immediately with a clear message.
        """
        import shutil
        assert shutil.which("oc") is not None, (
            "oc binary not found on PATH. "
            "Install the OpenShift CLI before running integration tests. "
            "https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html"
        )

    def test_oc_whoami(self) -> None:
        """oc whoami must succeed — session must be authenticated.

        A failed whoami means the oc session has expired or was never established.
        Run ``oc login <cluster-url>`` before running integration tests.
        """
        result = oc(["whoami"], check=True, timeout=15)
        username = result.stdout.strip()
        assert username, (
            "oc whoami returned empty output. "
            "Ensure you are logged in: oc login <cluster-url>"
        )
        # Do not print the username — it may be a service account token path.
        # Just assert it is non-empty.


class TestNamespaceAccess:
    """Level 0: verify namespace existence and read access."""

    def test_test_namespace_exists(self, test_namespace: str) -> None:
        """PRAGMA_TEST_NAMESPACE must exist on the cluster.

        The namespace is expected to be Argo CD-managed. These tests do not
        create or manage the namespace — they only verify it is present.
        """
        assert resource_exists("namespace", test_namespace, namespace=None), (
            f"Namespace {test_namespace!r} does not exist on the cluster. "
            "Verify PRAGMA_TEST_NAMESPACE is set to the correct namespace "
            "and that Argo CD has deployed the PRAGMA platform substrate."
        )

    def test_can_list_pods_in_test_namespace(self, test_namespace: str) -> None:
        """The test runner must have permission to list pods in PRAGMA_TEST_NAMESPACE.

        A successful pod list (even an empty one) confirms that:
        1. The namespace is reachable.
        2. The service account / user running tests has at least list-pods RBAC.

        This does not verify any specific pod is running — only that the call succeeds.
        """
        result = oc(
            ["get", "pods", "--no-headers"],
            namespace=test_namespace,
            check=True,
            timeout=20,
        )
        # Acceptable outputs: a list of pods, or "No resources found in <ns> namespace."
        # Either is fine — we only require exit 0.
        assert result.returncode == 0, (
            f"Could not list pods in namespace {test_namespace!r}. "
            "Check RBAC: the test runner needs 'get pods' permission."
        )

    def test_runtime_namespace_exists_if_different(
        self,
        test_namespace: str,
        runtime_namespace: str,
    ) -> None:
        """If PRAGMA_TEST_RUNTIME_NAMESPACE is set to a different namespace, it must exist.

        When the runtime namespace is the same as the test namespace (default),
        this test is a no-op pass — it was already verified by
        ``test_test_namespace_exists``.
        """
        if runtime_namespace == test_namespace:
            pytest.skip(
                "PRAGMA_TEST_RUNTIME_NAMESPACE is not set or matches PRAGMA_TEST_NAMESPACE. "
                "Skipping separate runtime namespace check."
            )

        assert resource_exists("namespace", runtime_namespace, namespace=None), (
            f"Runtime namespace {runtime_namespace!r} does not exist on the cluster. "
            "Verify PRAGMA_TEST_RUNTIME_NAMESPACE is set correctly."
        )
