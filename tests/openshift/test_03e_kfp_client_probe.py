"""Level 3e — KFP/DSPA client connectivity probe.

Purpose:
  Prove the exact kfp.Client configuration required to reach the OpenShift AI
  DSPA / KFP v2 API from inside the PRAGMA workbench pod. This is a prerequisite
  for Level 3 (pipeline upload + run creation).

  These tests are a bridge between discovery (Level 3d) and full pipeline
  runtime submission (Level 3).

In-cluster vs external:
  This probe targets the IN-CLUSTER HTTP endpoint (port 8888), not the external
  OAuth-protected HTTPS route. Port 8888 is the direct API server port, accessible
  from within the same namespace without OAuth.

    In-cluster (no OAuth):
      http://ds-pipeline-pipelines-definition.<namespace>.svc.cluster.local:8888
      http://ml-pipeline.<namespace>.svc.cluster.local:8888

    External HTTPS (OAuth token required — tested separately if needed):
      https://ds-pipeline-pipelines-definition-<namespace>.apps.<cluster-domain>

  These tests will SKIP (not fail) if run from outside the cluster where the
  in-cluster endpoint is unreachable. Run them from inside the workbench pod.

Authentication model:
  DSPA with enableOauth=true uses an OAuth proxy at port 8443 (HTTPS).
  Port 8888 (HTTP) is the direct API server — within the cluster namespace this
  port may be reachable without a token. These tests probe both cases:
    1. No token (port 8888 direct).
    2. With SA token from /var/run/secrets/kubernetes.io/serviceaccount/token,
       if the first attempt requires auth.

Opt-in gate:
  Tests require BOTH:
    RUN_OPENSHIFT_TESTS=1    — enables the openshift suite
    RUN_DSPA_CLIENT_PROBE=1  — enables this probe specifically
  If RUN_DSPA_CLIENT_PROBE=1 is set but kfp is not installed, prereq tests FAIL
  (not skip) so the missing dependency is clearly visible.

Safety:
  - Read-only: no pipeline uploads, no run creation.
  - No oc exec.
  - No Tekton.
  - SA token never printed.

Run from inside workbench pod:
  export RUN_OPENSHIFT_TESTS=1
  export RUN_DSPA_CLIENT_PROBE=1
  export PRAGMA_TEST_NAMESPACE=pragma-encoder
  cd /opt/app-root/src
  PYTHONPATH=. python -m pytest tests/openshift/test_03e_kfp_client_probe.py -v -s

Prerequisites:
  - kfp >= 2.0.0 installed
  - DSPA pods Running/Ready (verified by Level 3d)
  - In-cluster network access to ds-pipeline-* service
  - Typically requires running inside the workbench pod
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import socket
import urllib.error
import urllib.request

import pytest

# ---------------------------------------------------------------------------
# Opt-in gate — this probe requires both flags
# ---------------------------------------------------------------------------

_PROBE_ENABLED = os.environ.get("RUN_DSPA_CLIENT_PROBE") == "1"

_skip_probe_not_enabled = pytest.mark.skipif(
    not _PROBE_ENABLED,
    reason=(
        "KFP/DSPA client probe is opt-in. "
        "Set RUN_DSPA_CLIENT_PROBE=1 to enable. "
        "Run from inside the PRAGMA workbench pod for in-cluster endpoint access."
    ),
)

# ---------------------------------------------------------------------------
# KFP availability sentinel
# ---------------------------------------------------------------------------

_KFP_AVAILABLE = importlib.util.find_spec("kfp") is not None

# ---------------------------------------------------------------------------
# In-cluster endpoint candidates (discovered in Level 3d).
# Port 8888 is the direct HTTP API server port (no OAuth proxy).
# ---------------------------------------------------------------------------

_KFP_API_PORT = 8888
_KFP_API_BASE_PATH = "/apis/v2beta1"

_SA_TOKEN_PATH = pathlib.Path(
    "/var/run/secrets/kubernetes.io/serviceaccount/token"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_endpoint(namespace: str, service_name: str = "ds-pipeline-pipelines-definition") -> str:
    """Return the in-cluster HTTP endpoint for the KFP API service."""
    return f"http://{service_name}.{namespace}.svc.cluster.local:{_KFP_API_PORT}"


def _probe_http(url: str, token: str | None = None, timeout: int = 5) -> int:
    """Attempt a GET request and return the HTTP status code.

    Returns the HTTP status code.
    Raises ``_EndpointUnreachableError`` if the TCP connection fails (not in cluster).
    Does not raise on HTTP error codes — callers inspect the code.

    Never prints the token value.
    """
    req = urllib.request.Request(url, method="GET")
    if token is not None:
        req.add_header("Authorization", "Bearer <token>")  # log-safe placeholder
        req.headers["Authorization"] = f"Bearer {token}"  # actual header (no print)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (ConnectionRefusedError, OSError, socket.timeout, urllib.error.URLError) as exc:
        raise _EndpointUnreachableError(
            f"Cannot reach {url}: {type(exc).__name__}: {exc}"
        ) from exc


class _EndpointUnreachableError(Exception):
    """Raised when the in-cluster endpoint is not reachable from this environment."""


def _read_sa_token() -> str | None:
    """Return the mounted SA token text, or None if not inside a pod."""
    if _SA_TOKEN_PATH.exists():
        try:
            return _SA_TOKEN_PATH.read_text().strip()
        except OSError:
            return None
    return None


def _kfp_client(endpoint: str, token: str | None = None):  # type: ignore[return]
    """Return a kfp.Client instance for the given endpoint.

    Args:
        endpoint: Full URL, e.g. ``http://ds-pipeline-pipelines-definition:8888``.
        token:    Optional bearer token string. Not printed.

    Returns:
        ``kfp.Client`` instance.
    """
    import kfp  # noqa: PLC0415

    kwargs: dict = {"host": endpoint}
    if token is not None:
        kwargs["existing_token"] = token
    return kfp.Client(**kwargs)


# ---------------------------------------------------------------------------
# Prerequisite checks (local, no connection attempted)
# ---------------------------------------------------------------------------


class TestKFPClientProbePrereqs:
    """Level 3e: Prerequisite checks before attempting KFP client connection.

    These tests run locally (no connection to the cluster).
    When RUN_DSPA_CLIENT_PROBE=1:
      - kfp missing → FAIL (not skip).
      - endpoint string construction → checked.
    """

    @_skip_probe_not_enabled
    def test_kfp_installed_for_probe(self) -> None:
        """kfp SDK must be installed when the client probe is enabled.

        Unlike the optional SDK check in Level 3d, this test FAILS (not skips)
        when the probe is requested but kfp is absent, because the entire purpose
        of this probe file is to test kfp.Client connectivity.

        Install: pip install kfp  (or use the PRAGMA workbench image)
        """
        if not _KFP_AVAILABLE:
            pytest.fail(
                "kfp is not installed but RUN_DSPA_CLIENT_PROBE=1 is set. "
                "Install kfp to run the client probe: pip install kfp\n"
                "Alternatively, run this probe from inside the PRAGMA workbench "
                "notebook image where kfp is pre-installed."
            )

        import kfp  # noqa: PLC0415
        version = kfp.__version__
        major = int(version.split(".")[0])

        assert major >= 2, (  # noqa: PLR2004
            f"kfp version {version!r} is too old. "
            "DSPA / KFP v2 requires kfp SDK >= 2.0.0. "
            "Upgrade: pip install 'kfp>=2'"
        )

        print(f"\nkfp SDK version: {version} — OK")

    @_skip_probe_not_enabled
    def test_primary_endpoint_string_valid(self, test_namespace: str) -> None:
        """The primary in-cluster endpoint string must be well-formed.

        Verifies that the candidate endpoint URL can be constructed from the
        namespace without errors. Does not attempt a connection.
        """
        endpoint = _build_endpoint(test_namespace)
        assert endpoint.startswith("http://"), (
            f"Endpoint should be http://: {endpoint!r}"
        )
        assert f":{_KFP_API_PORT}" in endpoint, (
            f"Endpoint should include port {_KFP_API_PORT}: {endpoint!r}"
        )
        assert test_namespace in endpoint, (
            f"Endpoint should reference namespace {test_namespace!r}: {endpoint!r}"
        )
        print(f"\nPrimary KFP API endpoint: {endpoint}")

    @_skip_probe_not_enabled
    def test_sa_token_environment_documented(self) -> None:
        """Report whether the SA token file is present (inside pod indicator).

        The SA token is available when running inside a Kubernetes pod.
        Its presence does not guarantee the token has DSPA API access rights.
        This test never reads or prints the token content.
        """
        is_inside_pod = _SA_TOKEN_PATH.exists()
        if is_inside_pod:
            print(f"\nSA token file: {_SA_TOKEN_PATH} — present (running inside pod)")
        else:
            print(
                f"\nSA token file: {_SA_TOKEN_PATH} — absent (running outside cluster). "
                "Token-based auth will not be attempted. "
                "Run this probe from inside the PRAGMA workbench pod."
            )


# ---------------------------------------------------------------------------
# Connectivity probe — attempts actual TCP connection, skips if outside cluster
# ---------------------------------------------------------------------------


class TestKFPEndpointReachability:
    """Level 3e: TCP connectivity to the in-cluster KFP API endpoint.

    These tests require being inside the cluster. They SKIP (not fail) when
    the endpoint is not reachable, producing a clear message about environment.
    """

    @_skip_probe_not_enabled
    def test_in_cluster_endpoint_reachable_without_token(
        self, test_namespace: str
    ) -> None:
        """Probe the in-cluster KFP HTTP endpoint (port 8888) without a token.

        Port 8888 is the direct HTTP port on the ds-pipeline API server.
        Depending on the DSPA configuration:
          - If no auth is required in-cluster: HTTP 200 or HTTP 2xx expected.
          - If auth is required (OAuth enabled): HTTP 401 or HTTP 403 expected.
          - If endpoint is unreachable: test SKIPS (not in cluster).

        This test does NOT fail on 401/403 — it only confirms that the endpoint
        is reachable. Auth handling is in TestKFPClientAuth.
        """
        endpoint = _build_endpoint(test_namespace)
        health_url = f"{endpoint}{_KFP_API_BASE_PATH}/healthz"

        try:
            status = _probe_http(health_url, token=None, timeout=5)
        except _EndpointUnreachableError as exc:
            pytest.skip(
                f"In-cluster endpoint not reachable from this environment.\n"
                f"Endpoint: {health_url}\n"
                f"Error: {exc}\n"
                "Run this probe from inside the PRAGMA workbench pod:\n"
                "  oc exec -n pragma-encoder <workbench-pod> -- \\\n"
                "    env RUN_OPENSHIFT_TESTS=1 RUN_DSPA_CLIENT_PROBE=1 \\\n"
                f"    PRAGMA_TEST_NAMESPACE={test_namespace} \\\n"
                "    python -m pytest tests/openshift/test_03e_kfp_client_probe.py -v"
            )

        print(
            f"\nEndpoint: {health_url}\n"
            f"HTTP status (no token): {status}"
        )

        assert status in (200, 201, 401, 403), (
            f"Unexpected HTTP status {status} from {health_url!r}. "
            "Expected 200 (OK), 401 (auth required), or 403 (forbidden). "
            "A 5xx or unusual status suggests a DSPA configuration problem."
        )

        if status in (401, 403):
            print(
                f"  → Auth required (HTTP {status}). "
                "Token-based auth will be attempted in TestKFPClientAuth."
            )
        else:
            print(f"  → Endpoint accessible without token (HTTP {status}).")

    @_skip_probe_not_enabled
    def test_ml_pipeline_alias_reachable(
        self, test_namespace: str
    ) -> None:
        """Probe the ml-pipeline service alias (port 8888) without a token.

        ml-pipeline is an alias service for the DSPA KFP API server.
        If the primary ds-pipeline-* service is unreachable, the alias may differ.
        Both should expose the same API.
        """
        endpoint = _build_endpoint(test_namespace, service_name="ml-pipeline")
        health_url = f"{endpoint}{_KFP_API_BASE_PATH}/healthz"

        try:
            status = _probe_http(health_url, token=None, timeout=5)
        except _EndpointUnreachableError as exc:
            pytest.skip(
                f"ml-pipeline alias not reachable: {exc}. "
                "This is expected when running outside the cluster. "
                "The primary ds-pipeline-* service is the recommended endpoint."
            )

        print(
            f"\nml-pipeline alias: {health_url}\n"
            f"HTTP status (no token): {status}"
        )

        # ml-pipeline alias should behave identically to the primary service.
        assert status in (200, 201, 401, 403), (
            f"Unexpected HTTP status {status} from ml-pipeline alias. "
            "Expected same behaviour as primary ds-pipeline-* service."
        )


# ---------------------------------------------------------------------------
# KFP Client auth probe — instantiates kfp.Client with/without token
# ---------------------------------------------------------------------------


class TestKFPClientAuth:
    """Level 3e: kfp.Client instantiation and auth probe.

    Tries to instantiate kfp.Client against the in-cluster endpoint.
    Handles the auth flow:
      1. Try without token (may work if port 8888 is unprotected in-cluster).
      2. If 401/403, retry with SA token if inside a pod.
    """

    @_skip_probe_not_enabled
    def test_kfp_client_instantiates(self, test_namespace: str) -> None:
        """kfp.Client must instantiate without raising for the in-cluster endpoint.

        kfp.Client() does not connect at construction time in kfp v2 — it only
        stores configuration. This test confirms the Client object is constructable
        with the DSPA endpoint URL.
        """
        import kfp  # noqa: PLC0415

        endpoint = _build_endpoint(test_namespace)
        client = kfp.Client(host=endpoint)

        assert client is not None, "kfp.Client() returned None"
        print(f"\nkfp.Client instantiated for: {endpoint}")

    @_skip_probe_not_enabled
    def test_kfp_client_list_pipelines_no_token(
        self, test_namespace: str
    ) -> None:
        """Attempt to list pipelines without a token (port 8888 direct access).

        If port 8888 does not require authentication in-cluster, this call
        returns a pipeline list (possibly empty). If auth is required,
        the call raises an exception — this test marks the result and skips
        rather than failing, since the token-based test handles the auth path.

        Result is reported in the test output for diagnostic purposes.
        """
        import kfp  # noqa: PLC0415
        import kfp.exceptions  # noqa: PLC0415

        endpoint = _build_endpoint(test_namespace)

        # First confirm endpoint is reachable (skip if outside cluster).
        health_url = f"{endpoint}{_KFP_API_BASE_PATH}/healthz"
        try:
            _probe_http(health_url, token=None, timeout=5)
        except _EndpointUnreachableError as exc:
            pytest.skip(f"Endpoint not reachable: {exc}")

        client = kfp.Client(host=endpoint)

        try:
            result = client.list_pipelines()
            pipeline_count = len(result.pipelines) if result.pipelines else 0
            print(
                f"\nlist_pipelines() SUCCESS (no token)\n"
                f"Endpoint: {endpoint}\n"
                f"Pipelines found: {pipeline_count}\n"
                "→ Port 8888 does NOT require authentication in-cluster."
            )
        except Exception as exc:  # noqa: BLE001
            exc_type = type(exc).__name__
            exc_msg = str(exc)[:200]
            print(
                f"\nlist_pipelines() FAILED without token\n"
                f"Endpoint: {endpoint}\n"
                f"Error: {exc_type}: {exc_msg}\n"
                "→ Port 8888 requires authentication. "
                "See test_kfp_client_list_pipelines_with_sa_token."
            )
            pytest.skip(
                f"list_pipelines() failed without token ({exc_type}). "
                "This is expected when DSPA requires auth even on port 8888. "
                "The SA token test (test_kfp_client_list_pipelines_with_sa_token) "
                "will attempt the same call with the mounted SA token."
            )

    @_skip_probe_not_enabled
    def test_kfp_client_list_pipelines_with_sa_token(
        self, test_namespace: str
    ) -> None:
        """Attempt to list pipelines using the mounted SA token.

        The SA token is mounted at:
          /var/run/secrets/kubernetes.io/serviceaccount/token

        This file is only present when running inside a Kubernetes pod.
        If not inside a pod, this test skips with a clear message.

        The SA token is never printed.
        """
        import kfp  # noqa: PLC0415

        endpoint = _build_endpoint(test_namespace)

        # First confirm endpoint is reachable (skip if outside cluster).
        health_url = f"{endpoint}{_KFP_API_BASE_PATH}/healthz"
        try:
            _probe_http(health_url, token=None, timeout=5)
        except _EndpointUnreachableError as exc:
            pytest.skip(f"Endpoint not reachable: {exc}")

        # Require SA token file to be present.
        token = _read_sa_token()
        if token is None:
            pytest.skip(
                "SA token file not present at "
                f"{_SA_TOKEN_PATH}. "
                "This test must run inside a Kubernetes pod. "
                "Run from inside the PRAGMA workbench pod:\n"
                "  oc exec -n pragma-encoder <workbench-pod> -- \\\n"
                "    env RUN_OPENSHIFT_TESTS=1 RUN_DSPA_CLIENT_PROBE=1 "
                f"PRAGMA_TEST_NAMESPACE={test_namespace} \\\n"
                "    python -m pytest tests/openshift/test_03e_kfp_client_probe.py -v"
            )

        print(
            f"\nSA token: present ({len(token)} chars, not printed)\n"
            f"Endpoint: {endpoint}"
        )

        client = kfp.Client(host=endpoint, existing_token=token)

        result = client.list_pipelines()
        pipeline_count = len(result.pipelines) if result.pipelines else 0

        print(
            f"list_pipelines() SUCCESS (with SA token)\n"
            f"Pipelines found: {pipeline_count}\n"
            "→ SA token grants access to DSPA KFP v2 API.\n"
            "→ Confirmed client configuration for pipeline upload + run creation:\n"
            f"   kfp.Client(host='{endpoint}', existing_token=<sa-token>)"
        )


# ---------------------------------------------------------------------------
# Read-only API probe — deeper read-only calls to confirm working client
# ---------------------------------------------------------------------------


class TestKFPReadOnlyAPI:
    """Level 3e: Read-only KFP v2 API calls to confirm working client.

    These tests assume kfp.Client is working (either with or without token).
    They call read-only endpoints that will be used by the pipeline upload
    path (list existing pipelines, verify API version).
    """

    @_skip_probe_not_enabled
    def test_kfp_api_version_endpoint(self, test_namespace: str) -> None:
        """The /apis/v2beta1/healthz endpoint must return 2xx or auth error.

        This confirms the API version path (/v2beta1) is correct for this
        DSPA instance.
        """
        endpoint = _build_endpoint(test_namespace)
        health_url = f"{endpoint}{_KFP_API_BASE_PATH}/healthz"

        try:
            status = _probe_http(health_url, token=None, timeout=5)
        except _EndpointUnreachableError as exc:
            pytest.skip(f"Endpoint not reachable: {exc}")

        print(
            f"\nKFP API health probe:\n"
            f"  URL:    {health_url}\n"
            f"  Status: {status}"
        )

        # 200 = healthy, 401/403 = auth needed but API exists
        # Anything else suggests wrong port or path
        assert status in (200, 401, 403, 404), (  # 404 if /healthz not implemented
            f"Unexpected HTTP status {status} from {health_url!r}. "
            "Expected 200/401/403 from a KFP v2 server, or 404 if /healthz "
            "is not implemented in this DSPA version. "
            "A 5xx or connection error suggests the wrong endpoint."
        )

        endpoint_verdict = {
            200: "API is healthy and accessible without authentication.",
            401: "API is reachable; authentication (Bearer token) required.",
            403: "API is reachable; authorization check failed (token may be wrong).",
            404: "API is reachable; /healthz not implemented (try /apis/v2beta1/pipelines).",
        }.get(status, f"Unexpected status {status}")

        print(f"  Verdict: {endpoint_verdict}")

    @_skip_probe_not_enabled
    def test_kfp_client_configuration_report(
        self, test_namespace: str
    ) -> None:
        """Report the complete discovered KFP client configuration for the PR.

        This test always passes (it's a report, not an assertion). It prints
        all the information needed to implement pipeline upload in src/workbench/.
        """
        endpoint_primary = _build_endpoint(test_namespace)
        endpoint_alias = _build_endpoint(test_namespace, service_name="ml-pipeline")

        is_inside_pod = _SA_TOKEN_PATH.exists()
        kfp_version = "not installed"
        if _KFP_AVAILABLE:
            import kfp  # noqa: PLC0415
            kfp_version = kfp.__version__

        # Check if primary endpoint is reachable.
        try:
            status_no_token = _probe_http(
                f"{endpoint_primary}{_KFP_API_BASE_PATH}/healthz",
                token=None,
                timeout=5,
            )
            endpoint_reachable = True
        except _EndpointUnreachableError:
            status_no_token = None
            endpoint_reachable = False

        report = [
            "",
            "=== KFP/DSPA Client Configuration Report ===",
            f"Namespace:            {test_namespace}",
            f"Primary endpoint:     {endpoint_primary}",
            f"Alias endpoint:       {endpoint_alias}",
            f"API base path:        {_KFP_API_BASE_PATH}",
            f"kfp SDK version:      {kfp_version}",
            f"Running inside pod:   {is_inside_pod}",
            f"SA token available:   {is_inside_pod}",
            f"Endpoint reachable:   {endpoint_reachable}",
        ]
        if endpoint_reachable:
            report.append(f"HTTP status (no tok):  {status_no_token}")
        report.extend([
            "",
            "Recommended client configuration (inside workbench pod):",
            "  # No token needed (if port 8888 allows in-cluster access):",
            f"  client = kfp.Client(host='{endpoint_primary}')",
            "  # With SA token (if port 8888 requires auth):",
            f"  token = pathlib.Path('{_SA_TOKEN_PATH}').read_text().strip()",
            f"  client = kfp.Client(host='{endpoint_primary}', existing_token=token)",
            "",
            "Next steps for pipeline upload + run creation:",
            "  1. Create pipeline: client.upload_pipeline(yaml_path, pipeline_name)",
            "  2. Create run:      client.create_run_from_pipeline_func(..., experiment_name=...)",
            "==============================================",
        ])

        print("\n".join(report))
