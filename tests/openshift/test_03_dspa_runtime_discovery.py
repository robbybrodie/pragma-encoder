"""Level 3d — DSPA / KFP v2 runtime discovery (read-only).

Purpose:
  Map the OpenShift AI Data Science Pipelines (DSPA) / KFP v2 substrate
  as it actually exists in PRAGMA_TEST_NAMESPACE. These tests are entirely
  read-only: no resources are created, patched, or deleted.

  This file is the precursor to Level 3 (test_03_pipeline_smoke_run.py),
  which will implement actual pipeline upload and run creation once the
  DSPA submission path is understood.

Pipeline runtime context:
  This environment uses OpenShift AI Data Science Pipelines (DSPA) / KFP v2.
  Red Hat OpenShift AI Data Science Pipelines 2.0 does NOT use kfp-tekton.
  The documented pattern is:
    1. Compile KFP v2 pipeline to YAML (local, no cluster needed).
    2. Upload/import compiled YAML to DSPA via the KFP v2 API.
    3. Create a pipeline run via the DSPA / KFP v2 API.
    4. Track the run via the DSPA API or OpenShift AI dashboard.
  This file maps steps 2–4 without implementing them yet.

Safety rules (all respected by every test in this file):
  - No resources created, patched, or deleted.
  - No oc exec.
  - No Tekton.
  - Secret data never printed.
  - Cluster access required for all tests except TestKFPSDK.

Tests:
  TestDSPAObjectDiscovery
    test_dspa_object_discovered
  TestDSPAPodsRunning
    test_dspa_pods_running
  TestDSPAServicesDiscovery
    test_dspa_services_discovered
  TestDSPARoutesDiscovery
    test_dspa_routes_discovered
  TestKFPSDK
    test_kfp_sdk_import_optional
    test_compiled_pipeline_yaml_exists_or_can_be_generated
  TestKFPEndpointDiscovery
    test_kfp_client_candidate_endpoint_documented
  TestDSPARuntimeFuture
    test_dspa_runtime_submission_marked_future

Prerequisites:
  - RUN_OPENSHIFT_TESTS=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - Level 0 and Level 1 tests passing (oc access, DSPA CRDs present)
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

import pytest

from tests.openshift.oc import oc_json

# ---------------------------------------------------------------------------
# KFP availability sentinel
# ---------------------------------------------------------------------------

_KFP_AVAILABLE = importlib.util.find_spec("kfp") is not None

_skip_no_kfp = pytest.mark.skipif(
    not _KFP_AVAILABLE,
    reason=(
        "kfp is not installed — skipping compile tests. "
        "Install with: pip install kfp, or "
        "use the PRAGMA workbench notebook image which includes kfp."
    ),
)

# ---------------------------------------------------------------------------
# KFP HTTP API port — the port used by ds-pipeline services for the REST API.
# DSPA exposes KFP v2 on port 8888 (plain http) inside the cluster.
# OAuth-protected access uses port 8443.
# ---------------------------------------------------------------------------
_KFP_HTTP_PORT = 8888

# ---------------------------------------------------------------------------
# Discovery report helper (Part C)
# ---------------------------------------------------------------------------


def _discovery_report(
    namespace: str,
    dspa_names: list[str],
    dspa_versions: list[str],
    pipeline_pods: list[str],
    candidate_services: list[str],
    candidate_routes: list[str],
    kfp_version: str | None,
    generated_yaml_path: str | None,
) -> str:
    """Return a concise DSPA discovery summary string.

    Used in assertion messages and printed when a test fails or when
    pytest -s is used. Does not print tokens or credentials.

    Args:
        namespace:           PRAGMA_TEST_NAMESPACE value.
        dspa_names:          Names of DataSciencePipelinesApplication objects found.
        dspa_versions:       DSP version strings for each DSPA object.
        pipeline_pods:       Names of ds-pipeline-* pods found.
        candidate_services:  Service names with KFP API port (8888).
        candidate_routes:    External route hostnames for ds-pipeline routes.
        kfp_version:         kfp SDK version string if installed, else None.
        generated_yaml_path: Path to compiled pipeline YAML if generated, else None.

    Returns:
        Multi-line discovery summary string.
    """
    lines = [
        "--- DSPA / KFP v2 Discovery Report ---",
        f"Namespace:         {namespace}",
        f"DSPA objects:      {dspa_names or ['(none found)']}",
        f"DSP versions:      {dspa_versions or ['(unknown)']}",
        f"Pipeline pods:     {len(pipeline_pods)} found",
    ]
    for pod in pipeline_pods:
        lines.append(f"  pod: {pod}")
    lines.append(
        f"KFP API services:  {candidate_services or ['(none with port 8888)']}"
    )
    lines.append(
        f"External routes:   {candidate_routes or ['(none found — in-cluster only)']}"
    )
    lines.append(
        f"kfp SDK:           {'v' + kfp_version if kfp_version else 'not installed'}"
    )
    lines.append(
        f"Compiled YAML:     {generated_yaml_path or '(not generated)'}"
    )
    lines.append("---------------------------------------")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dspa_objects(namespace: str) -> list[dict[str, Any]]:
    """Return DataSciencePipelinesApplication objects in the namespace."""
    result = oc_json(
        ["get", "datasciencepipelinesapplication"],
        namespace=namespace,
        timeout=20,
    )
    return result.get("items", [])


def _get_pods(namespace: str) -> list[dict[str, Any]]:
    """Return all pods in the namespace."""
    result = oc_json(["get", "pod"], namespace=namespace, timeout=20)
    return result.get("items", [])


def _get_services(namespace: str) -> list[dict[str, Any]]:
    """Return all services in the namespace."""
    result = oc_json(["get", "svc"], namespace=namespace, timeout=20)
    return result.get("items", [])


def _get_routes(namespace: str) -> list[dict[str, Any]]:
    """Return all routes in the namespace."""
    result = oc_json(["get", "route"], namespace=namespace, timeout=20)
    return result.get("items", [])


def _service_ports(svc: dict[str, Any]) -> list[int]:
    """Return all port numbers exposed by a service."""
    return [p.get("port", 0) for p in svc.get("spec", {}).get("ports", [])]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDSPAObjectDiscovery:
    """Level 3d: Discover DataSciencePipelinesApplication objects in namespace.

    These tests are read-only. No resources are created.
    """

    def test_dspa_object_discovered(self, test_namespace: str) -> None:
        """At least one DataSciencePipelinesApplication must exist in namespace.

        The DSPA object is deployed by Argo CD as part of the PRAGMA platform
        substrate. Its presence confirms that the DSPA operator has reconciled
        and the KFP v2 API server is configured.

        This test discovers the object(s) dynamically — it does not assume the
        object name. It reports names and dspVersion fields for diagnosis.

        If this fails: verify the RHOAI operator is installed and the
        DataSciencePipelines component is enabled in the DataScienceCluster CR.
        Then check that Argo CD has synced the PRAGMA substrate.
        """
        dspa_objects = _get_dspa_objects(test_namespace)

        dspa_names = [o["metadata"]["name"] for o in dspa_objects]
        dspa_versions = [
            o.get("spec", {}).get("dspVersion", "unknown") for o in dspa_objects
        ]

        report = _discovery_report(
            namespace=test_namespace,
            dspa_names=dspa_names,
            dspa_versions=dspa_versions,
            pipeline_pods=[],
            candidate_services=[],
            candidate_routes=[],
            kfp_version=None,
            generated_yaml_path=None,
        )

        assert len(dspa_objects) > 0, (
            f"No DataSciencePipelinesApplication found in namespace {test_namespace!r}. "
            "Argo CD may not have synced the PRAGMA substrate, or the RHOAI "
            "DataSciencePipelines component may not be enabled.\n" + report
        )

        # All discovered objects should report dspVersion=v2.
        for name, version in zip(dspa_names, dspa_versions):
            assert version == "v2", (
                f"DSPA object {name!r} has dspVersion={version!r}, expected 'v2'. "
                "PRAGMA targets DSPA v2 / KFP SDK v2.\n" + report
            )

        # Surface the report in pytest output when -s is used.
        print(f"\n{report}")


class TestDSPAPodsRunning:
    """Level 3d: Verify DSPA pipeline pods are Running and Ready."""

    def test_dspa_pods_running(self, test_namespace: str) -> None:
        """All ds-pipeline-* pods in namespace must be Running and Ready.

        DSPA deploys several pods to serve the KFP v2 API:
          - ds-pipeline-<name>         — main KFP API server
          - ds-pipeline-persistenceagent-<name>
          - ds-pipeline-scheduledworkflow-<name>
          - ds-pipeline-workflow-controller-<name>
          - ds-pipeline-metadata-grpc-<name>
          - ds-pipeline-metadata-envoy-<name>

        All must be Running/Ready for pipeline operations to succeed.

        If this fails: check 'oc get pods -n <namespace>' and describe
        any non-Ready pods. Argo CD or the RHOAI operator may be reconciling.
        """
        all_pods = _get_pods(test_namespace)

        # Filter to DSPA pipeline pods by name prefix.
        pipeline_pods = [
            p for p in all_pods
            if p["metadata"]["name"].startswith("ds-pipeline-")
        ]

        pod_names = [p["metadata"]["name"] for p in pipeline_pods]

        assert len(pipeline_pods) > 0, (
            f"No ds-pipeline-* pods found in namespace {test_namespace!r}. "
            "The DSPA instance may not have deployed yet, or the DSPA object "
            "does not have spec.apiServer.deploy=true. "
            f"All pods found: {[p['metadata']['name'] for p in all_pods]}"
        )

        not_ready = []
        for pod in pipeline_pods:
            name = pod["metadata"]["name"]
            phase = pod["status"].get("phase", "Unknown")
            container_statuses = pod["status"].get("containerStatuses", [])
            all_ready = all(cs.get("ready", False) for cs in container_statuses)
            if phase != "Running" or not all_ready:
                not_ready.append(f"{name} (phase={phase}, ready={all_ready})")

        report = _discovery_report(
            namespace=test_namespace,
            dspa_names=[],
            dspa_versions=[],
            pipeline_pods=pod_names,
            candidate_services=[],
            candidate_routes=[],
            kfp_version=None,
            generated_yaml_path=None,
        )

        assert not not_ready, (
            f"The following ds-pipeline-* pods are not Running/Ready in "
            f"namespace {test_namespace!r}:\n  "
            + "\n  ".join(not_ready)
            + "\nCheck pod events: oc describe pod <name> -n "
            + test_namespace
            + "\n" + report
        )

        print(f"\n{report}")


class TestDSPAServicesDiscovery:
    """Level 3d: Discover KFP API services (port 8888) in namespace."""

    def test_dspa_services_discovered(self, test_namespace: str) -> None:
        """At least one service exposing the KFP HTTP API port (8888) must exist.

        The DSPA operator creates services for the KFP v2 API server.
        The main KFP REST API is exposed on port 8888 (plain http in-cluster).
        OAuth-protected access uses port 8443.

        Known services on this cluster:
          - ds-pipeline-<dspa-name>: ports 8888 (http), 8887 (grpc), 8443 (oauth)
          - ml-pipeline: alias exposing the same ports

        The in-cluster KFP HTTP endpoint is:
          http://<service-name>.<namespace>.svc.cluster.local:8888

        This test does not attempt to connect to the endpoint.
        Use test_kfp_client_candidate_endpoint_documented for endpoint summary.
        """
        services = _get_services(test_namespace)

        kfp_api_services = [
            svc for svc in services
            if _KFP_HTTP_PORT in _service_ports(svc)
        ]

        svc_names = [svc["metadata"]["name"] for svc in kfp_api_services]

        assert len(kfp_api_services) > 0, (
            f"No services with port {_KFP_HTTP_PORT} (KFP HTTP API) found "
            f"in namespace {test_namespace!r}. "
            "Expected at least one ds-pipeline-* service. "
            "Verify the DSPA instance is healthy and spec.apiServer.deploy=true. "
            f"All services found: {[s['metadata']['name'] for s in services]}"
        )

        print(
            f"\nKFP API services (port {_KFP_HTTP_PORT}) in {test_namespace!r}: "
            f"{svc_names}"
        )


class TestDSPARoutesDiscovery:
    """Level 3d: Discover external DSPA routes in namespace.

    Routes provide external HTTPS access to the DSPA / KFP v2 API.
    External access requires OAuth token authentication.
    In-cluster access via service (port 8888) may bypass OAuth for
    same-namespace workloads, depending on DSPA configuration.
    """

    def test_dspa_routes_discovered(self, test_namespace: str) -> None:
        """Discover routes with 'ds-pipeline' prefix in namespace.

        If no routes are found, the test skips (not fails). In-cluster
        service access may still be available even without external routes.

        Routes found on this cluster follow the pattern:
          ds-pipeline-<dspa-name>-<namespace>.apps.<cluster-domain>

        The external HTTPS endpoint format (with OAuth token):
          https://<route-host>/apis/v2beta1/

        This test does not attempt connection or authentication.
        """
        routes = _get_routes(test_namespace)

        dspa_routes = [
            r for r in routes
            if r["metadata"]["name"].startswith("ds-pipeline")
        ]

        if not dspa_routes:
            pytest.skip(
                f"No ds-pipeline routes found in namespace {test_namespace!r}. "
                "In-cluster service access may still be available via port 8888. "
                "Set DSPA spec.apiServer to expose a route if external access is needed."
            )

        route_hosts = [
            r["spec"].get("host", "(no host)") for r in dspa_routes
        ]
        route_names = [r["metadata"]["name"] for r in dspa_routes]
        tls_routes = [
            r["metadata"]["name"]
            for r in dspa_routes
            if "tls" in r["spec"]
        ]

        print(
            f"\nDSPA routes in {test_namespace!r}: {route_names}\n"
            f"External hosts: {route_hosts}\n"
            f"TLS-secured:    {tls_routes}"
        )

        # All discovered ds-pipeline routes should have a host and be TLS.
        for route in dspa_routes:
            name = route["metadata"]["name"]
            host = route["spec"].get("host", "")
            assert host, (
                f"DSPA route {name!r} has no host. "
                "The route may still be provisioning."
            )


class TestKFPSDK:
    """Level 3d: KFP SDK availability and pipeline compile verification."""

    def test_kfp_sdk_import_optional(self) -> None:
        """Pass if the kfp SDK is importable; skip if not installed.

        The kfp SDK is required to compile KFP v2 pipelines to YAML and
        to interact with the DSPA / KFP v2 API programmatically.

        On this cluster: kfp is available in the PRAGMA workbench notebook
        image. It is NOT installed in the local development venv by default.

        To install locally:
          pip install kfp
        """
        if not _KFP_AVAILABLE:
            pytest.skip(
                "kfp is not installed in the current environment. "
                "Install with: pip install kfp\n"
                "The PRAGMA workbench notebook image includes kfp."
            )

        import kfp  # noqa: PLC0415
        version = kfp.__version__
        print(f"\nkfp SDK version: {version}")

        # Confirm the version is v2+ (DSPA requires KFP SDK v2).
        major = int(version.split(".")[0])
        assert major >= 2, (
            f"kfp version {version!r} is not v2+. "
            "DSPA / KFP v2 requires kfp SDK >= 2.0.0. "
            f"Upgrade: pip install 'kfp>=2'"
        )

    @_skip_no_kfp
    def test_compiled_pipeline_yaml_exists_or_can_be_generated(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """If kfp is installed, compile the decorated pipeline to KFP v2 YAML.

        This mirrors the Level 2 compile check but targets the discovery context:
        confirming that the PRAGMA pipeline can produce a valid KFP v2 YAML
        artifact that could be uploaded to DSPA.

        Does not upload or submit. Does not require cluster access.
        """
        import kfp  # noqa: PLC0415

        example_path = pathlib.Path("examples/workbench/05_decorated_pipeline.py")
        if not example_path.exists():
            pytest.skip(
                f"Decorated pipeline example {example_path} not found. "
                "This example is required to demonstrate the DSPA upload target."
            )

        output_yaml = tmp_path / "dspa_discovery_pipeline.yaml"

        # Import the pipeline module and call compile().
        import importlib.util as _ilu  # noqa: PLC0415
        spec = _ilu.spec_from_file_location("_dspa_discovery_example", example_path)
        assert spec is not None and spec.loader is not None, (
            f"Could not load module spec from {example_path}"
        )
        module = _ilu.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except SystemExit:
            pass  # some example files call sys.exit on completion

        # The example should expose a pipeline object via pragma_pipeline.
        # Find any object with a .compile() method (KFP pipeline wrapper).
        pipeline_obj = None
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name, None)
            if attr is not None and hasattr(attr, "compile"):
                pipeline_obj = attr
                break

        if pipeline_obj is None:
            pytest.skip(
                f"No compilable pipeline object found in {example_path}. "
                "The example may not expose a pipeline with a .compile() method."
            )

        pipeline_obj.compile(str(output_yaml))

        assert output_yaml.exists(), (
            f"KFP compile() ran but {output_yaml} was not created."
        )
        content = output_yaml.read_text()
        assert len(content) > 100, (  # noqa: PLR2004
            f"Compiled YAML at {output_yaml} is suspiciously small ({len(content)} bytes). "
            "Compile may have failed silently."
        )

        print(
            f"\nCompiled pipeline YAML: {output_yaml} "
            f"({len(content)} bytes)\n"
            f"kfp version: {kfp.__version__}"
        )


class TestKFPEndpointDiscovery:
    """Level 3d: Document candidate KFP v2 API endpoints from discovered substrate.

    This test does not attempt authentication or connection. It assembles
    the candidate endpoint list from discovered services and routes so that
    the next implementation step (DSPA pipeline upload) has a clear target.
    """

    def test_kfp_client_candidate_endpoint_documented(
        self, test_namespace: str
    ) -> None:
        """Document in-cluster and external KFP API endpoint candidates.

        Candidate endpoints are assembled from:
          1. In-cluster services with port 8888 (KFP HTTP API, no OAuth):
               http://<service-name>.<namespace>.svc.cluster.local:8888
          2. External routes with 'ds-pipeline' prefix (OAuth required):
               https://<route-host>

        The DSPA API base path for KFP v2 is:
          /apis/v2beta1/

        At least one in-cluster candidate should always be available if
        the DSPA pods are Running/Ready. External routes are optional.

        This test does NOT attempt to connect to any endpoint.
        Authentication (OAuth token) is required for external routes.
        """
        services = _get_services(test_namespace)
        routes = _get_routes(test_namespace)

        # In-cluster candidates: services with KFP HTTP port.
        in_cluster_candidates = [
            f"http://{svc['metadata']['name']}.{test_namespace}.svc.cluster.local:{_KFP_HTTP_PORT}"
            for svc in services
            if _KFP_HTTP_PORT in _service_ports(svc)
        ]

        # External candidates: ds-pipeline routes with a host.
        external_candidates = [
            f"https://{r['spec']['host']}"
            for r in routes
            if r["metadata"]["name"].startswith("ds-pipeline")
            and r["spec"].get("host")
        ]

        all_candidates = in_cluster_candidates + external_candidates

        report = _discovery_report(
            namespace=test_namespace,
            dspa_names=[],
            dspa_versions=[],
            pipeline_pods=[],
            candidate_services=[
                svc["metadata"]["name"]
                for svc in services
                if _KFP_HTTP_PORT in _service_ports(svc)
            ],
            candidate_routes=[
                r["spec"].get("host", "")
                for r in routes
                if r["metadata"]["name"].startswith("ds-pipeline")
            ],
            kfp_version=None,
            generated_yaml_path=None,
        )

        assert len(all_candidates) > 0, (
            f"No KFP API endpoint candidates found in namespace {test_namespace!r}. "
            "Expected at least one in-cluster service with port 8888. "
            "Verify DSPA pods are Running/Ready.\n" + report
        )

        print(f"\n{report}")
        print("\nCandidate KFP endpoints (not yet connected):")
        for endpoint in all_candidates:
            note = " (OAuth token required)" if endpoint.startswith("https://") else ""
            print(f"  {endpoint}{note}")
        print("API base path: /apis/v2beta1/")
        print("Next step: implement kfp.Client(host=<endpoint>, ...) with OAuth token.")


class TestDSPARuntimeFuture:
    """Level 3d: Boundary marker — pipeline submission is Level 3, not Level 3d.

    This class exists to prevent any ambiguity: Level 3d is discovery only.
    Actual pipeline upload and run creation is Level 3 (test_03_pipeline_smoke_run.py),
    which is now IMPLEMENTED and proven working on the cluster.

    Level 3 implementation proven:
      - src/workbench/_submit.py: get_dspa_endpoint, make_kfp_client,
        upload_pipeline, submit_pipeline_run
      - Endpoint: https://ds-pipeline-pipelines-definition.<ns>.svc.cluster.local:8888
      - Auth: SA token (verify_ssl=False for self-signed cert)
      - kfp 2.7.0 API: upload_pipeline → list_pipeline_versions → run_pipeline
    """

    @pytest.mark.xfail(
        reason=(
            "Level 3d (this file) is discovery only — it maps the substrate and "
            "documents candidate endpoints without creating resources. "
            "Level 3 pipeline upload + run creation is IMPLEMENTED in "
            "src/workbench/_submit.py and proven in test_03_pipeline_smoke_run.py. "
            "This placeholder remains xfail to preserve the Level 3d / Level 3 boundary."
        ),
        strict=False,
    )
    def test_dspa_runtime_submission_marked_future(
        self, test_namespace: str
    ) -> None:
        """Boundary marker: Level 3d is discovery only; Level 3 is implemented.

        Level 3 (pipeline upload + run creation) is now implemented in
        src/workbench/_submit.py and proven via test_03_pipeline_smoke_run.py.

        This test remains xfail to mark the Level 3d / Level 3 boundary cleanly.
        Level 3d tests (this file) never create resources. Level 3 tests do.
        """
        pytest.xfail(
            "Level 3d is discovery only. "
            "Level 3 pipeline upload + run creation is implemented — "
            "see test_03_pipeline_smoke_run.py."
        )
