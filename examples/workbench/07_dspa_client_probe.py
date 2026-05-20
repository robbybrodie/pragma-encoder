"""Level 3e — DSPA / KFP v2 client connectivity probe (standalone script).

Purpose:
  Prove the exact kfp.Client configuration required to reach the OpenShift AI
  DSPA / KFP v2 API from inside the PRAGMA workbench pod.

  Run this script from inside the workbench pod to confirm which client
  configuration reaches the DSPA API. It is read-only: no pipelines are
  uploaded, no runs are created.

Usage (from inside workbench pod):
  export PRAGMA_TEST_NAMESPACE=pragma-encoder
  cd /opt/app-root/src
  PYTHONPATH=. python examples/workbench/07_dspa_client_probe.py

Usage (local — endpoint will be unreachable, output reports environment):
  PRAGMA_TEST_NAMESPACE=pragma-encoder python examples/workbench/07_dspa_client_probe.py

Authentication:
  Port 8888 is the direct KFP API server — in-cluster, no OAuth proxy.
  Port 8443 is the OAuth-protected port (requires Bearer token).

  This probe targets port 8888. It tries without a token first, then falls
  back to the mounted SA token if available (inside pod only).

Safety:
  - Read-only: no pipeline uploads, no run creation.
  - SA token never printed.
  - No oc exec.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import socket
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_NAMESPACE = os.environ.get("PRAGMA_TEST_NAMESPACE", "pragma-encoder")
_KFP_API_PORT = 8888
_KFP_API_BASE_PATH = "/apis/v2beta1"
_SA_TOKEN_PATH = pathlib.Path("/var/run/secrets/kubernetes.io/serviceaccount/token")

_KFP_AVAILABLE = importlib.util.find_spec("kfp") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_endpoint(service_name: str = "ds-pipeline-pipelines-definition") -> str:
    return f"http://{service_name}.{_NAMESPACE}.svc.cluster.local:{_KFP_API_PORT}"


def _probe_http(url: str, token: str | None = None, timeout: int = 5) -> int | None:
    """Attempt a GET request and return the HTTP status code, or None if unreachable."""
    req = urllib.request.Request(url, method="GET")
    if token is not None:
        req.headers["Authorization"] = f"Bearer {token}"  # never printed
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (ConnectionRefusedError, OSError, socket.timeout, urllib.error.URLError):
        return None


def _read_sa_token() -> str | None:
    """Return SA token if inside a pod, else None. Never printed."""
    if _SA_TOKEN_PATH.exists():
        try:
            return _SA_TOKEN_PATH.read_text().strip()
        except OSError:
            return None
    return None


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Probe steps
# ---------------------------------------------------------------------------


def probe_environment() -> dict:
    """Report environment: namespace, kfp version, inside-pod indicator."""
    _section("1. Environment")

    is_inside_pod = _SA_TOKEN_PATH.exists()
    kfp_version = None
    if _KFP_AVAILABLE:
        import kfp  # noqa: PLC0415
        kfp_version = kfp.__version__

    print(f"  Namespace:         {_NAMESPACE}")
    print(f"  Running inside pod: {is_inside_pod}")
    print(
        f"  SA token:          "
        + ("present (not printed)" if is_inside_pod else "absent (outside cluster)")
    )
    print(
        f"  kfp SDK:           "
        + (f"v{kfp_version}" if kfp_version else "not installed")
    )
    if not _KFP_AVAILABLE:
        print("  → Install kfp: pip install kfp")
        print("  → Or use the PRAGMA workbench notebook image (kfp pre-installed)")

    return {
        "namespace": _NAMESPACE,
        "is_inside_pod": is_inside_pod,
        "kfp_version": kfp_version,
    }


def probe_endpoints() -> dict:
    """Probe candidate in-cluster endpoints for reachability."""
    _section("2. Endpoint Reachability")

    candidates = [
        ("primary", "ds-pipeline-pipelines-definition"),
        ("alias", "ml-pipeline"),
    ]

    results: dict[str, int | None] = {}

    for label, svc_name in candidates:
        endpoint = _build_endpoint(svc_name)
        health_url = f"{endpoint}{_KFP_API_BASE_PATH}/healthz"
        status = _probe_http(health_url, timeout=5)

        if status is None:
            verdict = "UNREACHABLE (not in cluster — run from inside workbench pod)"
        elif status in (200, 201):
            verdict = f"REACHABLE — HTTP {status} (no auth required on port 8888)"
        elif status in (401, 403):
            verdict = f"REACHABLE — HTTP {status} (auth required on port 8888)"
        else:
            verdict = f"HTTP {status} (unexpected — check DSPA configuration)"

        print(f"  [{label}] {endpoint}")
        print(f"    healthz status: {verdict}")
        results[label] = status

    return results


def probe_auth(endpoint_results: dict) -> dict:
    """Attempt kfp.Client list_pipelines with and without SA token."""
    _section("3. KFP Client Auth Probe")

    if not _KFP_AVAILABLE:
        print("  SKIP — kfp not installed. Install with: pip install kfp")
        return {"skip": "kfp not installed"}

    import kfp  # noqa: PLC0415

    # Select first reachable endpoint.
    endpoint = None
    for svc_name in ("ds-pipeline-pipelines-definition", "ml-pipeline"):
        candidate = _build_endpoint(svc_name)
        health_url = f"{candidate}{_KFP_API_BASE_PATH}/healthz"
        status = _probe_http(health_url, timeout=5)
        if status is not None:
            endpoint = candidate
            break

    if endpoint is None:
        print(
            "  SKIP — no in-cluster endpoint reachable.\n"
            "  Run from inside the PRAGMA workbench pod:\n"
            f"    oc exec -n {_NAMESPACE} <workbench-pod> -- \\\n"
            f"      env PRAGMA_TEST_NAMESPACE={_NAMESPACE} \\\n"
            "      python examples/workbench/07_dspa_client_probe.py"
        )
        return {"skip": "endpoint not reachable"}

    print(f"  Using endpoint: {endpoint}")

    # --- Attempt 1: no token ---
    print("\n  Attempt 1: kfp.Client without token (port 8888 direct)")
    client = kfp.Client(host=endpoint)
    no_token_success = False
    pipeline_count = 0
    try:
        result = client.list_pipelines()
        pipeline_count = len(result.pipelines) if result.pipelines else 0
        no_token_success = True
        print(f"  → SUCCESS: list_pipelines() returned {pipeline_count} pipeline(s)")
        print("  → Port 8888 does NOT require authentication in-cluster.")
    except Exception as exc:  # noqa: BLE001
        print(f"  → FAILED: {type(exc).__name__}: {str(exc)[:120]}")
        print("  → Port 8888 requires authentication. Trying SA token...")

    if no_token_success:
        return {
            "endpoint": endpoint,
            "auth_method": "none",
            "pipeline_count": pipeline_count,
            "client_config": f"kfp.Client(host='{endpoint}')",
        }

    # --- Attempt 2: SA token ---
    print("\n  Attempt 2: kfp.Client with mounted SA token")
    token = _read_sa_token()
    if token is None:
        print(
            f"  SKIP — SA token not found at {_SA_TOKEN_PATH}.\n"
            "  Run from inside a Kubernetes pod (workbench)."
        )
        return {"skip": "SA token not available outside pod"}

    print(f"  SA token: present ({len(token)} chars, not printed)")
    client_with_token = kfp.Client(host=endpoint, existing_token=token)
    try:
        result = client_with_token.list_pipelines()
        pipeline_count = len(result.pipelines) if result.pipelines else 0
        print(f"  → SUCCESS: list_pipelines() returned {pipeline_count} pipeline(s)")
        print("  → SA token grants access to DSPA KFP v2 API.")
        return {
            "endpoint": endpoint,
            "auth_method": "sa_token",
            "pipeline_count": pipeline_count,
            "client_config": (
                f"import pathlib\n"
                f"token = pathlib.Path('{_SA_TOKEN_PATH}').read_text().strip()\n"
                f"client = kfp.Client(host='{endpoint}', existing_token=token)"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  → FAILED with SA token: {type(exc).__name__}: {str(exc)[:120]}")
        print("  → SA token may not have DSPA RBAC rights.")
        return {
            "endpoint": endpoint,
            "auth_method": "sa_token_failed",
            "error": str(exc)[:200],
        }


def print_summary(env: dict, endpoints: dict, auth: dict) -> None:
    """Print the final configuration report and next steps."""
    _section("4. Summary and Next Steps")

    working_config = auth.get("client_config")
    auth_method = auth.get("auth_method")
    pipeline_count = auth.get("pipeline_count")
    skip_reason = auth.get("skip")

    if skip_reason:
        print(f"  Probe incomplete: {skip_reason}")
        print()
        print("  To run the full probe, execute from inside the PRAGMA workbench pod:")
        print(f"    oc exec -n {_NAMESPACE} <workbench-pod> -- \\")
        print(f"      env PRAGMA_TEST_NAMESPACE={_NAMESPACE} \\")
        print("      python examples/workbench/07_dspa_client_probe.py")
    elif working_config:
        print("  Confirmed working kfp.Client configuration:")
        print()
        print("  import kfp")
        if auth_method == "none":
            print(f"  client = kfp.Client(host='{auth.get('endpoint')}')")
        else:
            print(f"  import pathlib")
            print(
                f"  token = pathlib.Path('{_SA_TOKEN_PATH}').read_text().strip()"
            )
            print(f"  client = kfp.Client(host='{auth.get('endpoint')}', existing_token=token)")
        print()
        print(f"  list_pipelines() returned: {pipeline_count} pipeline(s)")
        print()
        print("  Next steps for pipeline upload and run creation:")
        print("    1. Compile pipeline to YAML:")
        print("         from pipeline.pragma_pipeline import pragma_pretraining_pipeline")
        print("         pragma_pretraining_pipeline.compile('pipeline.yaml')")
        print("    2. Upload to DSPA:")
        print("         client.upload_pipeline('pipeline.yaml', pipeline_name='pragma-s')")
        print("    3. Create a run:")
        print("         client.create_run_from_pipeline_func(")
        print("             pragma_pretraining_pipeline,")
        print("             experiment_name='pragma-pretraining',")
        print("             arguments={'model_size': 'S', 'max_steps': 1},")
        print("         )")
    else:
        print("  No working client configuration found.")
        print("  Check the DSPA pods and RBAC configuration.")
        print(f"    oc get pods -n {_NAMESPACE} -l app=ds-pipeline")
        print(f"    oc describe datasciencepipelinesapplication -n {_NAMESPACE}")

    print()
    print("  Run the full test suite from inside the workbench pod:")
    print(f"    export RUN_OPENSHIFT_TESTS=1")
    print(f"    export RUN_DSPA_CLIENT_PROBE=1")
    print(f"    export PRAGMA_TEST_NAMESPACE={_NAMESPACE}")
    print(
        "    PYTHONPATH=. python -m pytest "
        "tests/openshift/test_03e_kfp_client_probe.py -v -s"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = probe_environment()
    endpoints = probe_endpoints()
    auth = probe_auth(endpoints)
    print_summary(env, endpoints, auth)
    print()
