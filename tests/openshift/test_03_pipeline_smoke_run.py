"""Level 3 — DSPA / KFP v2 pipeline upload and run creation smoke.

Purpose:
  Verify end-to-end PRAGMA pipeline submission via OpenShift AI Data Science
  Pipelines (DSPA) / KFP v2. A PRAGMA-S pipeline is compiled to YAML,
  uploaded to the DSPA registry, and a run is created.

Pipeline runtime context:
  This environment uses OpenShift AI Data Science Pipelines (DSPA) / KFP v2.
  Red Hat OpenShift AI Data Science Pipelines 2.0 does NOT use kfp-tekton.
  The submission path is:
    1. Compile KFP v2 pipeline to YAML (local, no cluster needed).
    2. Upload/import pipeline to DSPA via the KFP v2 API.
    3. Create a pipeline run via the DSPA / KFP v2 API.
    4. Verify run was created (run_id returned).
  Waiting for run completion and verifying training logs is a future step
  that requires PyTorchJob execution (Level 4).

Tekton PipelineRun/TaskRun are NOT part of this execution path.

Implementation status:
  Upload + run creation: IMPLEMENTED (src/workbench/_submit.py)
  Wait for completion:   FUTURE (requires PyTorchJob Level 4)
  Log verification:      FUTURE (requires running training container)

Safety rules:
  - Skip unless RUN_OPENSHIFT_TESTS=1 (conftest gate) + RUN_OPENSHIFT_PIPELINE_SMOKE=1.
  - Skip (not fail) when kfp is not installed — environment issue, not code issue.
  - Skip (not fail) when DSPA endpoint is unreachable — not in cluster.
  - Compile only writes to tmp_path — no permanent local side effects.
  - Never modify Argo CD-managed resources.
  - Never delete secrets, serviceaccounts, or namespaces.
  - Do not use oc exec as a pipeline execution path.

Prerequisites (when enabled):
  - RUN_OPENSHIFT_TESTS=1
  - RUN_OPENSHIFT_PIPELINE_SMOKE=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - kfp SDK installed (PRAGMA workbench notebook image includes kfp)
  - Running from inside the workbench pod (in-cluster endpoint required)
  - Level 0, 1, and 3d tests passing
  - DSPA instance healthy in PRAGMA_TEST_NAMESPACE

Run from inside workbench pod:
  export RUN_OPENSHIFT_TESTS=1
  export RUN_OPENSHIFT_PIPELINE_SMOKE=1
  export PRAGMA_TEST_NAMESPACE=pragma-encoder
  PYTHONPATH=. python -m pytest tests/openshift/test_03_pipeline_smoke_run.py -v -s
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import socket
import urllib.error
import urllib.request

import ssl

import pytest

# ---------------------------------------------------------------------------
# Suite-level skip guard
# ---------------------------------------------------------------------------

_PIPELINE_SMOKE_ENABLED = os.environ.get("RUN_OPENSHIFT_PIPELINE_SMOKE") == "1"

_require_smoke = pytest.mark.skipif(
    not _PIPELINE_SMOKE_ENABLED,
    reason=(
        "DSPA / KFP v2 pipeline smoke tests are opt-in. "
        "Set RUN_OPENSHIFT_PIPELINE_SMOKE=1 to enable. "
        "Run from inside the PRAGMA workbench pod (in-cluster endpoint required). "
        "Warning: these tests upload a pipeline to DSPA and create a run "
        "in PRAGMA_TEST_NAMESPACE."
    ),
)

_KFP_AVAILABLE = importlib.util.find_spec("kfp") is not None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KFP_API_PORT = 8888
_KFP_API_BASE_PATH = "/apis/v2beta1"


def _endpoint_reachable(endpoint: str, timeout: int = 5) -> bool:
    """Return True if the KFP API endpoint is reachable (HTTP or HTTPS).

    Uses an unverified SSL context because the in-cluster DSPA service uses
    a self-signed TLS certificate on port 8888.
    """
    health_url = f"{endpoint}{_KFP_API_BASE_PATH}/healthz"
    req = urllib.request.Request(health_url, method="GET")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        urllib.request.urlopen(req, timeout=timeout, context=ctx)
        return True
    except urllib.error.HTTPError:
        return True  # HTTP error means TCP + TLS worked; endpoint is reachable
    except (ConnectionRefusedError, OSError, socket.timeout, urllib.error.URLError):
        return False


def _compile_pipeline(tmp_path: pathlib.Path, pipeline_name: str) -> pathlib.Path:
    """Compile the PRAGMA pretraining pipeline to a temp YAML.

    Uses the PragmaPipeline decorator API. Returns the path to the YAML.
    Raises RuntimeError if kfp is not installed (propagated to skip the test).
    """
    from src.workbench import dataset, pragma_pipeline, train  # noqa: PLC0415

    @pragma_pipeline(name=pipeline_name)
    def _smoke():
        ds = dataset("ibm-tabformer", prepare_if_missing=True)
        train(dataset=ds, model_size="S", epochs=1, max_steps=1)

    yaml_path = tmp_path / f"{pipeline_name}.yaml"
    _smoke.compile(str(yaml_path))  # raises RuntimeError if kfp absent
    return yaml_path


# ---------------------------------------------------------------------------
# Pipeline upload smoke — Level 3 implementation
# ---------------------------------------------------------------------------


class TestOpenShiftPipelineSmoke:
    """Level 3: DSPA / KFP v2 pipeline upload and run creation.

    Compiles the PRAGMA pretraining pipeline to KFP v2 YAML, uploads it
    to the DSPA registry, and creates a pipeline run. Verifies that the
    DSPA API accepted the submission (run_id returned).

    Does NOT wait for run completion or inspect training logs — that
    requires PyTorchJob execution (Level 4).

    Skips (not fails) when:
      - kfp is not installed (run from PRAGMA workbench image)
      - DSPA endpoint is not reachable (run from inside workbench pod)
    """

    @_require_smoke
    def test_pipeline_smoke_upload_and_run(
        self,
        test_namespace: str,
        test_id: str,
        tmp_path: pathlib.Path,
    ) -> None:
        """Upload a compiled PRAGMA-S pipeline to DSPA and create a run.

        Steps:
          1. Compile the decorated pipeline to a KFP v2 YAML in tmp_path.
          2. Resolve the DSPA endpoint from namespace.
          3. Read the SA token (if inside a pod).
          4. Construct a kfp.Client.
          5. Upload the YAML to DSPA → assert pipeline_id returned.
          6. Create a pipeline run → assert run_id returned.
          7. Print diagnostics for confirmation.

        This test is the implementation of test_pipeline_smoke_run_future.
        It xpasses once the full environment is available (workbench pod).
        """
        from src.workbench._submit import (  # noqa: PLC0415
            get_dspa_endpoint,
            get_service_account_token,
            make_kfp_client,
            submit_pipeline_run,
            upload_pipeline,
        )

        # --- Step 1: kfp guard ---
        if not _KFP_AVAILABLE:
            pytest.skip(
                "kfp is not installed in the current environment. "
                "Run this test from the PRAGMA workbench pod where kfp is pre-installed:\n"
                f"  oc exec -n {test_namespace} <workbench-pod> -- \\\n"
                f"    env RUN_OPENSHIFT_TESTS=1 RUN_OPENSHIFT_PIPELINE_SMOKE=1 \\\n"
                f"    PRAGMA_TEST_NAMESPACE={test_namespace} \\\n"
                "    python -m pytest tests/openshift/test_03_pipeline_smoke_run.py -v -s"
            )

        # --- Step 2: Compile pipeline ---
        pipeline_name = f"pragma-s-smoke-{test_id}"
        print(f"\nCompiling pipeline: {pipeline_name}")
        try:
            yaml_path = _compile_pipeline(tmp_path, pipeline_name)
        except (RuntimeError, ImportError) as exc:
            pytest.skip(
                f"Pipeline compile failed (kfp may not be fully installed): {exc}\n"
                "Run from the PRAGMA workbench image where kfp is pre-installed."
            )
            return  # unreachable; appease type checker

        assert yaml_path.exists(), f"Compiled YAML not found at {yaml_path}"
        yaml_size = yaml_path.stat().st_size
        print(f"  Compiled YAML: {yaml_path} ({yaml_size} bytes)")

        # --- Step 3: Resolve endpoint ---
        endpoint = get_dspa_endpoint(namespace=test_namespace)
        print(f"  DSPA endpoint: {endpoint}")

        # --- Step 4: Reachability guard ---
        if not _endpoint_reachable(endpoint):
            pytest.skip(
                f"DSPA endpoint not reachable: {endpoint}\n"
                "Run from inside the PRAGMA workbench pod:\n"
                f"  oc exec -n {test_namespace} <workbench-pod> -- \\\n"
                f"    env RUN_OPENSHIFT_TESTS=1 RUN_OPENSHIFT_PIPELINE_SMOKE=1 \\\n"
                f"    PRAGMA_TEST_NAMESPACE={test_namespace} \\\n"
                "    python -m pytest tests/openshift/test_03_pipeline_smoke_run.py -v -s"
            )

        # --- Step 5: Auth ---
        token = get_service_account_token()
        if token is not None:
            print(f"  SA token: present ({len(token)} chars, not printed)")
        else:
            print("  SA token: absent — trying without token (port 8888 direct)")

        # --- Step 6: Construct kfp.Client ---
        try:
            client = make_kfp_client(endpoint=endpoint, token=token)
        except RuntimeError as exc:
            pytest.skip(f"kfp.Client construction failed: {exc}")
            return

        print(f"  kfp.Client: constructed for {endpoint}")

        # --- Step 7: Upload pipeline ---
        try:
            pipeline_id = upload_pipeline(
                client=client,
                yaml_path=yaml_path,
                pipeline_name=pipeline_name,
            )
        except Exception as exc:  # noqa: BLE001
            exc_type = type(exc).__name__
            exc_msg = str(exc)[:400]
            pytest.fail(
                f"upload_pipeline() failed.\n"
                f"  Endpoint: {endpoint}\n"
                f"  YAML:     {yaml_path}\n"
                f"  Name:     {pipeline_name}\n"
                f"  Error:    {exc_type}: {exc_msg}\n"
                "Diagnostics:\n"
                f"  oc get pods -n {test_namespace} -l app=ds-pipeline\n"
                f"  oc describe datasciencepipelinesapplication -n {test_namespace}\n"
                "Run Level 3d discovery first to confirm DSPA substrate is healthy:\n"
                f"  RUN_OPENSHIFT_TESTS=1 PRAGMA_TEST_NAMESPACE={test_namespace} "
                "pytest tests/openshift/test_03_dspa_runtime_discovery.py -v"
            )

        assert isinstance(pipeline_id, str) and pipeline_id, (
            f"upload_pipeline() must return a non-empty pipeline_id string. "
            f"Got: {pipeline_id!r}"
        )
        print(f"  Pipeline uploaded: pipeline_id={pipeline_id}")

        # --- Step 8: Create run ---
        run_name = f"smoke-{test_id}"
        try:
            run_id = submit_pipeline_run(
                client=client,
                pipeline_id=pipeline_id,
                run_name=run_name,
                arguments={"dataset_name": "ibm-tabformer", "model_size": "S"},
                experiment_name="pragma-smoke",
            )
        except Exception as exc:  # noqa: BLE001
            exc_type = type(exc).__name__
            exc_msg = str(exc)[:400]
            pytest.fail(
                f"submit_pipeline_run() failed.\n"
                f"  Endpoint:    {endpoint}\n"
                f"  pipeline_id: {pipeline_id}\n"
                f"  run_name:    {run_name}\n"
                f"  Error:       {exc_type}: {exc_msg}\n"
                "The pipeline was uploaded successfully but run creation failed.\n"
                "Check DSPA service account RBAC and experiment permissions."
            )

        assert isinstance(run_id, str) and run_id, (
            f"submit_pipeline_run() must return a non-empty run_id string. "
            f"Got: {run_id!r}"
        )

        print(
            f"\n{'=' * 60}\n"
            f"  Level 3 pipeline smoke: SUCCESS\n"
            f"{'=' * 60}\n"
            f"  Pipeline name: {pipeline_name}\n"
            f"  Pipeline ID:   {pipeline_id}\n"
            f"  Run name:      {run_name}\n"
            f"  Run ID:        {run_id}\n"
            f"  Endpoint:      {endpoint}\n"
            f"  Auth:          {'SA token' if token else 'no token (port 8888 direct)'}\n"
            f"\n"
            f"  Monitor run in OpenShift AI dashboard or via:\n"
            f"    oc get pods -n {test_namespace} -l pipeline/runid={run_id}\n"
            f"{'=' * 60}"
        )

        # Note: run completion and log verification are Level 4 (PyTorchJob).
        # This test verifies only that the DSPA API accepted the submission.

    @_require_smoke
    def test_pipeline_compile_produces_valid_yaml(
        self,
        test_namespace: str,
        tmp_path: pathlib.Path,
    ) -> None:
        """Compile step produces a non-empty KFP v2 YAML (no cluster required).

        This validates the compile step independently from upload. It runs
        whenever RUN_OPENSHIFT_PIPELINE_SMOKE=1 even outside the cluster.
        """
        if not _KFP_AVAILABLE:
            pytest.skip(
                "kfp is not installed — compile test skipped. "
                "Install kfp or use the PRAGMA workbench image."
            )

        try:
            yaml_path = _compile_pipeline(tmp_path, "pragma-s-compile-check")
        except (RuntimeError, ImportError) as exc:
            pytest.skip(f"Compile failed (kfp environment issue): {exc}")
            return

        assert yaml_path.exists(), f"YAML file not created at {yaml_path}"
        content = yaml_path.read_text()
        assert len(content) > 100, (  # noqa: PLR2004
            f"Compiled YAML is suspiciously small ({len(content)} bytes). "
            "Compile may have failed silently."
        )
        assert "schemaVersion" in content, (
            "Compiled YAML does not appear to be a valid KFP v2 YAML "
            "(missing 'schemaVersion')."
        )
        print(
            f"\nCompile check: {yaml_path.name} — {len(content)} bytes, "
            "schemaVersion present"
        )
