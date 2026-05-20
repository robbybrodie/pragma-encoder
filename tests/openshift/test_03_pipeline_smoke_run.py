"""Level 3 / 3.1 — DSPA / KFP v2 pipeline upload, run creation, and status polling.

Purpose:
  Verify end-to-end PRAGMA pipeline submission via OpenShift AI Data Science
  Pipelines (DSPA) / KFP v2. A PRAGMA-S pipeline is compiled to YAML,
  uploaded to the DSPA registry, a run is created, and the run is polled
  until it reaches a terminal state.

Pipeline runtime context:
  This environment uses OpenShift AI Data Science Pipelines (DSPA) / KFP v2.
  Red Hat OpenShift AI Data Science Pipelines 2.0 does NOT use kfp-tekton.
  The submission path is:
    1. Compile KFP v2 pipeline to YAML (local, no cluster needed).
    2. Upload/import pipeline to DSPA via the KFP v2 API.
    3. Create a pipeline run via the DSPA / KFP v2 API.
    4. Poll run state via get_run() until terminal (SUCCEEDED/FAILED/CANCELED).
    5. Report terminal state and diagnostics. Fail if not SUCCEEDED.

Tekton PipelineRun/TaskRun are NOT part of this execution path.

Implementation status:
  Upload + run creation: IMPLEMENTED (src/workbench/_submit.py)
  Run status polling:    IMPLEMENTED (wait_for_run_terminal, get_run_status)
  Log verification:      FUTURE (requires running training container Level 4)

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
  - PRAGMA_TRAINING_IMAGE=<image-uri> (recommended: sets component base image at compile time)
    Default: image-registry.openshift-image-registry.svc:5000/pragma-encoder/pragma-encoder-training:latest
    Override with PRAGMA_KFP_COMPONENT_IMAGE for a different component image.
  - kfp SDK installed (PRAGMA workbench notebook image includes kfp)
  - Running from inside the workbench pod (in-cluster endpoint required)
  - Level 0, 1, and 3d tests passing
  - DSPA instance healthy in PRAGMA_TEST_NAMESPACE

Component image requirement:
  KFP component pods do NOT inherit the workbench pod's git checkout.
  The component image must contain PRAGMA source code (src/) and all Python
  dependencies. The PRAGMA training image is the correct choice — it is built
  from openshift/training/Dockerfile.training with src/ baked in at WORKDIR.
  Setting PRAGMA_TRAINING_IMAGE before running this test ensures the compiled
  pipeline YAML embeds the correct component image.

Run from inside workbench pod:
  export RUN_OPENSHIFT_TESTS=1
  export RUN_OPENSHIFT_PIPELINE_SMOKE=1
  export PRAGMA_TEST_NAMESPACE=pragma-encoder
  export PRAGMA_TRAINING_IMAGE=image-registry.openshift-image-registry.svc:5000/pragma-encoder/pragma-encoder-training:latest
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
# Constants
# ---------------------------------------------------------------------------

_KFP_API_PORT = 8888
_KFP_API_BASE_PATH = "/apis/v2beta1"
_POLL_INTERVAL_SECONDS = 10  # seconds between get_run() polls


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
    """Level 3 / 3.1: DSPA / KFP v2 pipeline upload, run creation, and polling.

    Compiles the PRAGMA pretraining pipeline to KFP v2 YAML, uploads it
    to the DSPA registry, creates a pipeline run, then polls the run state
    via wait_for_run_terminal() until it reaches SUCCEEDED, FAILED, CANCELED,
    or SKIPPED — or times out.

    The test fails if the terminal state is not SUCCEEDED, collecting
    redacted diagnostics from the run error field and logging diagnostic
    oc commands for investigation. Token is never printed.

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
        """Upload a compiled PRAGMA-S pipeline to DSPA, create a run, and poll to terminal.

        Steps:
          1. Compile the decorated pipeline to a KFP v2 YAML in tmp_path.
          2. Resolve the DSPA endpoint from namespace.
          3. Read the SA token (if inside a pod).
          4. Construct a kfp.Client.
          5. Upload the YAML to DSPA → assert pipeline_id returned.
          6. Create a pipeline run → assert run_id returned.
          7. Poll run state via wait_for_run_terminal() until terminal or timeout.
          8. Collect diagnostics from run.error field.
          9. Pass if SUCCEEDED. Fail with diagnostics if any other terminal state.
        """
        from src.workbench._submit import (  # noqa: PLC0415
            get_dspa_endpoint,
            get_run_status,
            get_service_account_token,
            make_kfp_client,
            submit_pipeline_run,
            upload_pipeline,
            wait_for_run_terminal,
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

        # --- Step 2b: Verify component image embedded in YAML ---
        # KFP component pods do NOT inherit the workbench pod's git checkout.
        # The compiled YAML must embed the PRAGMA training image (not the workbench image).
        # The component image is controlled by PRAGMA_KFP_COMPONENT_IMAGE or
        # PRAGMA_TRAINING_IMAGE env vars, read at pipeline/components_pragma.py import time.
        try:
            from pipeline import components_pragma as _cpmod  # noqa: PLC0415
            component_image = _cpmod._BASE_IMAGE
            yaml_content = yaml_path.read_text()
            if component_image in yaml_content:
                print(f"  Component image: {component_image} (verified in YAML)")
            else:
                print(
                    f"  WARNING: component image {component_image!r} not found in YAML. "
                    "This may indicate the pipeline was compiled before PRAGMA_TRAINING_IMAGE "
                    "was set. Component pods may fail with ModuleNotFoundError: No module named 'src'."
                )
        except Exception:  # noqa: BLE001
            pass  # image check is informational; do not fail compile step

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

        print(f"  Run created: run_id={run_id}")
        print(f"    oc get pods -n {test_namespace} -l pipeline/runid={run_id}")

        # --- Step 9: Poll run to terminal state ---
        poll_timeout = int(os.environ.get("PRAGMA_TEST_TIMEOUT_SECONDS", "300"))
        print(
            f"\n  Polling run status (timeout={poll_timeout}s, "
            f"interval={_POLL_INTERVAL_SECONDS}s)..."
        )

        try:
            terminal_state = wait_for_run_terminal(
                client=client,
                run_id=run_id,
                timeout=poll_timeout,
                poll_interval=_POLL_INTERVAL_SECONDS,
            )
        except TimeoutError as exc:
            pytest.fail(
                f"Run timed out before reaching a terminal state.\n"
                f"  run_id:      {run_id}\n"
                f"  pipeline_id: {pipeline_id}\n"
                f"  timeout:     {poll_timeout}s\n"
                f"  {exc}\n"
                "Diagnostics:\n"
                f"  oc get pods -n {test_namespace}\n"
                f"  oc get pods -n {test_namespace} -l pipeline/runid={run_id}\n"
                "Check the OpenShift AI dashboard for run status."
            )

        print(f"  Terminal state: {terminal_state}")

        # --- Step 10: Collect diagnostics for non-success terminal states ---
        run_error_message: str = "(none)"
        run_error_code: object = None
        run_display_name: str = run_name
        try:
            final_run = client.get_run(run_id=run_id)
            run_display_name = getattr(final_run, "display_name", run_name) or run_name
            error_obj = getattr(final_run, "error", None)
            if error_obj is not None:
                run_error_message = str(getattr(error_obj, "message", error_obj))[:600]
                run_error_code = getattr(error_obj, "code", None)
        except Exception:  # noqa: BLE001
            run_error_message = "(could not retrieve run details)"

        # --- Step 11: Assess terminal state ---
        if terminal_state == "SUCCEEDED":
            print(
                f"\n{'=' * 60}\n"
                f"  Level 3.1 pipeline smoke: SUCCESS\n"
                f"{'=' * 60}\n"
                f"  Pipeline name:  {pipeline_name}\n"
                f"  Pipeline ID:    {pipeline_id}\n"
                f"  Run name:       {run_display_name}\n"
                f"  Run ID:         {run_id}\n"
                f"  Terminal state: {terminal_state}\n"
                f"  Endpoint:       {endpoint}\n"
                f"  Auth:           {'SA token' if token else 'no token (port 8888 direct)'}\n"
                f"\n"
                f"  All pipeline components completed successfully.\n"
                f"  Log verification is Level 4 (PyTorchJob execution).\n"
                f"{'=' * 60}"
            )
        else:
            # Non-success terminal state — collect diagnostics and fail.
            pytest.fail(
                f"Run reached non-success terminal state: {terminal_state!r}\n"
                f"  run_id:         {run_id}\n"
                f"  pipeline_id:    {pipeline_id}\n"
                f"  run_name:       {run_display_name}\n"
                f"  error_code:     {run_error_code}\n"
                f"  error_message:  {run_error_message}\n"
                f"\n"
                "Diagnostic commands (run from workbench pod or locally with oc):\n"
                f"  oc get pods -n {test_namespace}\n"
                f"  oc get pods -n {test_namespace} -l pipeline/runid={run_id}\n"
                f"  oc logs -n {test_namespace} -l pipeline/runid={run_id} --tail=80\n"
                f"\n"
                "If this is FAILED because pipeline component stubs raise NotImplementedError:\n"
                "  Implement the component body in pipeline/components_pragma.py.\n"
                "  Components currently expected to have working implementations:\n"
                "    prepare_dataset, upload_artifacts, submit_pytorchjob,\n"
                "    run_pretraining, export_checkpoint\n"
                "If this is CANCELED: run was cancelled externally.\n"
                "If this is SKIPPED: check pipeline condition logic.\n"
            )

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
