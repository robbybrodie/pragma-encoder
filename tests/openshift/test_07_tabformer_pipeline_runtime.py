"""Level 7 — OpenShift AI production pipeline runtime smoke (IBM TabFormer).

Purpose:
  Verify that the production PRAGMA pretraining pipeline
  (pipeline/pragma_pipeline.py :: pragma_pretraining_pipeline) runs
  end-to-end on OpenShift AI Data Science Pipelines (DSPA / KFP v2) using
  the IBM TabFormer dataset adapter with bounded parameters:
    - max_steps=2 (training exits after 2 gradient steps)
    - limit_rows=5 (only 5 customers used; fast data preparation)
    - model_size="S" (smallest config)
    - batch_size=1 (CPU-safe; no GPU required)
    - device="cpu"

  This is the Workbench-to-pipeline-to-model-publication path for the
  IBM TabFormer dataset. It exercises all five §2.4 stages:
    1. prepare_dataset  — DatasetAdapter prepares and returns manifest URI
    2. upload_artifacts — S3 upload (or pass-through if S3 not configured)
    3. submit_pytorchjob — PyTorchJob configuration stub
    4. run_pretraining  — Training via wheel-based module invocation
    5. export_checkpoint — Checkpoint publication to S3 export prefix

  Level 7 is NOT a duplicate of Level 3 (smoke pipeline test). Differences:
    Level 3 — smoke pipeline (pragma_smoke_pipeline.py), single component,
               no S3, synthetic data baked into the component.
    Level 7 — production pipeline (pragma_pipeline.py), all five components,
               IBM TabFormer dataset adapter, S3 via OpenShift AI Connection.

Architecture boundary:
  - Uses the DSPA submit path (tools/workbench/_submit.py).
  - IBM TabFormer CSV must be available in S3 or at DATA_URI env var.
  - S3 credentials come from OpenShift AI Connection (AWS_* env vars).
  - Namespace from PRAGMA_TEST_NAMESPACE.

Safety rules:
  - Skip unless BOTH RUN_OPENSHIFT_TESTS=1 AND RUN_OPENSHIFT_AI_PIPELINE_SMOKE=1.
  - Create only KFP Run resources. Never create namespaces, secrets, or SA.
  - SA token is NEVER printed, logged, or included in repr.
  - max_steps=2 + limit_rows=5 bounds wall-clock time to <5 min (CPU).

Prerequisites (when enabled):
  - RUN_OPENSHIFT_TESTS=1
  - RUN_OPENSHIFT_AI_PIPELINE_SMOKE=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - DSPA pods running in namespace (verified by Level 1 / Level 3)
  - IBM TabFormer data available at DATA_URI (S3 key) or local path
  - OpenShift AI Connection with AWS_* env vars injected into namespace

Platform version:
  Validated against OpenShift AI (RHOAI) 3.4.0.
  RHOAI 3.4 uses datasciencepipelinesapplications.opendatahub.io/v1 (DSPA),
  HardwareProfile in infrastructure.opendatahub.io, pytorchjobs.kubeflow.org/v1.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

from __future__ import annotations

import inspect
import os
import pathlib
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Suite-level skip guard
# ---------------------------------------------------------------------------

_OPENSHIFT_ENABLED = os.environ.get("RUN_OPENSHIFT_TESTS") == "1"
_PIPELINE_SMOKE_ENABLED = os.environ.get("RUN_OPENSHIFT_AI_PIPELINE_SMOKE") == "1"

_require_tabformer_pipeline = pytest.mark.skipif(
    not (_OPENSHIFT_ENABLED and _PIPELINE_SMOKE_ENABLED),
    reason=(
        "Level 7 TabFormer production pipeline tests are opt-in. "
        "Set both RUN_OPENSHIFT_TESTS=1 and RUN_OPENSHIFT_AI_PIPELINE_SMOKE=1 to enable. "
        "Also requires PRAGMA_TEST_NAMESPACE=<namespace> and IBM TabFormer data in S3."
    ),
)

# ---------------------------------------------------------------------------
# Bounded pipeline parameters (safe for CPU smoke runs)
# ---------------------------------------------------------------------------

_SMOKE_MAX_STEPS = int(os.environ.get("PRAGMA_SMOKE_MAX_STEPS", "2"))
_SMOKE_LIMIT_ROWS = int(os.environ.get("PRAGMA_SMOKE_LIMIT_ROWS", "5"))
_SMOKE_BATCH_SIZE = int(os.environ.get("PRAGMA_SMOKE_BATCH_SIZE", "1"))
_SMOKE_DEVICE = os.environ.get("PRAGMA_SMOKE_DEVICE", "cpu")
_SMOKE_MODEL_SIZE = os.environ.get("PRAGMA_SMOKE_MODEL_SIZE", "S")
_SMOKE_DATASET_NAME = os.environ.get("PRAGMA_SMOKE_DATASET_NAME", "ibm-tabformer")

# Run name label for S3 artifacts (helps identify smoke run outputs)
_SMOKE_RUN_NAME = os.environ.get("PRAGMA_SMOKE_RUN_NAME", "level7-tabformer-smoke")


# ===========================================================================
# 1. TestTabFormerPipelineLocalPrereqs
#    Local tests — no cluster access needed.
#    Run whenever RUN_OPENSHIFT_TESTS=1 (conftest skip guard applies).
#    Validate production pipeline can be compiled before attempting cluster run.
# ===========================================================================


class TestTabFormerPipelineLocalPrereqs:
    """Local pre-flight checks for the production pipeline.

    No cluster connection. These tests validate that the production pipeline
    compiles to valid YAML with the bounded smoke parameters, and that the
    pipeline signature exposes all required parameters.

    Run whenever RUN_OPENSHIFT_TESTS=1, regardless of RUN_OPENSHIFT_AI_PIPELINE_SMOKE.
    """

    def test_production_pipeline_importable(self) -> None:
        """pragma_pretraining_pipeline must be importable from pipeline.pragma_pipeline."""
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        assert callable(pragma_pretraining_pipeline), (
            "pragma_pretraining_pipeline must be callable. "
            "Check pipeline/pragma_pipeline.py for import errors."
        )

    def test_production_pipeline_has_max_steps(self) -> None:
        """pragma_pretraining_pipeline must expose max_steps parameter.

        max_steps=2 bounds the Level 7 smoke run to ≤2 gradient steps.
        Without this parameter, smoke runs cannot be bounded and would
        run for 10 full epochs (default).
        """
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_pretraining_pipeline)
        assert "max_steps" in sig.parameters, (
            "pragma_pretraining_pipeline must have max_steps parameter. "
            "Level 7 smoke uses max_steps=2 to bound wall-clock time."
        )

    def test_production_pipeline_has_limit_rows(self) -> None:
        """pragma_pretraining_pipeline must expose limit_rows parameter.

        limit_rows=5 bounds data preparation time to <30 seconds.
        Without this parameter, prepare_dataset would process all TabFormer rows.
        """
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_pretraining_pipeline)
        assert "limit_rows" in sig.parameters, (
            "pragma_pretraining_pipeline must have limit_rows parameter. "
            "Level 7 smoke uses limit_rows=5 to bound data prep time."
        )

    def test_production_pipeline_has_batch_size(self) -> None:
        """pragma_pretraining_pipeline must expose batch_size parameter.

        batch_size=1 makes the run CPU-safe — no GPU required for smoke.
        """
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_pretraining_pipeline)
        assert "batch_size" in sig.parameters, (
            "pragma_pretraining_pipeline must have batch_size parameter. "
            "Level 7 smoke uses batch_size=1 for CPU-safe operation."
        )

    def test_production_pipeline_has_device(self) -> None:
        """pragma_pretraining_pipeline must expose device parameter.

        device='cpu' forces CPU execution for smoke — no GPU provisioning needed.
        """
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_pretraining_pipeline)
        assert "device" in sig.parameters, (
            "pragma_pretraining_pipeline must have device parameter. "
            "Level 7 smoke uses device='cpu' for GPU-free operation."
        )

    def test_production_pipeline_has_run_name(self) -> None:
        """pragma_pretraining_pipeline must expose run_name parameter.

        run_name labels S3 artifacts so smoke run outputs can be identified
        and distinguished from production run outputs at the same prefix.
        """
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        sig = inspect.signature(pragma_pretraining_pipeline)
        assert "run_name" in sig.parameters, (
            "pragma_pretraining_pipeline must have run_name parameter. "
            "Level 7 smoke sets run_name='level7-tabformer-smoke' for artifact labelling."
        )

    def test_production_pipeline_compiles_to_yaml(self) -> None:
        """Production pipeline must compile to a valid KFP v2 YAML file.

        kfp is an optional dependency — this test is skipped when kfp is absent.
        When kfp is available, compile() must not raise and must produce a
        non-empty .yaml file at the given path.
        """
        kfp = pytest.importorskip("kfp", reason="kfp not installed — compile test skipped")
        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = pathlib.Path(tmpdir) / "pragma_pipeline_smoke.yaml"
            try:
                kfp.compiler.Compiler().compile(
                    pipeline_func=pragma_pretraining_pipeline,
                    package_path=str(yaml_path),
                )
            except Exception as exc:
                pytest.fail(
                    f"pragma_pretraining_pipeline failed to compile: {exc}. "
                    "Check pipeline/pragma_pipeline.py for KFP v2 compatibility issues."
                )

            assert yaml_path.exists(), (
                f"Compiler() did not produce {yaml_path}. "
                "Check kfp.compiler.Compiler().compile() call."
            )
            assert yaml_path.stat().st_size > 0, (
                f"Compiled pipeline YAML at {yaml_path} is empty. "
                "The compiled pipeline must contain component and DAG definitions."
            )

    def test_ibm_tabformer_adapter_registered(self) -> None:
        """IBM TabFormer adapter must be registered in the adapter registry.

        The production pipeline passes dataset_name='ibm-tabformer' to
        prepare_dataset, which resolves the adapter via get_adapter(). If the
        adapter is not registered, prepare_dataset raises KeyError at pipeline
        execution time (not compile time) — a silent failure mode.
        """
        from pragma_encoder.data.adapters import get_adapter  # noqa: PLC0415
        try:
            adapter_cls = get_adapter("ibm-tabformer")
        except KeyError:
            pytest.fail(
                "'ibm-tabformer' is not registered in the adapter registry. "
                "Level 7 requires the IBM TabFormer adapter to be registered in "
                "src/pragma_encoder/data/adapters/__init__.py. "
                "Check IBMTabFormerAdapter registration."
            )
        assert adapter_cls is not None, (
            "get_adapter('ibm-tabformer') returned None — registration is broken."
        )

    def test_components_use_wheel_based_training_invocation(self) -> None:
        """run_pretraining must invoke training as a Python module, not a source path.

        The component image has the pragma_encoder wheel installed in site-packages.
        Module invocation (python -m pragma_encoder.training.train) works.
        Direct file-path execution (src/pragma_encoder/...) does not — no src/
        directory exists in the component image.
        """
        import pipeline.components_pragma as mod  # noqa: PLC0415
        src = pathlib.Path(inspect.getfile(mod)).read_text()
        assert "pragma_encoder.training.train" in src, (
            "pipeline/components_pragma.py must reference pragma_encoder.training.train. "
            "run_pretraining must use module invocation: "
            "[sys.executable, '-m', 'pragma_encoder.training.train']."
        )
        assert "/src/pragma_encoder" not in src, (
            "pipeline/components_pragma.py must not reference /src/pragma_encoder. "
            "The component image has no src/ directory — use module invocation only."
        )

    def test_export_checkpoint_writes_export_manifest(self) -> None:
        """export_checkpoint component must reference export_manifest.json.

        export_manifest.json is the canonical artifact publication contract.
        It records checkpoint, metrics, vocab, and loss curve S3 URIs so
        downstream consumers can discover all run artifacts from a single key.
        """
        import pipeline.components_pragma as mod  # noqa: PLC0415
        src = pathlib.Path(inspect.getfile(mod)).read_text()
        assert "export_manifest.json" in src, (
            "pipeline/components_pragma.py export_checkpoint must write export_manifest.json. "
            "This is the canonical artifact publication record for the pipeline run."
        )


# ===========================================================================
# 2. TestTabFormerPipelineRuntime
#    Cluster-facing tests — require both RUN_OPENSHIFT_TESTS=1 and
#    RUN_OPENSHIFT_AI_PIPELINE_SMOKE=1.
#    These tests submit a real KFP Run to the DSPA server and wait for
#    terminal state.
# ===========================================================================


@_require_tabformer_pipeline
class TestTabFormerPipelineRuntime:
    """Level 7 — production pipeline runtime on OpenShift AI.

    Submits pragma_pretraining_pipeline to DSPA with bounded parameters:
        dataset_name = "ibm-tabformer"
        model_size   = "S"
        max_steps    = 2
        limit_rows   = 5
        batch_size   = 1
        device       = "cpu"
        run_name     = "level7-tabformer-smoke"

    The pipeline must reach terminal state SUCCEEDED within the configured
    timeout (default: 600 seconds, overridden by PRAGMA_TEST_TIMEOUT_SECONDS).

    Prerequisites:
        RUN_OPENSHIFT_TESTS=1
        RUN_OPENSHIFT_AI_PIPELINE_SMOKE=1
        PRAGMA_TEST_NAMESPACE=<namespace>
        DSPA API server reachable from test environment
        IBM TabFormer data available (S3 or DATA_URI env var)
    """

    def test_production_pipeline_reaches_succeeded(
        self,
        test_namespace: str,
        timeout_seconds: int,
    ) -> None:
        """Submit pragma_pretraining_pipeline and wait for SUCCEEDED state.

        Uses the DSPA submit path:
          1. Compile pipeline to YAML
          2. Upload to DSPA via KFP client
          3. Submit Run with bounded parameters
          4. Wait for terminal state with timeout
          5. Assert state == SUCCEEDED

        S3 credentials are injected via the namespace's OpenShift AI Connection
        (AWS_* env vars). This test does NOT manage credentials — they are
        expected to be in place as part of the platform substrate.

        If the pipeline fails (FAILED state), the test reports the Run ID
        so the failure can be investigated in the DSPA UI.

        Bounded parameters used:
            max_steps=2, limit_rows=5, batch_size=1, device='cpu', model_size='S'
        These bound wall-clock time to ≤5 minutes on CPU.
        """
        kfp = pytest.importorskip("kfp", reason="kfp not installed — pipeline submission skipped")

        from pipeline.pragma_pipeline import pragma_pretraining_pipeline  # noqa: PLC0415
        from tools.workbench._submit import (  # noqa: PLC0415
            get_dspa_endpoint,
            get_service_account_token,
            make_kfp_client,
            submit_pipeline_run,
            upload_pipeline,
            wait_for_run_terminal,
        )

        # --- Compile pipeline ---
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = pathlib.Path(tmpdir) / "pragma_tabformer_l7.yaml"
            kfp.compiler.Compiler().compile(
                pipeline_func=pragma_pretraining_pipeline,
                package_path=str(yaml_path),
            )

            assert yaml_path.exists(), (
                "Pipeline YAML compilation failed — nothing to submit."
            )

            # --- Construct KFP client ---
            endpoint = get_dspa_endpoint()
            token = get_service_account_token()
            client = make_kfp_client(host=endpoint, token=token, verify_ssl=False)

            # --- Upload pipeline to DSPA ---
            pipeline_id = upload_pipeline(
                client=client,
                yaml_path=str(yaml_path),
                pipeline_name="pragma-l7-tabformer-smoke",
            )

            # --- Submit Run with bounded parameters ---
            run_params = {
                "dataset_name":  _SMOKE_DATASET_NAME,
                "model_size":    _SMOKE_MODEL_SIZE,
                "max_steps":     _SMOKE_MAX_STEPS,
                "limit_rows":    _SMOKE_LIMIT_ROWS,
                "batch_size":    _SMOKE_BATCH_SIZE,
                "device":        _SMOKE_DEVICE,
                "run_name":      _SMOKE_RUN_NAME,
                "epochs":        1,
            }

            run_id = submit_pipeline_run(
                client=client,
                pipeline_id=pipeline_id,
                run_name=f"pragma-l7-tabformer-{_SMOKE_RUN_NAME}",
                params=run_params,
            )

            assert run_id, (
                "submit_pipeline_run() returned an empty run_id. "
                "Check DSPA connectivity and KFP client configuration."
            )

            # --- Wait for terminal state ---
            _effective_timeout = min(timeout_seconds, 600)  # cap at 10 minutes
            try:
                final_state = wait_for_run_terminal(
                    client=client,
                    run_id=run_id,
                    timeout_seconds=_effective_timeout,
                )
            except TimeoutError:
                pytest.fail(
                    f"Level 7 pipeline Run {run_id!r} did not reach terminal state "
                    f"within {_effective_timeout}s. "
                    f"Pipeline parameters: {run_params}. "
                    "Check DSPA UI for component-level failure logs. "
                    "Increase PRAGMA_TEST_TIMEOUT_SECONDS if the cluster is under load."
                )

            assert final_state == "SUCCEEDED", (
                f"Level 7 TabFormer pipeline Run {run_id!r} reached state "
                f"{final_state!r} (expected 'SUCCEEDED'). "
                f"Pipeline parameters used: {run_params}. "
                "Check DSPA UI -> Runs -> select Run ID for component logs. "
                "Common causes: S3 credentials missing (AWS_* env vars), "
                "IBM TabFormer data unavailable at manifest_uri, "
                "training image not pullable."
            )
