"""Level 3 — OpenShift AI KFP v2 pipeline smoke.

Purpose:
  Verify end-to-end PRAGMA pipeline execution via OpenShift AI Data Science
  Pipelines (DSPA / KFP v2). A minimal PRAGMA-S pipeline is compiled,
  uploaded to the KFP v2 API server, a Run is created with max_steps=1,
  and its completion and logs are verified.

Architecture boundary:
  This test suite targets the DSPA/KFP v2 pipeline path — the product
  pipeline runtime. It does NOT test:
    - batch/v1 Job execution (Level 3b — diagnostic only, separate gate)
    - PyTorchJob distributed training (Level 4 — future)
    - Tekton PipelineRun/TaskRun (not used in this environment)

  Level 3b PASS does NOT imply Level 3 PASS. They are independent gates.
  Level 3b proves the training image runs. Level 3 proves the product
  pipeline path (DSPA/KFP v2) runs.

Submit path (tools/openshift_ai/workbench/_submit.py — fully implemented):
  1. get_dspa_endpoint()         — resolve DSPA KFP API URL from env/namespace
  2. get_service_account_token() — read SA token from pod mount; never printed
  3. make_kfp_client()           — construct kfp.Client (kfp is optional dep)
  4. upload_pipeline()           — upload compiled YAML to DSPA
  5. submit_pipeline_run()       — create a KFP v2 Run
  6. get_run_status()            — normalise run state to uppercase string
  7. wait_for_run_terminal()     — poll until terminal state or TimeoutError

Smoke pipeline:
  pipeline/pragma_smoke_pipeline.py — pragma_smoke_training_pipeline.
  Single @dsl.component that runs fit_tokenizer + train_pragma --max-steps N
  directly in the component pod (no S3, no distributed training, no persistent
  volumes). Separate from production components (pipeline/components_pragma.py).

Safety rules:
  - Skip unless RUN_OPENSHIFT_PIPELINE_SMOKE=1.
  - Create only resources labelled pragma.redhat.com/test-run=true,
    pragma.redhat.com/test-id=<test_id>.
  - Cleanup deletes only labelled resources.
  - Never modify Argo CD-managed resources.
  - Never delete secrets, serviceaccounts, or namespaces.
  - SA token is NEVER printed, logged, or included in repr.
  - oc exec is for diagnostics only — never the pipeline submission path.

Prerequisites (when enabled):
  - RUN_OPENSHIFT_TESTS=1
  - RUN_OPENSHIFT_PIPELINE_SMOKE=1
  - PRAGMA_TEST_NAMESPACE=<namespace>
  - DSPA pods running in namespace (verified by Level 1)
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest.mock as mock

import pytest

from tools.openshift_ai.workbench._submit import (
    DSPAConfig,
    get_dspa_endpoint,
    get_run_status,  # noqa: F401 — imported for contract completeness
    get_service_account_token,
    make_kfp_client,
    submit_pipeline_run,  # noqa: F401 — imported for contract completeness
    upload_pipeline,  # noqa: F401 — imported for contract completeness
    wait_for_run_terminal,
)

# ---------------------------------------------------------------------------
# Suite-level skip guard (conftest handles RUN_OPENSHIFT_TESTS=1)
# ---------------------------------------------------------------------------

_PIPELINE_SMOKE_ENABLED = os.environ.get("RUN_OPENSHIFT_PIPELINE_SMOKE") == "1"

_require_smoke = pytest.mark.skipif(
    not _PIPELINE_SMOKE_ENABLED,
    reason=(
        "OpenShift AI KFP v2 pipeline smoke tests are opt-in. "
        "Set RUN_OPENSHIFT_PIPELINE_SMOKE=1 to enable. "
        "Also requires PRAGMA_TEST_NAMESPACE=<namespace> and RUN_OPENSHIFT_TESTS=1. "
        "These tests create short-lived KFP Run resources in the cluster."
    ),
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DSPA_SERVICE = "ds-pipeline-pipelines-definition"
_KFP_PORT = 8888


# ===========================================================================
# 1. TestKFPPipelineSmokePrereqs
#    Local tests — no cluster access needed.
#    Run whenever RUN_OPENSHIFT_TESTS=1 (conftest skip guard).
#    Validate the submit module contract before any cluster connection.
# ===========================================================================


class TestKFPPipelineSmokePrereqs:
    """Local contract tests for tools/openshift_ai/workbench/_submit.py.

    5 tests — no cluster access needed.
    Run whenever RUN_OPENSHIFT_TESTS=1, regardless of RUN_OPENSHIFT_PIPELINE_SMOKE.

    Purpose: confirm the DSPA submit module is correctly implemented and
    safe before any cluster-facing call is attempted.
    """

    def test_submit_module_importable_without_kfp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tools.openshift_ai.workbench._submit must import cleanly even when kfp is absent.

        kfp is an optional dependency. The submit module must not import
        kfp at module level — only lazily inside make_kfp_client(). This
        confirms the module loads in any environment (local dev, CI, training
        container, workbench without kfp extras).
        """
        # Block kfp from sys.modules to simulate it being absent.
        monkeypatch.setitem(sys.modules, "kfp", None)  # type: ignore[arg-type]

        try:
            submit_mod = importlib.import_module("tools.openshift_ai.workbench._submit")
            assert submit_mod is not None, (
                "tools.openshift_ai.workbench._submit must be importable without kfp. "
                "kfp must be imported lazily inside make_kfp_client(), not at module level."
            )
        except ImportError as exc:
            pytest.fail(
                f"tools.openshift_ai.workbench._submit raised ImportError without kfp: {exc}. "
                "kfp must be imported lazily inside make_kfp_client()."
            )

    def test_get_dspa_endpoint_uses_pragma_test_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_dspa_endpoint() constructs the KFP API URL from PRAGMA_TEST_NAMESPACE.

        PRAGMA_TEST_NAMESPACE drives both the conftest test_namespace fixture and
        the KFP endpoint construction. This integration is confirmed here before
        any real network call. The endpoint must use https:// and port 8888.

        Expected:
          https://ds-pipeline-pipelines-definition.<ns>.svc.cluster.local:8888
        """
        namespace = "pragma-encoder"
        monkeypatch.delenv("PRAGMA_DSPA_ENDPOINT", raising=False)
        monkeypatch.delenv("PRAGMA_DSPA_NAMESPACE", raising=False)
        monkeypatch.setenv("PRAGMA_TEST_NAMESPACE", namespace)

        result = get_dspa_endpoint()

        expected = f"https://{_DSPA_SERVICE}.{namespace}.svc.cluster.local:{_KFP_PORT}"
        assert result == expected, (
            f"Expected PRAGMA_TEST_NAMESPACE={namespace!r} to produce {expected!r}. "
            f"Got: {result!r}. "
            "Check get_dspa_endpoint() resolution order in tools/openshift_ai/workbench/_submit.py."
        )
        assert result.startswith("https://"), (
            "Endpoint must use https:// — DSPA port 8888 uses TLS (self-signed cert)."
        )
        assert f":{_KFP_PORT}" in result, f"Endpoint must include port {_KFP_PORT}."

    def test_dspa_config_token_not_in_repr(self) -> None:
        """DSPAConfig repr must not expose the SA token value.

        The SA token is a bearer credential and must never appear in any
        string representation. This is a safety property verified before
        the token is passed to any cluster-facing call.
        """
        sentinel = "LEVEL3_PREREQ_SENTINEL_TOKEN_MUST_NOT_APPEAR"

        config = DSPAConfig(
            endpoint=f"https://{_DSPA_SERVICE}.pragma-encoder.svc.cluster.local:{_KFP_PORT}",
            token=sentinel,
        )

        assert sentinel not in repr(config), (
            "SA token must NOT appear in repr(DSPAConfig). "
            "Token is a bearer credential — never expose in repr."
        )
        assert sentinel not in str(config), (
            "SA token must NOT appear in str(DSPAConfig). "
            "Token is a bearer credential — never expose in str."
        )
        assert config.has_token is True, (
            "DSPAConfig.has_token must be True when a token is provided."
        )

    def test_wait_for_run_terminal_raises_timeout_error(self) -> None:
        """wait_for_run_terminal() raises TimeoutError when run stays non-terminal.

        The TimeoutError message must include the run_id and last observed state.
        This confirms the polling loop timeout works correctly before any cluster call.
        """
        run = mock.MagicMock()
        run.state = "RUNNING"
        client = mock.MagicMock()
        client.get_run.return_value = run

        with pytest.raises(TimeoutError) as exc_info:
            wait_for_run_terminal(
                client=client,
                run_id="level3-prereq-stuck-run",
                timeout=0,
                poll_interval=0,
            )

        msg = str(exc_info.value)
        assert "level3-prereq-stuck-run" in msg, (
            f"TimeoutError must include the run_id. Got: {msg!r}"
        )
        assert "RUNNING" in msg, (
            f"TimeoutError must include the last observed state. Got: {msg!r}"
        )

    def test_smoke_pipeline_module_contract(self) -> None:
        """pragma_smoke_training_pipeline must exist in pragma_smoke_pipeline, NOT pragma_pipeline.

        Two invariants must hold simultaneously:

        1. pipeline.pragma_smoke_pipeline defines pragma_smoke_training_pipeline.
           This is the module that supplies the KFP v2 component for Level 3 smoke.
           If this fails: the smoke component was removed or renamed — restore it.

        2. pipeline.pragma_pipeline does NOT define pragma_smoke_training_pipeline.
           Production pipeline must stay clean of smoke/test components.
           If this fails: the smoke component was accidentally added to production.
           Move it back to pragma_smoke_pipeline.py.
        """
        # Invariant 1 — smoke module has the pipeline function.
        try:
            import pipeline.pragma_smoke_pipeline as smp
            has_smoke_in_smoke_module = hasattr(smp, "pragma_smoke_training_pipeline")
        except ImportError:
            has_smoke_in_smoke_module = False

        assert has_smoke_in_smoke_module, (
            "pipeline.pragma_smoke_pipeline must define pragma_smoke_training_pipeline. "
            "This is required for the Level 3 DSPA/KFP v2 smoke test. "
            "Check pipeline/pragma_smoke_pipeline.py."
        )

        # Invariant 2 — production pipeline does NOT have the smoke pipeline.
        try:
            import pipeline.pragma_pipeline as pp
            has_smoke_in_production = hasattr(pp, "pragma_smoke_training_pipeline")
        except ImportError:
            has_smoke_in_production = False

        assert not has_smoke_in_production, (
            "pragma_smoke_training_pipeline must NOT be defined in pipeline/pragma_pipeline.py. "
            "The production pipeline must remain clean of smoke/test components. "
            "Keep pragma_smoke_training_pipeline in pipeline/pragma_smoke_pipeline.py only."
        )


# ===========================================================================
# 2. TestKFPDSPAConnectivity
#    Opt-in cluster connectivity tests. Require RUN_OPENSHIFT_PIPELINE_SMOKE=1.
#    Read-only — no resources created.
# ===========================================================================


class TestKFPDSPAConnectivity:
    """Level 3: DSPA/KFP v2 connectivity checks (opt-in, read-only).

    3 tests — confirm the DSPA endpoint is reachable and the kfp.Client
    can be constructed and used before attempting the full pipeline smoke.

    Skip unless RUN_OPENSHIFT_PIPELINE_SMOKE=1 (via _require_smoke).
    All tests are read-only — no pipeline runs or cluster resources created.
    """

    @_require_smoke
    def test_dspa_endpoint_resolves_in_cluster(
        self, test_namespace: str
    ) -> None:
        """get_dspa_endpoint() builds a valid URL from the test namespace.

        Confirms endpoint construction is correct for the active namespace
        before any network call is attempted.
        """
        endpoint = get_dspa_endpoint(namespace=test_namespace)

        assert endpoint.startswith("https://"), (
            f"Endpoint must start with https://. Got: {endpoint!r}"
        )
        assert _DSPA_SERVICE in endpoint, (
            f"Endpoint must reference {_DSPA_SERVICE!r}. Got: {endpoint!r}"
        )
        assert test_namespace in endpoint, (
            f"Endpoint must include namespace {test_namespace!r}. Got: {endpoint!r}"
        )
        assert f":{_KFP_PORT}" in endpoint, (
            f"Endpoint must include port {_KFP_PORT}. Got: {endpoint!r}"
        )

        print(f"\n[Level 3] DSPA endpoint: {endpoint}")

    @_require_smoke
    def test_kfp_client_can_be_constructed(self, test_namespace: str) -> None:
        """make_kfp_client() returns a non-None kfp.Client for the DSPA endpoint.

        Skips if kfp is not installed — kfp is an optional dependency that
        must be available in the workbench image but may be absent locally.

        kfp.Client construction does not make a network call at init time.
        SA token is None outside a pod — handled correctly by make_kfp_client().
        """
        if importlib.util.find_spec("kfp") is None:
            pytest.skip(
                "kfp is not installed. Install with: pip install kfp. "
                "kfp is available in the workbench image."
            )

        endpoint = get_dspa_endpoint(namespace=test_namespace)
        token = get_service_account_token()  # None outside a pod — expected

        client = make_kfp_client(endpoint=endpoint, token=token)

        assert client is not None, "make_kfp_client() must return a non-None kfp.Client."

        print(f"\n[Level 3] kfp.Client constructed. endpoint={endpoint}")
        print(f"[Level 3] SA token present in pod: {token is not None}")

    @_require_smoke
    def test_kfp_client_can_list_pipelines(self, test_namespace: str) -> None:
        """kfp.Client.list_pipelines() reaches the DSPA KFP v2 API server.

        Real network call to the DSPA endpoint. Skips if kfp is not installed
        or the DSPA API is not reachable from this network environment (no VPN).

        Requires DSPA pods running in test_namespace (Level 1 verifies this).
        An empty pipeline list is acceptable — the call must complete without error.
        """
        if importlib.util.find_spec("kfp") is None:
            pytest.skip("kfp not installed — required for DSPA API connectivity check.")

        endpoint = get_dspa_endpoint(namespace=test_namespace)
        token = get_service_account_token()
        client = make_kfp_client(endpoint=endpoint, token=token)

        try:
            response = client.list_pipelines(page_size=1)
            count = len(getattr(response, "pipelines", None) or [])
            print(f"\n[Level 3] list_pipelines() succeeded. Pipelines in DSPA: {count}")
        except Exception as exc:
            err = str(exc)
            conn_keywords = ("connection", "refused", "timeout", "unreachable", "name or service")
            if any(kw in err.lower() for kw in conn_keywords):
                pytest.skip(
                    f"DSPA endpoint {endpoint!r} not reachable from this environment. "
                    "In-cluster network access is required (VPN or workbench pod). "
                    f"Connection error: {err}"
                )
            raise


# ===========================================================================
# 3. TestKFPPipelineSmoke
#    Full DSPA/KFP v2 pipeline smoke.
#    Requires RUN_OPENSHIFT_PIPELINE_SMOKE=1 and kfp installed.
# ===========================================================================


class TestKFPPipelineSmoke:
    """Level 3: OpenShift AI KFP v2 pipeline runtime smoke.

    Compiles pragma_smoke_training_pipeline, uploads to the DSPA KFP v2 API,
    creates a Run with max_steps=1, polls until SUCCEEDED, and asserts log markers.

    Requires RUN_OPENSHIFT_PIPELINE_SMOKE=1 and kfp installed.
    Skips if kfp is not installed (optional dependency).
    Cleanup via cleanup_labelled_resources fixture (label-scoped, automatic).
    """

    @_require_smoke
    def test_kfp_pipeline_smoke_run(
        self,
        test_namespace: str,
        runtime_namespace: str,
        test_id: str,
        test_labels: dict[str, str],
        timeout_seconds: int,
        cleanup_labelled_resources: None,
    ) -> None:
        """Compile, upload, and run pragma_smoke_training_pipeline via DSPA KFP v2.

        Steps:
          1. Skip if kfp is not installed.
          2. Compile pragma_smoke_training_pipeline → temp YAML.
          3. Resolve DSPA endpoint from runtime_namespace.
          4. Read SA token (never printed, never in repr).
          5. Construct kfp.Client.
          6. Upload compiled YAML as f"pragma-smoke-{test_id}".
          7. Submit KFP v2 Run (max_steps=1, experiment="pragma-smoke").
          8. Poll until terminal state (timeout=timeout_seconds, poll_interval=15).
          9. Assert SUCCEEDED. Collect pod logs via oc; assert log markers if available.
          Cleanup via cleanup_labelled_resources fixture (automatic).
        """
        # ------------------------------------------------------------------
        # Step 1 — skip if kfp is not installed.
        # ------------------------------------------------------------------
        if importlib.util.find_spec("kfp") is None:
            pytest.skip(
                "kfp is not installed. Install with: pip install kfp. "
                "kfp is available in the workbench image."
            )

        import kfp  # noqa: PLC0415 — guarded by find_spec above

        from pipeline.pragma_smoke_pipeline import (  # noqa: PLC0415
            pragma_smoke_training_pipeline,
        )

        # ------------------------------------------------------------------
        # Step 2 — compile pragma_smoke_training_pipeline to a KFP v2 YAML.
        # ------------------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = pathlib.Path(tmp_dir) / f"pragma-smoke-{test_id}.yaml"
            kfp.compiler.Compiler().compile(
                pipeline_func=pragma_smoke_training_pipeline,
                package_path=str(yaml_path),
            )
            assert yaml_path.exists(), (
                "kfp.compiler.Compiler().compile() did not produce a YAML file. "
                "Check pragma_smoke_training_pipeline definition."
            )
            print(f"\n[Level 3] Compiled pipeline YAML: {yaml_path} ({yaml_path.stat().st_size} bytes)")

            # ---------------------------------------------------------------
            # Step 3 — resolve DSPA endpoint.
            # ---------------------------------------------------------------
            endpoint = get_dspa_endpoint(namespace=runtime_namespace)
            print(f"[Level 3] DSPA endpoint: {endpoint}")

            # ---------------------------------------------------------------
            # Step 4 — read SA token (never printed, never in repr).
            # ---------------------------------------------------------------
            token = get_service_account_token()
            print(f"[Level 3] SA token present in pod: {token is not None}")

            # ---------------------------------------------------------------
            # Step 5 — construct kfp.Client.
            # ---------------------------------------------------------------
            client = make_kfp_client(endpoint=endpoint, token=token)
            assert client is not None, "make_kfp_client() must return a non-None kfp.Client."

            # ---------------------------------------------------------------
            # Step 6 — upload compiled pipeline YAML to DSPA.
            # ---------------------------------------------------------------
            pipeline_name = f"pragma-smoke-{test_id}"
            pipeline_id = upload_pipeline(
                client=client,
                yaml_path=yaml_path,
                pipeline_name=pipeline_name,
            )
            assert pipeline_id, (
                f"upload_pipeline() returned empty pipeline_id for {pipeline_name!r}. "
                "Check DSPA connectivity and pipeline YAML validity."
            )
            print(f"[Level 3] Uploaded pipeline: name={pipeline_name!r}  id={pipeline_id!r}")

            # ---------------------------------------------------------------
            # Step 7 — submit KFP v2 Run.
            # ---------------------------------------------------------------
            run_name = f"pragma-smoke-run-{test_id}"
            run_id = submit_pipeline_run(
                client=client,
                pipeline_id=pipeline_id,
                run_name=run_name,
                arguments={"max_steps": 1},
                experiment_name="pragma-smoke",
            )
            assert run_id, (
                f"submit_pipeline_run() returned empty run_id for {run_name!r}. "
                "Check DSPA connectivity and pipeline upload."
            )
            print(f"[Level 3] Submitted run: name={run_name!r}  id={run_id!r}")
            print(f"[Level 3] Polling until terminal state (timeout={timeout_seconds}s) ...")

            # ---------------------------------------------------------------
            # Step 8 — poll until terminal state.
            # ---------------------------------------------------------------
            try:
                final_state = wait_for_run_terminal(
                    client=client,
                    run_id=run_id,
                    timeout=timeout_seconds,
                    poll_interval=15,
                )
            except TimeoutError as exc:
                pytest.fail(
                    f"KFP Run {run_id!r} did not reach a terminal state within "
                    f"{timeout_seconds}s: {exc}"
                )

            print(f"[Level 3] Run terminal state: {final_state}")

            # ---------------------------------------------------------------
            # Step 9 — assert SUCCEEDED + log markers.
            # Collect pod logs via oc (best-effort; skip marker assertion if unavailable).
            # ---------------------------------------------------------------
            assert final_state == "SUCCEEDED", (
                f"KFP Run {run_id!r} did not SUCCEED. "
                f"Final state: {final_state!r}. "
                f"Check the DSPA KFP v2 UI for pod logs: run_name={run_name!r}."
            )

            # Attempt pod log collection via oc (best-effort — requires in-cluster access).
            log_result = subprocess.run(
                [
                    "oc", "logs",
                    "-n", runtime_namespace,
                    "-l", f"pipeline/runid={run_id}",
                    "--tail", "200",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            all_logs = log_result.stdout.strip()

            if all_logs:
                print("[Level 3] Log excerpt (last ~30 lines):")
                for line in all_logs.splitlines()[-30:]:
                    print(f"  {line}")

                assert "PRAGMA-S" in all_logs, (
                    "Pod logs must contain 'PRAGMA-S' — the model variant confirmation. "
                    f"Run: {run_id!r}  state: {final_state}. "
                    f"Full logs:\n{all_logs}"
                )
                assert "Reached --max-steps" in all_logs, (
                    "Pod logs must contain 'Reached --max-steps' — the early-stop marker. "
                    f"Run: {run_id!r}  state: {final_state}. "
                    f"Full logs:\n{all_logs}"
                )
                assert "PRAGMA smoke training completed" in all_logs, (
                    "Pod logs must contain 'PRAGMA smoke training completed'. "
                    f"Run: {run_id!r}  state: {final_state}. "
                    f"Full logs:\n{all_logs}"
                )
                print("[Level 3] Log markers confirmed: 'PRAGMA-S' ✓  'Reached --max-steps' ✓  'PRAGMA smoke training completed' ✓")
            else:
                # oc logs may not reach KFP pods from outside the cluster.
                # SUCCEEDED state is the definitive pass criterion — log markers
                # are supplementary validation. Warn but do not fail.
                print(
                    f"[Level 3] WARNING: pod logs not available via oc "
                    f"(label: pipeline/runid={run_id}). "
                    "Run SUCCEEDED — log markers not verified. "
                    "Check the DSPA KFP v2 UI for pod-level logs."
                )

        print("\n[Level 3] === PASSED: PRAGMA KFP v2 pipeline smoke complete ===")
        print(f"  Run:      {run_name}  →  {final_state}")
        print(f"  Pipeline: {pipeline_name}")
        print(f"  Endpoint: {endpoint}")
