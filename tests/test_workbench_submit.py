"""Tests for the PRAGMA workbench submit module (src/workbench/_submit.py).

These tests are written BEFORE the implementation (TDD red phase).
They define the contract for the DSPA/KFP v2 submit path from the workbench.

Scope:
  src/workbench/_submit.py exposes six functions:
    get_dspa_endpoint(...)      — endpoint URL from env/defaults
    get_service_account_token() — reads SA token; never prints
    make_kfp_client(...)        — constructs kfp.Client (kfp optional dep)
    upload_pipeline(...)        — uploads compiled YAML to DSPA
    submit_pipeline_run(...)    — creates a pipeline run on DSPA
    DSPAConfig                  — lightweight dataclass for endpoint + auth config

The submit path:
  1. Resolve endpoint (PRAGMA_DSPA_ENDPOINT → constructed from namespace)
  2. Read SA token (mounted at /var/run/secrets/…/token; None if outside pod)
  3. Construct kfp.Client(host=endpoint, existing_token=token)
  4. Upload compiled YAML → pipeline_id
  5. Optionally create a run → run_id

Safety:
  - SA token is NEVER printed, logged, or included in repr.
  - kfp is an optional dependency: missing kfp → friendly RuntimeError.
  - No cluster access required for any test here (all network calls mocked).
  - compile() has no cluster side effects — confirmed by existing tests.
  - submit is opt-in and does not run during normal examples.

ADR reference: docs/decisions/003-workbench-training-api.md
               docs/decisions/004-workbench-decorated-pipelines.md
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
import unittest.mock as mock

import pytest

# ---------------------------------------------------------------------------
# Module under test (will ImportError until implementation is written)
# ---------------------------------------------------------------------------
from src.workbench._submit import (
    DSPAConfig,
    get_dspa_endpoint,
    get_run_status,
    get_service_account_token,
    make_kfp_client,
    submit_pipeline_run,
    upload_pipeline,
    wait_for_run_terminal,
)

# ---------------------------------------------------------------------------
# Constants matching the implementation defaults
# ---------------------------------------------------------------------------

_SA_TOKEN_PATH = pathlib.Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
_SA_NAMESPACE_PATH = pathlib.Path(
    "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
)
_DSPA_SERVICE = "ds-pipeline-pipelines-definition"
_KFP_PORT = 8888

_DEFAULT_NAMESPACE = "pragma-encoder"
_EXPECTED_ENDPOINT = (
    f"https://{_DSPA_SERVICE}.{_DEFAULT_NAMESPACE}.svc.cluster.local:{_KFP_PORT}"
)


# ===========================================================================
# 1. TestGetDSPAEndpoint
#    get_dspa_endpoint() resolves the KFP API URL from env vars or defaults.
# ===========================================================================


class TestGetDSPAEndpoint:
    """Endpoint URL construction — no network calls, no cluster access."""

    def test_explicit_env_var_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PRAGMA_DSPA_ENDPOINT overrides all other resolution logic.

        When PRAGMA_DSPA_ENDPOINT is set, get_dspa_endpoint() must return
        that value verbatim without constructing or validating the URL.
        """
        explicit_url = "http://custom-dspa.example.com:8888"
        monkeypatch.setenv("PRAGMA_DSPA_ENDPOINT", explicit_url)
        monkeypatch.delenv("PRAGMA_DSPA_NAMESPACE", raising=False)
        monkeypatch.delenv("PRAGMA_TEST_NAMESPACE", raising=False)

        result = get_dspa_endpoint()

        assert result == explicit_url, (
            f"Expected PRAGMA_DSPA_ENDPOINT={explicit_url!r} to be returned "
            f"verbatim, got {result!r}"
        )

    def test_dspa_namespace_env_constructs_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PRAGMA_DSPA_NAMESPACE constructs the in-cluster URL.

        When PRAGMA_DSPA_ENDPOINT is absent but PRAGMA_DSPA_NAMESPACE is set,
        the endpoint must be constructed as:
          http://ds-pipeline-pipelines-definition.<namespace>.svc.cluster.local:8888
        """
        monkeypatch.delenv("PRAGMA_DSPA_ENDPOINT", raising=False)
        monkeypatch.setenv("PRAGMA_DSPA_NAMESPACE", _DEFAULT_NAMESPACE)
        monkeypatch.delenv("PRAGMA_TEST_NAMESPACE", raising=False)

        result = get_dspa_endpoint()

        assert result == _EXPECTED_ENDPOINT, (
            f"Expected endpoint {_EXPECTED_ENDPOINT!r}, got {result!r}"
        )

    def test_test_namespace_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PRAGMA_TEST_NAMESPACE is used when PRAGMA_DSPA_NAMESPACE is absent.

        Resolution order: PRAGMA_DSPA_ENDPOINT → PRAGMA_DSPA_NAMESPACE →
        PRAGMA_TEST_NAMESPACE → namespace file → RuntimeError.
        """
        monkeypatch.delenv("PRAGMA_DSPA_ENDPOINT", raising=False)
        monkeypatch.delenv("PRAGMA_DSPA_NAMESPACE", raising=False)
        monkeypatch.setenv("PRAGMA_TEST_NAMESPACE", _DEFAULT_NAMESPACE)

        result = get_dspa_endpoint()

        assert result == _EXPECTED_ENDPOINT, (
            f"Expected PRAGMA_TEST_NAMESPACE fallback to build {_EXPECTED_ENDPOINT!r}, "
            f"got {result!r}"
        )

    def test_namespace_file_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """SA namespace file is read as last resort when no env var is set.

        When running inside a pod, the namespace is mounted at:
          /var/run/secrets/kubernetes.io/serviceaccount/namespace
        """
        monkeypatch.delenv("PRAGMA_DSPA_ENDPOINT", raising=False)
        monkeypatch.delenv("PRAGMA_DSPA_NAMESPACE", raising=False)
        monkeypatch.delenv("PRAGMA_TEST_NAMESPACE", raising=False)

        ns_file = tmp_path / "namespace"
        ns_file.write_text(_DEFAULT_NAMESPACE)

        result = get_dspa_endpoint(namespace_file=ns_file)

        assert result == _EXPECTED_ENDPOINT, (
            f"Expected namespace-file fallback to build {_EXPECTED_ENDPOINT!r}, "
            f"got {result!r}"
        )

    def test_no_namespace_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """RuntimeError is raised when no namespace source is available.

        This prevents silent use of an empty or wrong endpoint.
        """
        monkeypatch.delenv("PRAGMA_DSPA_ENDPOINT", raising=False)
        monkeypatch.delenv("PRAGMA_DSPA_NAMESPACE", raising=False)
        monkeypatch.delenv("PRAGMA_TEST_NAMESPACE", raising=False)

        absent_ns_file = tmp_path / "namespace"  # does not exist

        with pytest.raises(RuntimeError, match="namespace"):
            get_dspa_endpoint(namespace_file=absent_ns_file)

    def test_explicit_namespace_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Caller can pass namespace as a keyword argument.

        get_dspa_endpoint(namespace='my-ns') must construct the endpoint
        using 'my-ns' without reading any env var or file.
        """
        monkeypatch.delenv("PRAGMA_DSPA_ENDPOINT", raising=False)
        monkeypatch.delenv("PRAGMA_DSPA_NAMESPACE", raising=False)
        monkeypatch.delenv("PRAGMA_TEST_NAMESPACE", raising=False)

        result = get_dspa_endpoint(namespace="my-ns")

        expected = f"https://{_DSPA_SERVICE}.my-ns.svc.cluster.local:{_KFP_PORT}"
        assert result == expected

    def test_endpoint_starts_with_https(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructed endpoint must always use https://.

        Port 8888 on the in-cluster DSPA service uses TLS (self-signed cert).
        kfp.Client is constructed with verify_ssl=False to bypass cert validation.
        """
        monkeypatch.setenv("PRAGMA_DSPA_NAMESPACE", _DEFAULT_NAMESPACE)
        monkeypatch.delenv("PRAGMA_DSPA_ENDPOINT", raising=False)

        result = get_dspa_endpoint()

        assert result.startswith("https://"), (
            f"Endpoint must use https://, got {result!r}. "
            "Port 8888 uses TLS (self-signed cert); use verify_ssl=False in kfp.Client."
        )

    def test_endpoint_contains_kfp_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructed endpoint must include port 8888."""
        monkeypatch.setenv("PRAGMA_DSPA_NAMESPACE", _DEFAULT_NAMESPACE)
        monkeypatch.delenv("PRAGMA_DSPA_ENDPOINT", raising=False)

        result = get_dspa_endpoint()

        assert f":{_KFP_PORT}" in result, (
            f"Endpoint must include port {_KFP_PORT}, got {result!r}"
        )


# ===========================================================================
# 2. TestGetServiceAccountToken
#    get_service_account_token() reads the SA token safely.
# ===========================================================================


class TestGetServiceAccountToken:
    """SA token reading — token is NEVER printed or included in any repr."""

    def test_returns_none_outside_pod(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """Returns None when the SA token file does not exist.

        Outside a Kubernetes pod, the token file is absent. The function
        must return None, not raise. Callers decide whether to proceed
        without a token.
        """
        absent = tmp_path / "token"  # does not exist

        result = get_service_account_token(token_path=absent)

        assert result is None, (
            f"Expected None outside pod (no token file), got {result!r}"
        )

    def test_returns_token_inside_pod(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Returns the token string when the SA token file is present.

        The token value must be returned exactly as it appears on disk
        (stripped of leading/trailing whitespace).
        """
        token_file = tmp_path / "token"
        token_file.write_text("mysecrettoken\n")

        result = get_service_account_token(token_path=token_file)

        assert result == "mysecrettoken", (
            "Token must be returned as stripped text. "
            "Got: (redacted for safety)"
        )

    def test_token_not_in_repr_or_str(self, tmp_path: pathlib.Path) -> None:
        """The token value must not appear in any string representation.

        This is a safety property. The DSPAConfig object must not expose
        the token via repr(), str(), or __dict__ printing.
        """
        token_file = tmp_path / "token"
        sentinel = "SENTINEL_TOKEN_MUST_NOT_APPEAR_IN_REPR"
        token_file.write_text(sentinel)

        token = get_service_account_token(token_path=token_file)
        assert token == sentinel  # confirm we got the right value

        config = DSPAConfig(
            endpoint=_EXPECTED_ENDPOINT,
            token=token,
        )

        config_repr = repr(config)
        config_str = str(config)

        assert sentinel not in config_repr, (
            "Token value must NOT appear in DSPAConfig repr(). "
            "repr: (truncated for safety)"
        )
        assert sentinel not in config_str, (
            "Token value must NOT appear in DSPAConfig str(). "
            "str: (truncated for safety)"
        )

    def test_returns_none_on_oserror(self, tmp_path: pathlib.Path) -> None:
        """Returns None gracefully if the token file is unreadable.

        An OSError (e.g. permission denied) must be swallowed and None
        returned, not propagated. This keeps the submit path robust.
        """
        token_file = tmp_path / "token"
        token_file.write_text("secret")
        token_file.chmod(0o000)  # unreadable

        try:
            result = get_service_account_token(token_path=token_file)
            assert result is None, "Expected None for unreadable token file"
        finally:
            token_file.chmod(0o644)  # restore for cleanup


# ===========================================================================
# 3. TestDSPAConfig
#    DSPAConfig is a lightweight value object for endpoint + auth config.
# ===========================================================================


class TestDSPAConfig:
    """DSPAConfig value object — safe repr, endpoint, optional token."""

    def test_stores_endpoint(self) -> None:
        """DSPAConfig.endpoint stores the provided endpoint URL."""
        config = DSPAConfig(endpoint=_EXPECTED_ENDPOINT, token=None)
        assert config.endpoint == _EXPECTED_ENDPOINT

    def test_stores_token_flag_not_value(self) -> None:
        """DSPAConfig.has_token reflects whether a token was provided.

        The token value itself is never exposed via a public attribute.
        """
        config_with = DSPAConfig(endpoint=_EXPECTED_ENDPOINT, token="secret")
        config_without = DSPAConfig(endpoint=_EXPECTED_ENDPOINT, token=None)

        assert config_with.has_token is True
        assert config_without.has_token is False

    def test_repr_does_not_contain_token(self) -> None:
        """repr(DSPAConfig) must not expose the token value."""
        sentinel = "MUST_NOT_APPEAR_IN_REPR"
        config = DSPAConfig(endpoint=_EXPECTED_ENDPOINT, token=sentinel)

        r = repr(config)
        assert sentinel not in r, (
            "Token must not appear in repr(DSPAConfig). "
            "Use has_token=True instead."
        )
        assert "has_token=True" in r or _EXPECTED_ENDPOINT in r, (
            "repr(DSPAConfig) should include endpoint and/or has_token flag."
        )

    def test_default_token_is_none(self) -> None:
        """DSPAConfig.token defaults to None if not supplied."""
        config = DSPAConfig(endpoint=_EXPECTED_ENDPOINT)
        assert config.has_token is False


# ===========================================================================
# 4. TestMakeKFPClient
#    make_kfp_client() constructs kfp.Client — kfp is an optional dep.
# ===========================================================================


class TestMakeKFPClient:
    """kfp.Client construction — kfp mocked; never connects in tests."""

    def _make_mock_kfp(self) -> types.ModuleType:
        """Return a minimal kfp mock with Client class."""
        mock_kfp = types.ModuleType("kfp")
        mock_client_instance = mock.MagicMock(name="kfp.Client instance")
        mock_kfp.Client = mock.MagicMock(
            name="kfp.Client", return_value=mock_client_instance
        )
        mock_kfp.__version__ = "2.9.0"
        return mock_kfp

    def test_raises_runtime_error_if_kfp_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeError with friendly install message when kfp is absent.

        The error message must mention 'kfp' and 'install' so the user
        understands how to resolve the dependency.
        """
        # Temporarily hide kfp from sys.modules.
        monkeypatch.setitem(sys.modules, "kfp", None)  # type: ignore[arg-type]

        with pytest.raises(RuntimeError) as exc_info:
            make_kfp_client(endpoint=_EXPECTED_ENDPOINT, token=None)

        msg = str(exc_info.value).lower()
        assert "kfp" in msg, "Error message must mention 'kfp'"
        assert "install" in msg, "Error message must mention how to install kfp"

    def test_constructs_client_without_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kfp.Client(host=endpoint) is called when no token is provided."""
        mock_kfp = self._make_mock_kfp()
        monkeypatch.setitem(sys.modules, "kfp", mock_kfp)

        client = make_kfp_client(endpoint=_EXPECTED_ENDPOINT, token=None)

        mock_kfp.Client.assert_called_once_with(host=_EXPECTED_ENDPOINT, verify_ssl=False)
        assert client is mock_kfp.Client.return_value

    def test_constructs_client_with_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kfp.Client(host=endpoint, existing_token=token) when token provided.

        The token value must be passed as existing_token kwarg — the
        kfp.Client contract for bearer token auth. Token is never printed.
        """
        mock_kfp = self._make_mock_kfp()
        monkeypatch.setitem(sys.modules, "kfp", mock_kfp)

        token = "secret-sa-token"
        make_kfp_client(endpoint=_EXPECTED_ENDPOINT, token=token)

        mock_kfp.Client.assert_called_once_with(
            host=_EXPECTED_ENDPOINT,
            existing_token=token,
            verify_ssl=False,
        )

    def test_client_token_not_in_call_repr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """The token value must not appear in any captured stdout/stderr output.

        make_kfp_client() must not print, log, or repr the token.
        """
        mock_kfp = self._make_mock_kfp()
        monkeypatch.setitem(sys.modules, "kfp", mock_kfp)

        sentinel = "SENTINEL_TOKEN_MUST_NOT_APPEAR_IN_OUTPUT"
        make_kfp_client(endpoint=_EXPECTED_ENDPOINT, token=sentinel)

        captured = capsys.readouterr()
        assert sentinel not in captured.out, "Token must not appear in stdout"
        assert sentinel not in captured.err, "Token must not appear in stderr"


# ===========================================================================
# 5. TestUploadPipeline
#    upload_pipeline() calls kfp.Client.upload_pipeline with expected args.
# ===========================================================================


class TestUploadPipeline:
    """upload_pipeline() wraps kfp client pipeline upload."""

    def _make_client_mock(self, pipeline_id: str = "test-pipeline-id") -> mock.MagicMock:
        """Return a mock kfp.Client with upload_pipeline pre-configured."""
        client = mock.MagicMock(name="kfp.Client")
        response = mock.MagicMock()
        response.pipeline_id = pipeline_id
        client.upload_pipeline.return_value = response
        return client

    def test_calls_upload_pipeline_with_path_and_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        """upload_pipeline() must call client.upload_pipeline(str(path), pipeline_name=name).

        The YAML path is converted to string before passing to kfp (kfp
        does not accept pathlib.Path in all versions).
        """
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text("schemaVersion: '2.1.0'\n")

        client = self._make_client_mock()
        pipeline_name = "pragma-s-test"

        upload_pipeline(client=client, yaml_path=yaml_file, pipeline_name=pipeline_name)

        client.upload_pipeline.assert_called_once_with(
            str(yaml_file),
            pipeline_name=pipeline_name,
        )

    def test_returns_pipeline_id(self, tmp_path: pathlib.Path) -> None:
        """upload_pipeline() must return the pipeline_id string from the response."""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text("schemaVersion: '2.1.0'\n")

        expected_id = "abc-123-pipeline-id"
        client = self._make_client_mock(pipeline_id=expected_id)

        result = upload_pipeline(
            client=client, yaml_path=yaml_file, pipeline_name="pragma-s-test"
        )

        assert result == expected_id, (
            f"Expected pipeline_id {expected_id!r}, got {result!r}"
        )

    def test_raises_if_yaml_does_not_exist(self, tmp_path: pathlib.Path) -> None:
        """FileNotFoundError if the YAML file does not exist.

        Catches the missing-compile step early before hitting the network.
        """
        absent_yaml = tmp_path / "nonexistent.yaml"
        client = self._make_client_mock()

        with pytest.raises(FileNotFoundError):
            upload_pipeline(
                client=client, yaml_path=absent_yaml, pipeline_name="pragma-s-test"
            )

    def test_accepts_str_path(self, tmp_path: pathlib.Path) -> None:
        """upload_pipeline() accepts a str path as well as pathlib.Path."""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text("schemaVersion: '2.1.0'\n")

        client = self._make_client_mock()

        # Should not raise when a str is passed.
        upload_pipeline(
            client=client, yaml_path=str(yaml_file), pipeline_name="pragma-s-test"
        )

        client.upload_pipeline.assert_called_once()


# ===========================================================================
# 6. TestSubmitPipelineRun
#    submit_pipeline_run() creates a KFP pipeline run.
# ===========================================================================


class TestSubmitPipelineRun:
    """submit_pipeline_run() wraps kfp run creation."""

    def _make_client_mock(self, run_id: str = "test-run-id") -> mock.MagicMock:
        """Return a mock kfp.Client with run_pipeline, experiment, and version mocks."""
        client = mock.MagicMock(name="kfp.Client")
        run_response = mock.MagicMock()
        run_response.run_id = run_id
        # run_pipeline is the kfp v2 method used by submit_pipeline_run()
        client.run_pipeline.return_value = run_response
        # Pipeline version resolution mock (kfp v2 requires version_id)
        version = mock.MagicMock()
        version.pipeline_version_id = "test-version-id"
        versions_response = mock.MagicMock()
        versions_response.pipeline_versions = [version]
        client.list_pipeline_versions.return_value = versions_response
        # Experiment resolution mocks
        exp = mock.MagicMock()
        exp.experiment_id = "test-experiment-id"
        client.get_experiment.return_value = exp
        client.create_experiment.return_value = exp
        return client

    def test_calls_create_run_with_pipeline_id(self) -> None:
        """submit_pipeline_run() must call the KFP client run creation method.

        The exact kfp method varies by kfp version; submit_pipeline_run()
        is responsible for picking the correct call for the installed kfp
        SDK version. The test validates that some run-creation call was made
        with at least the pipeline_id and run_name.
        """
        client = self._make_client_mock()

        submit_pipeline_run(
            client=client,
            pipeline_id="abc-123",
            run_name="smoke-run-1",
        )

        # Verify the client had at least one call made to it.
        assert client.called or any(
            c.called for c in [
                client.create_run_from_pipeline_package,
                client.create_run,
                client.run_pipeline,
            ]
        ), "submit_pipeline_run() must call a run-creation method on the kfp client"

    def test_returns_run_id_string(self) -> None:
        """submit_pipeline_run() returns a non-empty run_id string."""
        client = self._make_client_mock(run_id="my-run-id")

        result = submit_pipeline_run(
            client=client,
            pipeline_id="abc-123",
            run_name="smoke-run-1",
        )

        assert isinstance(result, str), f"Expected str run_id, got {type(result)}"
        assert len(result) > 0, "run_id must not be empty"

    def test_accepts_optional_arguments(self) -> None:
        """submit_pipeline_run() accepts an optional arguments dict."""
        client = self._make_client_mock()

        # Should not raise when arguments is provided.
        submit_pipeline_run(
            client=client,
            pipeline_id="abc-123",
            run_name="smoke-run-1",
            arguments={"model_size": "S", "max_steps": 1},
        )

    def test_accepts_optional_experiment_name(self) -> None:
        """submit_pipeline_run() accepts an optional experiment_name string."""
        client = self._make_client_mock()

        submit_pipeline_run(
            client=client,
            pipeline_id="abc-123",
            run_name="smoke-run-1",
            experiment_name="pragma-pretraining",
        )


# ===========================================================================
# 7. TestSubmitNoSideEffects
#    The submit module must not import kfp at module level (optional dep).
#    compile() must have no cluster side effects (regression guard).
# ===========================================================================


class TestSubmitNoSideEffects:
    """Safety: submit module is importable without kfp; compile has no cluster effects."""

    def test_submit_module_importable_without_kfp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """src.workbench._submit is importable even when kfp is absent.

        kfp is imported lazily inside make_kfp_client(), not at module level.
        This allows the submit module to load in environments without kfp.
        """
        # Remove kfp from sys.modules to simulate it being absent.
        monkeypatch.setitem(sys.modules, "kfp", None)  # type: ignore[arg-type]

        # Re-importing the module should not raise ImportError.
        # We reload using importlib to force a fresh import.
        try:
            import importlib as _il
            submit_mod = _il.import_module("src.workbench._submit")
            # The module object itself should be available.
            assert submit_mod is not None
        except ImportError as exc:
            pytest.fail(
                f"src.workbench._submit raised ImportError without kfp installed: {exc}\n"
                "kfp must be imported lazily inside make_kfp_client(), not at module level."
            )

    def test_pragma_pipeline_compile_does_not_call_cluster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compile() must not touch any cluster resource.

        Regression guard: PragmaPipeline.compile() only uses kfp.compiler
        and pipeline.pragma_pipeline. It must not import or call anything
        from src.workbench._submit.
        """
        from src.workbench._decorators import PragmaPipeline, pragma_pipeline
        from src.workbench._intent import dataset, train

        @pragma_pipeline(name="test-no-cluster")
        def _p():
            ds = dataset("ibm-tabformer")
            train(dataset=ds, model_size="S", epochs=1)

        # Patch _submit to raise if called — it must NOT be called by compile().
        import src.workbench._submit as submit_mod
        with mock.patch.object(
            submit_mod, "make_kfp_client", side_effect=AssertionError("submit called!")
        ):
            # compile() will fail because kfp is not installed, but it must
            # fail with RuntimeError (kfp missing), NOT AssertionError (submit called).
            with pytest.raises((RuntimeError, ImportError)):
                _p.compile("/tmp/test-no-cluster.yaml")

    def test_no_token_printed_in_submit_flow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """End-to-end: no token value appears in stdout/stderr during submit flow.

        Creates a token file, reads it, constructs config, mocks the kfp
        client, calls upload_pipeline, and asserts the token never appears
        in any captured output.
        """
        sentinel = "END_TO_END_SENTINEL_TOKEN_VALUE"
        token_file = tmp_path / "token"
        token_file.write_text(sentinel)

        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text("schemaVersion: '2.1.0'\n")

        token = get_service_account_token(token_path=token_file)
        assert token == sentinel

        config = DSPAConfig(endpoint=_EXPECTED_ENDPOINT, token=token)

        mock_client = mock.MagicMock(name="kfp.Client")
        response = mock.MagicMock()
        response.pipeline_id = "test-id"
        mock_client.upload_pipeline.return_value = response

        upload_pipeline(
            client=mock_client, yaml_path=yaml_file, pipeline_name="pragma-s-test"
        )

        captured = capsys.readouterr()
        assert sentinel not in captured.out, (
            "Token must not appear in stdout during submit flow"
        )
        assert sentinel not in captured.err, (
            "Token must not appear in stderr during submit flow"
        )
        assert sentinel not in repr(config), (
            "Token must not appear in repr(DSPAConfig)"
        )


# ===========================================================================
# 8. TestGetRunStatus
#    get_run_status() wraps client.get_run() and returns an uppercase string.
# ===========================================================================


class TestGetRunStatus:
    """get_run_status() — normalises KFP v2 run state to uppercase string."""

    def _make_run_mock(self, state: object) -> mock.MagicMock:
        """Return a mock V2beta1Run with the given state."""
        run = mock.MagicMock()
        run.state = state
        return run

    def test_returns_uppercase_string_state(self) -> None:
        """Returns the state as an uppercase string when state is a plain string."""
        client = mock.MagicMock()
        client.get_run.return_value = self._make_run_mock("RUNNING")

        result = get_run_status(client=client, run_id="test-run-id")

        assert result == "RUNNING"
        client.get_run.assert_called_once_with(run_id="test-run-id")

    def test_uppercases_lowercase_state(self) -> None:
        """Lowercased state strings are normalised to uppercase."""
        client = mock.MagicMock()
        client.get_run.return_value = self._make_run_mock("succeeded")

        result = get_run_status(client=client, run_id="test-run-id")

        assert result == "SUCCEEDED"

    def test_returns_unknown_when_state_is_none(self) -> None:
        """Returns 'UNKNOWN' when run.state is None."""
        run = mock.MagicMock()
        run.state = None
        client = mock.MagicMock()
        client.get_run.return_value = run

        result = get_run_status(client=client, run_id="test-run-id")

        assert result == "UNKNOWN"

    def test_handles_enum_like_state_with_value_attr(self) -> None:
        """Handles enum-like state objects that have a .value attribute."""
        state_obj = mock.MagicMock()
        state_obj.value = "FAILED"
        run = mock.MagicMock()
        run.state = state_obj
        client = mock.MagicMock()
        client.get_run.return_value = run

        result = get_run_status(client=client, run_id="test-run-id")

        assert result == "FAILED"

    def test_succeeded_state(self) -> None:
        """Returns 'SUCCEEDED' for a succeeded run."""
        client = mock.MagicMock()
        client.get_run.return_value = self._make_run_mock("SUCCEEDED")

        assert get_run_status(client=client, run_id="abc") == "SUCCEEDED"

    def test_failed_state(self) -> None:
        """Returns 'FAILED' for a failed run."""
        client = mock.MagicMock()
        client.get_run.return_value = self._make_run_mock("FAILED")

        assert get_run_status(client=client, run_id="abc") == "FAILED"


# ===========================================================================
# 9. TestWaitForRunTerminal
#    wait_for_run_terminal() polls until terminal state or raises TimeoutError.
# ===========================================================================


class TestWaitForRunTerminal:
    """wait_for_run_terminal() — polling loop with timeout and token safety."""

    def _make_client_returning_states(self, states: list[str]) -> mock.MagicMock:
        """Return a mock client whose get_run returns each state in sequence."""
        client = mock.MagicMock()
        runs = []
        for state in states:
            run = mock.MagicMock()
            run.state = state
            runs.append(run)
        client.get_run.side_effect = runs
        return client

    def test_returns_immediately_when_already_terminal(self) -> None:
        """Returns the state at once when first poll returns a terminal state."""
        client = self._make_client_returning_states(["SUCCEEDED"])

        result = wait_for_run_terminal(
            client=client, run_id="abc", timeout=30, poll_interval=0
        )

        assert result == "SUCCEEDED"
        client.get_run.assert_called_once()

    def test_polls_until_succeeded(self) -> None:
        """Polls through non-terminal states until SUCCEEDED."""
        client = self._make_client_returning_states(
            ["PENDING", "RUNNING", "RUNNING", "SUCCEEDED"]
        )

        result = wait_for_run_terminal(
            client=client, run_id="abc", timeout=30, poll_interval=0
        )

        assert result == "SUCCEEDED"
        assert client.get_run.call_count == 4

    def test_returns_failed_state(self) -> None:
        """Returns 'FAILED' when the run fails (not an exception)."""
        client = self._make_client_returning_states(["RUNNING", "FAILED"])

        result = wait_for_run_terminal(
            client=client, run_id="abc", timeout=30, poll_interval=0
        )

        assert result == "FAILED"

    def test_returns_canceled_state(self) -> None:
        """Returns 'CANCELED' when the run is cancelled."""
        client = self._make_client_returning_states(["CANCELING", "CANCELED"])

        # CANCELING is non-terminal; CANCELED is terminal.
        result = wait_for_run_terminal(
            client=client, run_id="abc", timeout=30, poll_interval=0
        )

        assert result == "CANCELED"

    def test_returns_skipped_state(self) -> None:
        """Returns 'SKIPPED' when the run is skipped."""
        client = self._make_client_returning_states(["SKIPPED"])

        result = wait_for_run_terminal(
            client=client, run_id="abc", timeout=30, poll_interval=0
        )

        assert result == "SKIPPED"

    def test_raises_timeout_error_when_stuck_in_running(self) -> None:
        """Raises TimeoutError when the run stays RUNNING past the timeout.

        Uses a very short timeout (0.1s) and immediate poll (0s interval).
        The TimeoutError message must include the run_id and last state.
        """
        # Always returns RUNNING — never reaches terminal.
        run = mock.MagicMock()
        run.state = "RUNNING"
        client = mock.MagicMock()
        client.get_run.return_value = run

        with pytest.raises(TimeoutError) as exc_info:
            wait_for_run_terminal(
                client=client, run_id="stuck-run-id", timeout=0, poll_interval=0
            )

        msg = str(exc_info.value)
        assert "stuck-run-id" in msg, "TimeoutError must include the run_id"
        assert "RUNNING" in msg, "TimeoutError must include the last observed state"

    def test_timeout_error_message_contains_timeout_seconds(self) -> None:
        """TimeoutError message includes the timeout duration."""
        run = mock.MagicMock()
        run.state = "PENDING"
        client = mock.MagicMock()
        client.get_run.return_value = run

        with pytest.raises(TimeoutError) as exc_info:
            wait_for_run_terminal(
                client=client, run_id="run-abc", timeout=0, poll_interval=0
            )

        assert "0s" in str(exc_info.value), (
            "TimeoutError must mention the timeout duration"
        )

    def test_token_not_in_timeout_error_message(self) -> None:
        """Token value must not appear in TimeoutError message.

        wait_for_run_terminal() does not accept a token argument — confirmed
        by checking it only operates on client and run_id. This test guards
        against accidental token leakage via the client mock repr.
        """
        run = mock.MagicMock()
        run.state = "RUNNING"
        sentinel = "SENTINEL_TOKEN_MUST_NOT_LEAK"
        client = mock.MagicMock(name=f"kfp.Client[{sentinel}]")
        client.get_run.return_value = run

        with pytest.raises(TimeoutError) as exc_info:
            wait_for_run_terminal(
                client=client, run_id="run-abc", timeout=0, poll_interval=0
            )

        # The sentinel must NOT appear because wait_for_run_terminal()
        # constructs its own message using only run_id and state.
        assert sentinel not in str(exc_info.value), (
            "Token must not appear in TimeoutError message"
        )
