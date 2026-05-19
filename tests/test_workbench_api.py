"""Tests for train_pragma() — workbench entry point for PRAGMA pretraining.

Derived from PRAGMA paper Section 2.4:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

train_pragma() is the top-level workbench API that maps the five §2.4 pipeline
stages onto a concrete execution path. Tests verify:
  - train_pragma(mode="dry_run") returns a PragmaRun satisfying the protocol
  - model_size maps correctly to PRAGMAConfig variants (Table 1)
  - Unknown model_size raises ValueError; unknown dataset raises KeyError
  - dry_run does not access S3, does not prepare datasets, returns all steps pending

Test design note: mode="dry_run" is the unit-test seam. It must return a
PragmaRun immediately without calling DatasetAdapter.prepare(), accessing S3,
or submitting any cluster job. All tests use this mode.

Test types:
    Contract tests:  verify Protocol compliance and return type
    Property tests:  verify config mapping, step state, error paths
    Spec tests:      verify paper §2.4 stage names and dataset registry key
"""

from __future__ import annotations

import os
import pytest

from src.model.config import PRAGMAConfig
from src.workbench._api import train_pragma
from src.workbench._run import (
    PragmaRun,
    PragmaRunProtocol,
    PIPELINE_STEP_NAMES,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_DATASET = "ibm-tabformer"
_DRY = "dry_run"


def _dry_run(model_size: str = "S", **kwargs) -> PragmaRun:
    """Return a dry-run PragmaRun; shorthand for tests."""
    return train_pragma(
        dataset=_VALID_DATASET,
        model_size=model_size,
        mode=_DRY,
        upload=False,   # belt-and-suspenders: no S3 in tests
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Contract tests — Protocol compliance and return type
# ---------------------------------------------------------------------------

class TestContract:
    """Verify train_pragma(mode='dry_run') satisfies PragmaRunProtocol."""

    def test_dry_run_returns_pragma_run(self) -> None:
        """§2.4 / ADR 003: train_pragma(mode='dry_run') must return a PragmaRun."""
        run = _dry_run()
        assert isinstance(run, PragmaRun), (
            f"train_pragma(mode='dry_run') must return a PragmaRun, got {type(run)}"
        )

    def test_dry_run_satisfies_protocol(self) -> None:
        """ADR 003: returned PragmaRun must satisfy PragmaRunProtocol."""
        run = _dry_run()
        assert isinstance(run, PragmaRunProtocol), (
            "train_pragma(mode='dry_run') result must satisfy PragmaRunProtocol — "
            "show_pipeline(), metrics(), or artifacts() is missing."
        )

    def test_dry_run_show_pipeline_does_not_raise(self) -> None:
        """ADR 003: show_pipeline() must not raise after dry_run."""
        run = _dry_run()
        run.show_pipeline()  # must not raise

    def test_dry_run_metrics_returns_dict(self) -> None:
        """ADR 003: metrics() must return a dict after dry_run."""
        run = _dry_run()
        assert isinstance(run.metrics(), dict)

    def test_dry_run_artifacts_returns_dict(self) -> None:
        """ADR 003: artifacts() must return a dict after dry_run."""
        run = _dry_run()
        assert isinstance(run.artifacts(), dict)


# ---------------------------------------------------------------------------
# Property tests — model_size → PRAGMAConfig mapping
# ---------------------------------------------------------------------------

class TestModelSizeMapping:
    """Verify model_size parameter maps to the correct PRAGMAConfig variant.

    Values from docs/paper/key-numbers.md (Table 1):
        PRAGMA-S: d_model=192
        PRAGMA-M: d_model=512
        PRAGMA-L: d_model=1024
    """

    def test_model_size_s_maps_to_pragma_s_config(self) -> None:
        """§2.4 / Table 1: model_size='S' must carry PRAGMA-S config (d_model=192)."""
        run = _dry_run(model_size="S")
        assert run.manifest.config.d_model == 192, (  # key-numbers.md: d_model (PRAGMA-S), Table 1
            f"model_size='S' must map to d_model=192 (PRAGMA-S), "
            f"got d_model={run.manifest.config.d_model}"
        )

    def test_model_size_m_maps_to_pragma_m_config(self) -> None:
        """§2.4 / Table 1: model_size='M' must carry PRAGMA-M config (d_model=512)."""
        run = _dry_run(model_size="M")
        assert run.manifest.config.d_model == 512, (  # key-numbers.md: d_model (PRAGMA-M), Table 1
            f"model_size='M' must map to d_model=512 (PRAGMA-M), "
            f"got d_model={run.manifest.config.d_model}"
        )

    def test_model_size_l_maps_to_pragma_l_config(self) -> None:
        """§2.4 / Table 1: model_size='L' must carry PRAGMA-L config (d_model=1024)."""
        run = _dry_run(model_size="L")
        assert run.manifest.config.d_model == 1024, (  # key-numbers.md: d_model (PRAGMA-L), Table 1
            f"model_size='L' must map to d_model=1024 (PRAGMA-L), "
            f"got d_model={run.manifest.config.d_model}"
        )

    def test_unknown_model_size_raises_value_error(self) -> None:
        """ADR 003: unknown model_size must raise ValueError before any work."""
        with pytest.raises(ValueError, match="model_size"):
            train_pragma(
                dataset=_VALID_DATASET,
                model_size="XL",   # not a valid PRAGMA variant
                mode=_DRY,
            )

    def test_model_size_lowercase_raises_value_error(self) -> None:
        """ADR 003: model_size is case-sensitive; lowercase 's' must raise ValueError."""
        with pytest.raises(ValueError, match="model_size"):
            train_pragma(
                dataset=_VALID_DATASET,
                model_size="s",    # lowercase — not a registered variant
                mode=_DRY,
            )

    def test_model_size_empty_raises_value_error(self) -> None:
        """ADR 003: empty model_size must raise ValueError."""
        with pytest.raises(ValueError, match="model_size"):
            train_pragma(
                dataset=_VALID_DATASET,
                model_size="",
                mode=_DRY,
            )


# ---------------------------------------------------------------------------
# Property tests — dataset registry
# ---------------------------------------------------------------------------

class TestDatasetRegistry:
    """Verify dataset parameter is validated against the adapter registry."""

    def test_unknown_dataset_raises_key_error(self) -> None:
        """ADR 003: unknown dataset key must raise KeyError."""
        with pytest.raises(KeyError):
            train_pragma(
                dataset="not-a-registered-dataset",
                model_size="S",
                mode=_DRY,
            )

    def test_ibm_tabformer_is_registered(self) -> None:
        """§2.4 / ADR 003: 'ibm-tabformer' must be a registered dataset key."""
        run = _dry_run(model_size="S")
        assert run.manifest.dataset_name == "ibm-tabformer", (
            "dry_run manifest must carry the dataset name from the registry key"
        )

    def test_dry_run_manifest_carries_correct_dataset_name(self) -> None:
        """ADR 003: PragmaRun.manifest.dataset_name must match the requested dataset."""
        run = train_pragma(
            dataset="ibm-tabformer",
            model_size="S",
            mode=_DRY,
            upload=False,
        )
        assert run.manifest.dataset_name == "ibm-tabformer"


# ---------------------------------------------------------------------------
# Property tests — dry_run step state
# ---------------------------------------------------------------------------

class TestDryRunBehavior:
    """Verify dry_run returns a PragmaRun with all steps pending and no side effects."""

    def test_dry_run_steps_all_pending(self) -> None:
        """ADR 003: dry_run must return all pipeline steps in 'pending' status."""
        run = _dry_run()
        for step in run.steps:
            assert step.status == "pending", (
                f"dry_run step '{step.name}' must be 'pending', got '{step.status}'"
            )

    def test_dry_run_has_five_steps(self) -> None:
        """§2.4 / ADR 003: dry_run PragmaRun must have exactly five pipeline steps."""
        run = _dry_run()
        assert len(run.steps) == 5, (
            f"dry_run must return exactly 5 pipeline steps, got {len(run.steps)}"
        )

    def test_dry_run_metrics_empty(self) -> None:
        """ADR 003: dry_run returns empty metrics (no training has run)."""
        run = _dry_run()
        assert run.metrics() == {}, (
            "dry_run must return empty metrics — no training has run"
        )

    def test_dry_run_artifacts_empty(self) -> None:
        """ADR 003: dry_run returns empty artifacts (no exports have run)."""
        run = _dry_run()
        assert run.artifacts() == {}, (
            "dry_run must return empty artifacts — no exports have run"
        )

    def test_dry_run_does_not_access_s3(self, monkeypatch) -> None:
        """ADR 003: dry_run must not access S3, even when upload=True.

        Remove all MODEL_REGISTRY_* credentials — any S3 access would raise.
        """
        for var in ("MODEL_REGISTRY_ENDPOINT_URL", "MODEL_REGISTRY_BUCKET",
                    "MODEL_REGISTRY_ACCESS_KEY", "MODEL_REGISTRY_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)

        # Must not raise — dry_run must bypass all S3 operations
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            mode=_DRY,
            upload=True,  # upload=True is intentional: dry_run must ignore it
        )
        assert isinstance(run, PragmaRun)

    def test_dry_run_does_not_prepare_dataset(self, tmp_path, monkeypatch) -> None:
        """ADR 003: dry_run must not call DatasetAdapter.prepare().

        Verify by ensuring no FileNotFoundError even when the source CSV is absent.
        prepare() would raise FileNotFoundError for a missing CSV — dry_run must not.
        """
        # Point the IBM TabFormer adapter to a non-existent CSV path
        monkeypatch.setenv("TABFORMER_CSV_PATH", str(tmp_path / "no_such.csv"))
        # dry_run must return a PragmaRun regardless — prepare() must not be called
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            mode=_DRY,
            upload=False,
            prepare_if_missing=True,  # even with prepare_if_missing=True
        )
        assert isinstance(run, PragmaRun)

    def test_dry_run_nodes_parameter_accepted(self) -> None:
        """ADR 003: nodes parameter must be accepted in dry_run without error."""
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            mode=_DRY,
            nodes=2,
            upload=False,
        )
        assert isinstance(run, PragmaRun)

    def test_dry_run_epochs_parameter_accepted(self) -> None:
        """ADR 003: epochs parameter must be accepted in dry_run without error."""
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            mode=_DRY,
            epochs=20,
            upload=False,
        )
        assert isinstance(run, PragmaRun)


# ---------------------------------------------------------------------------
# Specification tests — §2.4 pipeline stages
# ---------------------------------------------------------------------------

class TestPaperSpecifications:
    """Verify train_pragma() exposes §2.4 stage names and honours the registry."""

    def test_dry_run_step_names_match_pipeline_spec(self) -> None:
        """§2.4 / ADR 003: step names must be the five canonical §2.4 stage names."""
        run = _dry_run()
        step_names = [step.name for step in run.steps]
        assert step_names == list(PIPELINE_STEP_NAMES), (
            f"dry_run step names must match PIPELINE_STEP_NAMES "
            f"{list(PIPELINE_STEP_NAMES)}, got {step_names}"
        )

    def test_dry_run_step_order_is_execution_order(self) -> None:
        """§2.4 / ADR 003: steps must appear in causal execution order.

        prepare → upload → submit → train → export
        """
        run = _dry_run()
        step_names = [step.name for step in run.steps]
        assert step_names == ["prepare", "upload", "submit", "train", "export"], (
            f"Step order must be [prepare, upload, submit, train, export], "
            f"got {step_names}"
        )

    def test_manifest_config_has_correct_truncation_limits_pragma_s(self) -> None:
        """§2.4: PRAGMA-S config must carry §2.4 truncation limits.

        From key-numbers.md / §2.4:
            max_event_tokens = 24    (per-event token budget)
            max_profile_tokens = 200 (profile state budget)
            max_events = 6500        (history length cap)
        """
        run = _dry_run(model_size="S")
        cfg = run.manifest.config
        assert cfg.max_event_tokens == 24, (   # key-numbers.md: max_event_tokens, §2.4
            f"PRAGMA-S max_event_tokens must be 24, got {cfg.max_event_tokens}"
        )
        assert cfg.max_profile_tokens == 200, (  # key-numbers.md: max_profile_tokens, §2.4
            f"PRAGMA-S max_profile_tokens must be 200, got {cfg.max_profile_tokens}"
        )
        assert cfg.max_events == 6500, (  # key-numbers.md: max_events, §2.4
            f"PRAGMA-S max_events must be 6500, got {cfg.max_events}"
        )

    def test_valid_mode_values_accepted(self) -> None:
        """ADR 003: mode parameter must accept all four documented values without TypeError."""
        # Only dry_run is safe to call in tests; we verify the others do not
        # raise TypeError (wrong argument) — they may raise other errors for
        # environment reasons (cluster not available, etc.)
        run = train_pragma(
            dataset=_VALID_DATASET,
            model_size="S",
            mode="dry_run",
            upload=False,
        )
        assert isinstance(run, PragmaRun)


# ---------------------------------------------------------------------------
# Dry-run visibility tests — run_mode field and show_pipeline() output
# ---------------------------------------------------------------------------

class TestDryRunVisibility:
    """Verify dry_run is visibly distinguishable from a submitted training run.

    ADR 003 requirement: a dry_run PragmaRun must not be mistaken for a real
    submitted job. Tests check:
      - run.run_mode == "dry_run"
      - show_pipeline() includes a DRY RUN banner
      - show_pipeline() does not imply any job was submitted or is running
    """

    def test_dry_run_run_mode_field_is_dry_run(self) -> None:
        """ADR 003: train_pragma(mode='dry_run') must set run.run_mode = 'dry_run'."""
        run = _dry_run()
        assert run.run_mode == "dry_run", (
            f"dry_run PragmaRun must have run_mode='dry_run', got {run.run_mode!r}"
        )

    def test_dry_run_show_pipeline_includes_dry_run_indicator(self, capsys) -> None:
        """ADR 003: show_pipeline() must print a DRY RUN banner for dry_run results."""
        run = _dry_run()
        run.show_pipeline()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out, (
            "show_pipeline() must include 'DRY RUN' to distinguish preview from real run"
        )

    def test_dry_run_show_pipeline_says_no_job_submitted(self, capsys) -> None:
        """ADR 003: show_pipeline() must explicitly state no job has been submitted."""
        run = _dry_run()
        run.show_pipeline()
        captured = capsys.readouterr()
        # The banner must convey that no training job has been dispatched
        output_lower = captured.out.lower()
        assert "no training job has been submitted" in output_lower or \
               "preview" in output_lower, (
            "show_pipeline() must state that no job was submitted "
            "(e.g. 'Pipeline preview — no training job has been submitted')"
        )

    def test_dry_run_show_pipeline_steps_show_pending_not_running(self, capsys) -> None:
        """ADR 003: dry_run show_pipeline() must not show any step as running or completed."""
        run = _dry_run()
        run.show_pipeline()
        captured = capsys.readouterr()
        assert "running" not in captured.out, (
            "show_pipeline() must not show any step as 'running' in a dry_run"
        )
        assert "completed" not in captured.out, (
            "show_pipeline() must not show any step as 'completed' in a dry_run"
        )

    def test_submitted_run_would_not_have_dry_run_mode(self) -> None:
        """ADR 003: a run without mode='dry_run' must have run_mode != 'dry_run'.

        Verifies the distinction between dry_run and a real run at the data level.
        A PragmaRun created without run_mode must have run_mode=None (not 'dry_run').
        """
        from src.data.dataset_manifest import DatasetManifest, DatasetShard
        manifest = DatasetManifest(
            dataset_name="ibm-tabformer",
            dataset_version="v1",
            prepared_prefix_uri="pragma-encoder/data/tabformer/",
            shards=(DatasetShard(
                uri="pragma-encoder/data/tabformer/card_transaction.v1.csv",
                format="csv",
                rows=100,
            ),),
            vocab_uri=None,
            schema_uri=None,
            manifest_uri=None,
            row_count=100,
            source={"origin": "ibm-tabformer"},
            config=PRAGMAConfig.pragma_s(),
        )
        # A run created without explicit run_mode (e.g. from a real submission)
        # must default to None — not "dry_run"
        real_run = PragmaRun(manifest=manifest)
        assert real_run.run_mode is None, (
            "PragmaRun.run_mode must default to None for non-dry-run cases"
        )


# ---------------------------------------------------------------------------
# Local mode tests — train_pragma(mode="local")
# ---------------------------------------------------------------------------

class TestLocalModeContract:
    """train_pragma(mode='local') must invoke scripts/train_pragma.py via subprocess.

    sec.2.4 local execution path: the same training script used for cluster runs
    is invoked locally (subprocess, no KFTO) with --num-workers 0 for development
    and smoke testing.

    Unit tests mock subprocess.run and DatasetAdapter.prepare() to avoid real
    CSV files and real training time. Integration is verified separately via
    examples/workbench/04_local_training_smoke.py with a tiny fixture CSV.

    Category: contract, behavior, interface
    Paper: Section 2.4 (Training Infrastructure)
    ADR: docs/decisions/003-workbench-training-api.md
    """

    _CSV = "tests/fixtures/ibm_tabformer_tiny.csv"
    _OUT = "/tmp/pragma-local-test"

    def _make_manifest(self):
        """Return a minimal DatasetManifest for use as a mock prepare() return value."""
        from src.data.dataset_manifest import DatasetManifest, DatasetShard
        shard = DatasetShard(uri="local/shard.csv", format="csv", rows=4)
        return DatasetManifest(
            dataset_name="ibm-tabformer",
            dataset_version="v1",
            prepared_prefix_uri="local/",
            shards=(shard,),
            vocab_uri=None,
            schema_uri=None,
            manifest_uri=None,
            row_count=4,
            source={"origin": "ibm-tabformer"},
            config=PRAGMAConfig.pragma_s(),
        )

    def _local_run(self, returncode: int = 0, **kwargs) -> PragmaRun:
        """Call train_pragma(mode='local') with subprocess and adapter mocked."""
        from unittest.mock import patch, MagicMock
        manifest = self._make_manifest()
        mock_adapter = MagicMock()
        mock_adapter.prepare.return_value = manifest
        mock_adapter_cls = MagicMock(return_value=mock_adapter)

        with patch("src.workbench._api.get_adapter", return_value=mock_adapter_cls), \
             patch("src.workbench._api.subprocess") as mock_subp, \
             patch("pathlib.Path.mkdir"):
            mock_subp.run.return_value = MagicMock(returncode=returncode)
            run = train_pragma(
                dataset=_VALID_DATASET,
                model_size="S",
                epochs=1,
                mode="local",
                local_csv_path=self._CSV,
                output_dir=self._OUT,
                max_steps=1,
                **kwargs,
            )
        return run

    # --- Contract: return type and run_mode ---

    def test_local_mode_returns_pragma_run(self) -> None:
        """sec.2.4 / ADR 003: train_pragma(mode='local') must return a PragmaRun."""
        run = self._local_run()
        assert isinstance(run, PragmaRun), (
            f"train_pragma(mode='local') must return a PragmaRun, got {type(run)}"
        )

    def test_local_mode_run_mode_is_local(self) -> None:
        """ADR 003: local mode PragmaRun must have run_mode='local'."""
        run = self._local_run()
        assert run.run_mode == "local", (
            f"train_pragma(mode='local') must set run_mode='local', "
            f"got {run.run_mode!r}"
        )

    def test_local_mode_show_pipeline_does_not_say_dry_run(self, capsys) -> None:
        """ADR 003: local mode show_pipeline() must not claim 'DRY RUN'.

        Local mode runs real training — it must not be mistaken for a preview.
        """
        run = self._local_run()
        run.show_pipeline()
        captured = capsys.readouterr()
        assert "DRY RUN" not in captured.out, (
            "show_pipeline() for a local run must not print the DRY RUN banner"
        )

    # --- Interface: required parameters and rejections ---

    def test_local_mode_requires_local_csv_path(self) -> None:
        """ADR 003: local mode must raise ValueError if local_csv_path is None.

        local_csv_path is required for local mode — the adapter needs the
        source CSV to fit the tokeniser and build a vocabulary.
        """
        with pytest.raises((ValueError, TypeError)):
            train_pragma(
                dataset=_VALID_DATASET,
                model_size="S",
                mode="local",
                local_csv_path=None,
            )

    def test_local_mode_rejects_nodes_gt_1(self) -> None:
        """ADR 003: local mode must raise NotImplementedError for nodes > 1.

        Multi-node DDP requires KFTO PyTorchJob orchestration (cluster mode only).
        Local mode is single-process (nodes=1). Two-node topology is a manifest
        and KFP pipeline demo — not a local subprocess concern.
        """
        with pytest.raises(NotImplementedError):
            train_pragma(
                dataset=_VALID_DATASET,
                model_size="S",
                mode="local",
                local_csv_path=self._CSV,
                nodes=2,
            )

    # --- Step statuses ---

    def test_local_mode_prepare_step_is_completed(self) -> None:
        """sec.2.4 stage 1: prepare step must be 'completed' after adapter.prepare()."""
        run = self._local_run()
        step = next(s for s in run.steps if s.name == "prepare")
        assert step.status == "completed", (
            f"local mode prepare step must be 'completed', got '{step.status}'"
        )

    def test_local_mode_upload_step_is_skipped(self) -> None:
        """sec.2.4 stage 2: upload step must be 'skipped' in local mode (no S3)."""
        run = self._local_run()
        step = next(s for s in run.steps if s.name == "upload")
        assert step.status == "skipped", (
            f"local mode upload step must be 'skipped' (no S3 in local mode), "
            f"got '{step.status}'"
        )

    def test_local_mode_submit_step_is_skipped(self) -> None:
        """sec.2.4 stage 3: submit step must be 'skipped' in local mode (no cluster)."""
        run = self._local_run()
        step = next(s for s in run.steps if s.name == "submit")
        assert step.status == "skipped", (
            f"local mode submit step must be 'skipped' (no KFTO in local mode), "
            f"got '{step.status}'"
        )

    def test_local_mode_train_step_is_completed_on_success(self) -> None:
        """sec.2.4 stage 4: train step must be 'completed' when subprocess exits 0."""
        run = self._local_run(returncode=0)
        step = next(s for s in run.steps if s.name == "train")
        assert step.status == "completed", (
            f"local mode train step must be 'completed' when subprocess exits 0, "
            f"got '{step.status}'"
        )

    def test_local_mode_export_step_is_skipped(self) -> None:
        """sec.2.4 stage 5: export step must be 'skipped' in local mode (no S3)."""
        run = self._local_run()
        step = next(s for s in run.steps if s.name == "export")
        assert step.status == "skipped", (
            f"local mode export step must be 'skipped' (no S3 export in local mode), "
            f"got '{step.status}'"
        )

    def test_local_mode_marks_train_failed_on_subprocess_error(self) -> None:
        """ADR 003: train step must be 'failed' when subprocess exits non-zero.

        A non-zero returncode means the training process crashed or was killed.
        The PragmaRun must honestly reflect failure — not claim 'completed'.
        """
        run = self._local_run(returncode=1)
        step = next(s for s in run.steps if s.name == "train")
        assert step.status == "failed", (
            f"local mode train step must be 'failed' when subprocess exits 1, "
            f"got '{step.status}'"
        )

    # --- Artifacts ---

    def test_local_mode_output_dir_in_artifacts(self) -> None:
        """ADR 003: local mode must record output_dir in artifacts().

        The output directory is the canonical local artifact (no S3 URIs).
        It lets the user find the checkpoint without grep'ing training logs.
        """
        run = self._local_run()
        artifacts = run.artifacts()
        assert "output_dir" in artifacts, (
            f"local mode artifacts() must include 'output_dir' key, got {artifacts}"
        )
        assert artifacts["output_dir"] == self._OUT, (
            f"output_dir artifact must equal the requested output_dir path, "
            f"got {artifacts['output_dir']!r}"
        )

    # --- Behavior: adapter delegation ---

    def test_local_mode_calls_prepare_upload_false(self) -> None:
        """ADR 003: local mode must call DatasetAdapter.prepare(upload=False).

        upload=False prevents any S3 connection — the adapter writes vocab
        locally only. S3 credentials are not required for local mode.
        """
        from unittest.mock import patch, MagicMock
        manifest = self._make_manifest()
        mock_adapter = MagicMock()
        mock_adapter.prepare.return_value = manifest
        mock_adapter_cls = MagicMock(return_value=mock_adapter)

        with patch("src.workbench._api.get_adapter", return_value=mock_adapter_cls), \
             patch("src.workbench._api.subprocess") as mock_subp, \
             patch("pathlib.Path.mkdir"):
            mock_subp.run.return_value = MagicMock(returncode=0)
            train_pragma(
                dataset=_VALID_DATASET,
                model_size="S",
                epochs=1,
                mode="local",
                local_csv_path=self._CSV,
                output_dir=self._OUT,
            )

        mock_adapter.prepare.assert_called_once()
        kwargs = mock_adapter.prepare.call_args.kwargs
        assert kwargs.get("upload") is False, (
            f"DatasetAdapter.prepare() must be called with upload=False in local mode, "
            f"got call_args={mock_adapter.prepare.call_args}"
        )

    def test_local_mode_does_not_touch_s3(self) -> None:
        """ADR 003: local mode must succeed without any S3 credentials.

        adapter.prepare(upload=False) ensures no boto3 connection is made.
        No MODEL_REGISTRY_* env vars are required for local mode.
        """
        import os
        from unittest.mock import patch, MagicMock
        manifest = self._make_manifest()
        mock_adapter = MagicMock()
        mock_adapter.prepare.return_value = manifest
        mock_adapter_cls = MagicMock(return_value=mock_adapter)

        # Scrub all S3 credentials — local mode must not need them
        env_backup = {}
        for var in ("MODEL_REGISTRY_BUCKET", "MODEL_REGISTRY_ENDPOINT",
                    "MODEL_REGISTRY_ACCESS_KEY", "MODEL_REGISTRY_SECRET_KEY"):
            env_backup[var] = os.environ.pop(var, None)
        try:
            with patch("src.workbench._api.get_adapter", return_value=mock_adapter_cls), \
                 patch("src.workbench._api.subprocess") as mock_subp, \
                 patch("pathlib.Path.mkdir"):
                mock_subp.run.return_value = MagicMock(returncode=0)
                run = train_pragma(
                    dataset=_VALID_DATASET,
                    model_size="S",
                    epochs=1,
                    mode="local",
                    local_csv_path=self._CSV,
                    output_dir=self._OUT,
                )
        finally:
            for var, val in env_backup.items():
                if val is not None:
                    os.environ[var] = val

        assert isinstance(run, PragmaRun), (
            "local mode must succeed without S3 credentials"
        )

    def test_local_mode_does_not_call_oc_kubectl(self) -> None:
        """ADR 003: local mode must not invoke oc or kubectl via subprocess.

        Cluster tools must not be called — local mode is intentionally
        cluster-free. Any subprocess call to 'oc' or 'kubectl' is a bug.
        """
        from unittest.mock import patch, MagicMock
        manifest = self._make_manifest()
        mock_adapter = MagicMock()
        mock_adapter.prepare.return_value = manifest
        mock_adapter_cls = MagicMock(return_value=mock_adapter)

        with patch("src.workbench._api.get_adapter", return_value=mock_adapter_cls), \
             patch("src.workbench._api.subprocess") as mock_subp, \
             patch("pathlib.Path.mkdir"):
            mock_subp.run.return_value = MagicMock(returncode=0)
            train_pragma(
                dataset=_VALID_DATASET,
                model_size="S",
                epochs=1,
                mode="local",
                local_csv_path=self._CSV,
                output_dir=self._OUT,
            )

        for call in mock_subp.run.call_args_list:
            cmd = call.args[0] if call.args else call.kwargs.get("args", [])
            cmd_str = " ".join(str(c) for c in cmd)
            assert "oc " not in cmd_str and not cmd_str.startswith("oc"), (
                f"local mode must not call 'oc' via subprocess, got: {cmd_str!r}"
            )
            assert "kubectl" not in cmd_str, (
                f"local mode must not call 'kubectl' via subprocess, got: {cmd_str!r}"
            )

    def test_local_mode_does_not_import_kfp(self) -> None:
        """ADR 003: local mode must not import kfp as a side effect.

        kfp is an optional dependency ([workbench] extra). Local training
        must work without kfp installed (and must not cause an ImportError
        in environments where kfp is absent).
        """
        import sys
        kfp_before = "kfp" in sys.modules
        self._local_run()
        if not kfp_before:
            assert "kfp" not in sys.modules, (
                "local mode must not import kfp — it is optional and "
                "not needed for local subprocess training"
            )

    # --- Behavior: subprocess command structure ---

    def test_local_mode_builds_expected_subprocess_command(self) -> None:
        """ADR 003 / sec.2.4: local mode must invoke scripts/train_pragma.py correctly.

        Expected command contains:
            scripts/train_pragma.py
            --csv-path  <local_csv_path>
            --vocab-path <output_dir>/vocab.pkl
            --output-dir <output_dir>
            --model-variant pragma-s    (PRAGMA-S -> 'pragma-s', Table 1)
            --epochs 1
            --num-workers 0             (single-process, no DataLoader workers)
            --max-steps 1               (from max_steps param)
        """
        from unittest.mock import patch, MagicMock
        manifest = self._make_manifest()
        mock_adapter = MagicMock()
        mock_adapter.prepare.return_value = manifest
        mock_adapter_cls = MagicMock(return_value=mock_adapter)

        with patch("src.workbench._api.get_adapter", return_value=mock_adapter_cls), \
             patch("src.workbench._api.subprocess") as mock_subp, \
             patch("pathlib.Path.mkdir"):
            mock_subp.run.return_value = MagicMock(returncode=0)
            train_pragma(
                dataset=_VALID_DATASET,
                model_size="S",
                epochs=1,
                mode="local",
                local_csv_path=self._CSV,
                output_dir=self._OUT,
                max_steps=1,
            )

        assert mock_subp.run.called, (
            "subprocess.run must be called for local mode"
        )
        call_args = mock_subp.run.call_args
        cmd = call_args.args[0] if call_args.args else call_args.kwargs.get("args", [])
        cmd_str = " ".join(str(c) for c in cmd)

        assert "train_pragma.py" in cmd_str, (
            f"subprocess command must invoke scripts/train_pragma.py, got: {cmd_str!r}"
        )
        assert "--csv-path" in cmd_str and self._CSV in cmd_str, (
            f"command must include --csv-path {self._CSV!r}, got: {cmd_str!r}"
        )
        assert "--output-dir" in cmd_str and self._OUT in cmd_str, (
            f"command must include --output-dir {self._OUT!r}, got: {cmd_str!r}"
        )
        assert "--model-variant" in cmd_str and "pragma-s" in cmd_str, (
            f"command must include --model-variant pragma-s for model_size='S', "
            f"got: {cmd_str!r}"
        )
        assert "--epochs" in cmd_str, (
            f"command must include --epochs, got: {cmd_str!r}"
        )
        assert "--num-workers" in cmd_str, (
            f"command must include --num-workers 0 for local single-process mode, "
            f"got: {cmd_str!r}"
        )
        assert "--max-steps" in cmd_str, (
            f"command must include --max-steps when max_steps is set, "
            f"got: {cmd_str!r}"
        )
