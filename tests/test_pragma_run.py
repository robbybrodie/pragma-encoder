"""Tests for PragmaRun and PipelineStep.

Derived from PRAGMA paper Section 2.4:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

PragmaRun is an implementation decision (ADR 003). It surfaces the five §2.4
training infrastructure stages as observable pipeline steps. Tests verify:
  - PragmaRun satisfies PragmaRunProtocol
  - show_pipeline() names all five §2.4 stages
  - metrics() and artifacts() return dicts with the correct structure
  - PipelineStep rejects invalid names and statuses

Test types:
    Contract tests:  verify Protocol compliance and method signatures
    Property tests:  verify pipeline step invariants and dict structure
    Spec tests:      verify step names match §2.4 stages exactly
"""

from __future__ import annotations

import pytest

from src.model.config import PRAGMAConfig
from src.data.dataset_manifest import DatasetManifest, DatasetShard
from src.workbench._run import (
    PragmaRun,
    PragmaRunProtocol,
    PipelineStep,
    PIPELINE_STEP_NAMES,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_name="ibm-tabformer",
        dataset_version="v1",
        prepared_prefix_uri="pragma-encoder/data/tabformer/",
        shards=(DatasetShard(
            uri="pragma-encoder/data/tabformer/card_transaction.v1.csv",
            format="csv",
            rows=100,
        ),),
        vocab_uri="pragma-encoder/data/tabformer/vocab.pkl",
        schema_uri=None,
        manifest_uri=None,
        row_count=100,
        source={"origin": "ibm-tabformer", "format": "csv"},
        config=PRAGMAConfig.pragma_s(),
    )


def _make_run(**kwargs) -> PragmaRun:
    """Return a fresh PragmaRun in default (all-pending) state."""
    return PragmaRun(manifest=_make_manifest(), **kwargs)


# ---------------------------------------------------------------------------
# Contract tests — Protocol compliance
# ---------------------------------------------------------------------------

class TestContract:
    """Verify PragmaRun satisfies PragmaRunProtocol."""

    def test_run_implements_protocol(self) -> None:
        """ADR 003: PragmaRun must satisfy PragmaRunProtocol."""
        run = _make_run()
        assert isinstance(run, PragmaRunProtocol), (
            "PragmaRun does not satisfy PragmaRunProtocol — "
            "show_pipeline(), metrics(), or artifacts() is missing."
        )

    def test_show_pipeline_does_not_raise(self) -> None:
        """ADR 003: show_pipeline() must not raise in any run state."""
        run = _make_run()
        run.show_pipeline()  # must not raise

    def test_metrics_returns_dict(self) -> None:
        """ADR 003: metrics() must return a dict."""
        run = _make_run()
        result = run.metrics()
        assert isinstance(result, dict), (
            f"metrics() must return a dict, got {type(result)}"
        )

    def test_artifacts_returns_dict(self) -> None:
        """ADR 003: artifacts() must return a dict."""
        run = _make_run()
        result = run.artifacts()
        assert isinstance(result, dict), (
            f"artifacts() must return a dict, got {type(result)}"
        )

    def test_metrics_returns_empty_dict_before_training(self) -> None:
        """ADR 003: metrics() returns {} before training has started."""
        run = _make_run()
        assert run.metrics() == {}, (
            "metrics() must return an empty dict when no training has started"
        )

    def test_artifacts_returns_empty_dict_when_none_available(self) -> None:
        """ADR 003: artifacts() returns {} when no artifacts are available yet."""
        run = _make_run()
        assert run.artifacts() == {}, (
            "artifacts() must return an empty dict when no artifacts are available"
        )

    def test_metrics_returns_copy_not_internal_dict(self) -> None:
        """ADR 003: metrics() must return a copy — not a reference to internal state."""
        run = _make_run()
        run._metrics["loss"] = 2.5
        result = run.metrics()
        result["loss"] = 999.0   # modify the returned copy
        # Internal state must be unchanged
        assert run._metrics["loss"] == 2.5, (
            "metrics() must return a copy, not the internal dict"
        )

    def test_artifacts_returns_copy_not_internal_dict(self) -> None:
        """ADR 003: artifacts() must return a copy — not a reference to internal state."""
        run = _make_run()
        run._artifacts["checkpoint"] = "pragma-encoder/checkpoints/pragma-s/ckpt.pt"
        result = run.artifacts()
        result["checkpoint"] = "tampered"
        assert run._artifacts["checkpoint"] == (
            "pragma-encoder/checkpoints/pragma-s/ckpt.pt"
        ), "artifacts() must return a copy, not the internal dict"


# ---------------------------------------------------------------------------
# Property tests — pipeline step invariants
# ---------------------------------------------------------------------------

class TestProperties:
    """Verify pipeline step structure and state invariants."""

    def test_default_run_has_five_steps(self) -> None:
        """ADR 003 / §2.4: a default PragmaRun must have exactly five pipeline steps."""
        run = _make_run()
        assert len(run.steps) == 5, (
            f"PragmaRun must have 5 pipeline steps, got {len(run.steps)}"
        )

    def test_default_steps_are_all_pending(self) -> None:
        """ADR 003: steps start in 'pending' status (no training has run yet)."""
        run = _make_run()
        for step in run.steps:
            assert step.status == "pending", (
                f"Step '{step.name}' must start as 'pending', got '{step.status}'"
            )

    def test_show_pipeline_prints_all_step_names(self, capsys) -> None:
        """ADR 003 / §2.4: show_pipeline() output must mention all five step names."""
        run = _make_run()
        run.show_pipeline()
        captured = capsys.readouterr()
        for name in PIPELINE_STEP_NAMES:
            assert name in captured.out, (
                f"show_pipeline() output must include step name '{name}'"
            )

    def test_show_pipeline_prints_step_statuses(self, capsys) -> None:
        """ADR 003: show_pipeline() must show the status of each step."""
        run = _make_run()
        run.steps[0].status = "completed"
        run.steps[1].status = "running"
        run.show_pipeline()
        captured = capsys.readouterr()
        assert "completed" in captured.out
        assert "running" in captured.out
        assert "pending" in captured.out

    def test_pipeline_step_rejects_invalid_name(self) -> None:
        """ADR 003: PipelineStep must reject names not in PIPELINE_STEP_NAMES."""
        with pytest.raises(ValueError, match="Unknown pipeline step name"):
            PipelineStep(name="nonexistent-step", description="test")

    def test_pipeline_step_rejects_invalid_status(self) -> None:
        """ADR 003: PipelineStep must reject status values not in the valid set."""
        with pytest.raises(ValueError, match="Unknown step status"):
            PipelineStep(
                name="prepare",
                description="test",
                status="not-a-real-status",
            )

    def test_run_carries_manifest(self) -> None:
        """ADR 003: PragmaRun must carry the DatasetManifest used for training."""
        manifest = _make_manifest()
        run = PragmaRun(manifest=manifest)
        assert run.manifest is manifest, (
            "PragmaRun must carry the exact manifest passed in"
        )

    def test_artifacts_checkpoint_key_when_set(self) -> None:
        """ADR 003: when a checkpoint is available, artifacts()['checkpoint'] is set."""
        run = _make_run()
        expected = "pragma-encoder/checkpoints/pragma-s/checkpoint_epoch0001.pt"
        run._artifacts["checkpoint"] = expected
        assert run.artifacts()["checkpoint"] == expected

    def test_kfp_run_url_shown_when_available(self, capsys) -> None:
        """ADR 003: show_pipeline() must print the KFP run URL when set."""
        run = _make_run()
        run.kfp_run_url = "https://kfp.example.com/runs/abc123"
        run.show_pipeline()
        captured = capsys.readouterr()
        assert "https://kfp.example.com/runs/abc123" in captured.out, (
            "show_pipeline() must include the KFP run URL when kfp_run_url is set"
        )

    def test_kfp_run_url_not_shown_when_none(self, capsys) -> None:
        """ADR 003: show_pipeline() must not mention KFP when kfp_run_url is None."""
        run = _make_run()
        assert run.kfp_run_url is None
        run.show_pipeline()
        captured = capsys.readouterr()
        assert "kfp.example.com" not in captured.out


# ---------------------------------------------------------------------------
# Specification tests — §2.4 pipeline stage names
# ---------------------------------------------------------------------------

class TestPaperSpecifications:
    """Verify pipeline step names match §2.4 training infrastructure stages."""

    def test_five_pipeline_step_names_defined(self) -> None:
        """§2.4 / ADR 003: exactly five canonical pipeline step names must be defined."""
        assert len(PIPELINE_STEP_NAMES) == 5, (
            f"§2.4 has five training infrastructure stages; "
            f"PIPELINE_STEP_NAMES must have 5 entries, got {len(PIPELINE_STEP_NAMES)}"
        )

    def test_prepare_step_exists(self) -> None:
        """§2.4 / ADR 003: 'prepare' step covers DatasetAdapter.prepare() — data storage."""
        assert "prepare" in PIPELINE_STEP_NAMES

    def test_upload_step_exists(self) -> None:
        """§2.4 / ADR 003: 'upload' step covers S3 upload (idempotent)."""
        assert "upload" in PIPELINE_STEP_NAMES

    def test_submit_step_exists(self) -> None:
        """§2.4 / ADR 003: 'submit' step covers PyTorchJob submission."""
        assert "submit" in PIPELINE_STEP_NAMES

    def test_train_step_exists(self) -> None:
        """§2.4 / ADR 003: 'train' step covers pretraining execution (MEM objective §2.3.5)."""
        assert "train" in PIPELINE_STEP_NAMES

    def test_export_step_exists(self) -> None:
        """§2.4 / ADR 003: 'export' step covers model output upload to S3."""
        assert "export" in PIPELINE_STEP_NAMES

    def test_step_order_matches_execution_order(self) -> None:
        """§2.4 / ADR 003: pipeline steps must be ordered by execution sequence.

        prepare → upload → submit → train → export
        This is the causal order: you cannot train before uploading data,
        and cannot export before training.
        """
        assert list(PIPELINE_STEP_NAMES) == [
            "prepare", "upload", "submit", "train", "export"
        ], (
            f"Pipeline steps must be in execution order "
            f"[prepare, upload, submit, train, export], "
            f"got {list(PIPELINE_STEP_NAMES)}"
        )

    def test_run_mode_default_is_none(self) -> None:
        """ADR 003: run_mode must default to None for runs created without a mode."""
        run = _make_run()
        assert run.run_mode is None, (
            "PragmaRun.run_mode must default to None when not set explicitly"
        )

    def test_run_mode_dry_run_shows_banner(self, capsys) -> None:
        """ADR 003: show_pipeline() must print a DRY RUN banner when run_mode='dry_run'."""
        run = _make_run()
        run.run_mode = "dry_run"
        run.show_pipeline()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out, (
            "show_pipeline() must include 'DRY RUN' when run_mode='dry_run'"
        )

    def test_run_mode_none_does_not_show_dry_run_banner(self, capsys) -> None:
        """ADR 003: show_pipeline() must not show DRY RUN banner when run_mode is None."""
        run = _make_run()
        assert run.run_mode is None
        run.show_pipeline()
        captured = capsys.readouterr()
        assert "DRY RUN" not in captured.out, (
            "show_pipeline() must not show DRY RUN banner when run_mode is None"
        )

    def test_run_mode_dry_run_does_not_show_submitted_status(self, capsys) -> None:
        """ADR 003: show_pipeline() with run_mode='dry_run' must not imply job submitted."""
        run = _make_run()
        run.run_mode = "dry_run"
        run.show_pipeline()
        captured = capsys.readouterr()
        # All steps are pending; none should show running or completed
        assert "running" not in captured.out
        assert "completed" not in captured.out

    def test_pragma_s_checkpoint_uri_format(self) -> None:
        """§2.4 / openshift-storage-pattern.md: PRAGMA-S checkpoint S3 path format.

        From openshift-storage-pattern.md:
            pragma-encoder/checkpoints/pragma-s/checkpoint_epoch<NNNN>.pt
        """
        # Verify the expected format is expressible and follows the convention
        checkpoint_uri = "pragma-encoder/checkpoints/pragma-s/checkpoint_epoch0001.pt"
        assert checkpoint_uri.startswith("pragma-encoder/checkpoints/pragma-s/")
        assert checkpoint_uri.endswith(".pt")
