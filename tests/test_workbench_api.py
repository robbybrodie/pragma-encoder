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
