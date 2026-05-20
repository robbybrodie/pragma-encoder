"""PragmaRun — result object for a PRAGMA pretraining invocation.

Wraps a submitted (or completed) training job. Exposes:
  - show_pipeline(): print the five §2.4 pipeline steps with status
  - metrics():       return training metrics dict (loss, epoch, throughput)
  - artifacts():     return S3 artifact URI dict (checkpoint, model, vocab)

The five pipeline steps correspond to §2.4 training infrastructure:
  1. prepare  — Validate and prepare dataset (DatasetAdapter.prepare())
  2. upload   — Upload prepared artifacts to S3 (idempotent)
  3. submit   — Submit PyTorchJob to the cluster (or configure local run)
  4. train    — Pretraining execution (masked event modelling, §2.3.5)
  5. export   — Export model checkpoints and outputs to S3

This is an implementation decision (ADR 003). The paper does not specify a
Python result API — it describes training infrastructure (§2.4).

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.data.dataset_manifest import DatasetManifest

# ---------------------------------------------------------------------------
# The five §2.4 pipeline step names — canonical, must not drift
# ---------------------------------------------------------------------------

PIPELINE_STEP_NAMES: tuple[str, ...] = (
    "prepare",   # Validate and prepare dataset (DatasetAdapter.prepare())
    "upload",    # Upload prepared artifacts to S3 (idempotent)
    "submit",    # Submit PyTorchJob to cluster (or configure local run)
    "train",     # Pretraining execution (MEM objective §2.3.5)
    "export",    # Export model checkpoints and outputs to S3
)

STEP_DESCRIPTIONS: dict[str, str] = {
    "prepare": "Validate and prepare dataset (fit tokeniser)",
    "upload":  "Upload prepared artifacts to S3 (idempotent)",
    "submit":  "Submit PyTorchJob to cluster or configure local run",
    "train":   "Pretraining — masked event modelling (§2.3.5)",
    "export":  "Export model checkpoints and outputs to S3",
}

# Valid step status values
_VALID_STATUSES = {"pending", "running", "completed", "failed", "skipped"}


# ---------------------------------------------------------------------------
# PipelineStep — one step in the five-stage pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineStep:
    """One step in the PRAGMA training pipeline.

    Args:
        name:        Step identifier — one of PIPELINE_STEP_NAMES.
        description: Human-readable description of what this step does.
        status:      Current status: "pending", "running", "completed",
                     "failed", or "skipped".

    Reference: Ostroukhov et al. (2026), Section 2.4
    """

    name: str
    description: str
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.name not in PIPELINE_STEP_NAMES:
            raise ValueError(
                f"Unknown pipeline step name '{self.name}'. "
                f"Valid names: {PIPELINE_STEP_NAMES}"
            )
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"Unknown step status '{self.status}'. "
                f"Valid statuses: {_VALID_STATUSES}"
            )


# ---------------------------------------------------------------------------
# PragmaRunProtocol — interface contract
# ---------------------------------------------------------------------------

@runtime_checkable
class PragmaRunProtocol(Protocol):
    """Interface contract for the result of a train_pragma() invocation.

    A PragmaRun wraps a submitted or completed training job and provides
    three observation methods:
      - show_pipeline(): print the five pipeline steps with their status
      - metrics():       return training metrics (loss, epoch, throughput)
      - artifacts():     return S3 artifact URIs

    This is an implementation decision (ADR 003). The paper does not define
    a Python result API, but §2.4 describes the five infrastructure stages
    that show_pipeline() must surface.
    """

    def show_pipeline(self) -> None:
        """Print the five §2.4 pipeline steps with their current status.

        Output format (printed to stdout, not returned):
            [completed] prepare  — Validate and prepare dataset (fit tokeniser)
            [completed] upload   — Upload prepared artifacts to S3 (idempotent)
            [running  ] submit   — Submit PyTorchJob to cluster or configure local run
            [pending  ] train    — Pretraining — masked event modelling (§2.3.5)
            [pending  ] export   — Export model checkpoints and outputs to S3

        If KFP is available, also emits the KFP run URL.
        """
        ...

    def metrics(self) -> dict[str, Any]:
        """Return training metrics dict.

        Keys (present when training has started or completed):
            "epoch":      int   — current or final epoch number
            "loss":       float — latest training loss
            "step":       int   — latest global step

        Returns empty dict if training has not yet started.
        """
        ...

    def artifacts(self) -> dict[str, Any]:
        """Return S3 artifact URI dict.

        Keys (present when available):
            "checkpoint": str — S3 URI to latest checkpoint
            "model":      str — S3 URI to final model outputs directory
            "vocab":      str — S3 URI to fitted tokeniser vocab

        Returns empty dict if no artifacts are available yet.
        """
        ...


# ---------------------------------------------------------------------------
# PragmaRun — concrete implementation
# ---------------------------------------------------------------------------

@dataclass
class PragmaRun:
    """Result of a train_pragma() invocation.

    Wraps a submitted or completed training job. Exposes show_pipeline(),
    metrics(), and artifacts() for workbench inspection.

    Args:
        manifest:     DatasetManifest describing the prepared training dataset.
        steps:        List of PipelineStep objects, one per §2.4 stage.
        _metrics:     Training metrics dict (populated as training progresses).
        _artifacts:   S3 artifact URI dict (populated as artifacts are uploaded).
        kfp_run_url:  KFP run URL (None if KFP is not available).
        run_mode:     Execution mode that produced this run:
                        None          — unknown / created directly (default)
                        "dry_run"     — preview only; no job submitted
                        "cluster"     — submitted as a KFTO PyTorchJob
                        "local"       — executed as a local subprocess

    Reference: Ostroukhov et al. (2026), Section 2.4
    ADR: docs/decisions/003-workbench-training-api.md
    """

    manifest: DatasetManifest
    steps: list[PipelineStep] = field(default_factory=list)
    _metrics: dict[str, Any] = field(default_factory=dict, repr=False)
    _artifacts: dict[str, Any] = field(default_factory=dict, repr=False)
    kfp_run_url: str | None = None
    run_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            # Default: all five steps in "pending" state
            self.steps = [
                PipelineStep(name=name, description=STEP_DESCRIPTIONS[name])
                for name in PIPELINE_STEP_NAMES
            ]

    def show_pipeline(self) -> None:
        """Print the five §2.4 pipeline steps with their current status."""
        print("PRAGMA Training Pipeline")
        print("=" * 50)
        if self.run_mode == "dry_run":
            print("  [DRY RUN] Pipeline preview — no training job has been submitted.")
            print()
        for step in self.steps:
            status_label = f"[{step.status:<9}]"
            print(f"  {status_label} {step.name:<8} — {step.description}")
        if self.kfp_run_url:
            print(f"\nKFP run: {self.kfp_run_url}")

    def metrics(self) -> dict[str, Any]:
        """Return current training metrics dict."""
        return dict(self._metrics)

    def artifacts(self) -> dict[str, Any]:
        """Return current S3 artifact URI dict."""
        return dict(self._artifacts)
