"""Tests for pragma_encoder.evaluation.plot_loss.

Verifies:
  - load_step_metrics reads step-level records correctly
  - load_step_metrics skips epoch-end and checkpoint event records
  - load_step_metrics raises FileNotFoundError for missing file
  - load_step_metrics raises ValueError for empty / no step-level records
  - plot_loss produces a non-empty PNG file
  - plot_loss CLI (main) works end-to-end via argparse
  - plot_loss skips gracefully when matplotlib is absent

No GPU, no training, no S3 required.
All tests use synthetic metrics.jsonl written to tmp_path.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from pragma_encoder.evaluation.plot_loss import load_step_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_metrics(path: pathlib.Path, records: list[dict]) -> None:
    """Write a list of dicts to a metrics.jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _step_record(step: int, epoch: int = 1, loss: float = 4.5) -> dict:
    return {
        "step": step,
        "epoch": epoch,
        "train_loss": loss,
        "learning_rate": 1e-4,
        "timestamp": "2026-05-25T12:00:00+00:00",
        "checkpoint_path": None,
    }


def _epoch_end_record(step: int, epoch: int, avg_loss: float = 4.0) -> dict:
    return {
        "step": step,
        "epoch": epoch,
        "event": "epoch_end",
        "avg_train_loss": avg_loss,
        "timestamp": "2026-05-25T12:00:00+00:00",
    }


def _checkpoint_record(step: int, epoch: int, path: str = "/tmp/ckpt.pt") -> dict:
    return {
        "step": step,
        "epoch": epoch,
        "event": "checkpoint",
        "checkpoint_path": path,
        "timestamp": "2026-05-25T12:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# TestLoadStepMetrics
# ---------------------------------------------------------------------------


class TestLoadStepMetrics:
    """Verify load_step_metrics reads step records and skips event records."""

    def test_reads_step_level_records(self, tmp_path: pathlib.Path) -> None:
        """load_step_metrics must return (steps, losses) from step-level records."""
        metrics_path = tmp_path / "metrics.jsonl"
        _write_metrics(metrics_path, [
            _step_record(1, loss=5.0),
            _step_record(2, loss=4.5),
            _step_record(3, loss=4.0),
        ])
        steps, losses = load_step_metrics(metrics_path)
        assert steps == [1, 2, 3], f"Expected steps [1, 2, 3], got {steps}"
        assert losses == [5.0, 4.5, 4.0], f"Expected losses [5.0, 4.5, 4.0], got {losses}"

    def test_skips_epoch_end_records(self, tmp_path: pathlib.Path) -> None:
        """load_step_metrics must skip epoch_end event records."""
        metrics_path = tmp_path / "metrics.jsonl"
        _write_metrics(metrics_path, [
            _step_record(1, loss=5.0),
            _epoch_end_record(step=1, epoch=1, avg_loss=5.0),
            _step_record(2, loss=4.5),
        ])
        steps, losses = load_step_metrics(metrics_path)
        assert steps == [1, 2], f"Epoch-end records must be skipped. Got steps: {steps}"
        assert len(losses) == 2

    def test_skips_checkpoint_records(self, tmp_path: pathlib.Path) -> None:
        """load_step_metrics must skip checkpoint event records."""
        metrics_path = tmp_path / "metrics.jsonl"
        _write_metrics(metrics_path, [
            _step_record(1, loss=5.0),
            _checkpoint_record(step=1, epoch=1),
            _step_record(2, loss=4.5),
        ])
        steps, losses = load_step_metrics(metrics_path)
        assert steps == [1, 2], f"Checkpoint records must be skipped. Got steps: {steps}"
        assert len(losses) == 2

    def test_skips_blank_lines(self, tmp_path: pathlib.Path) -> None:
        """load_step_metrics must ignore blank lines in the file."""
        metrics_path = tmp_path / "metrics.jsonl"
        metrics_path.write_text(
            json.dumps(_step_record(1, loss=3.0)) + "\n"
            "\n"
            "\n"
            + json.dumps(_step_record(2, loss=2.5)) + "\n"
        )
        steps, losses = load_step_metrics(metrics_path)
        assert steps == [1, 2]
        assert losses == [3.0, 2.5]

    def test_raises_file_not_found(self, tmp_path: pathlib.Path) -> None:
        """load_step_metrics must raise FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError, match="metrics.jsonl not found"):
            load_step_metrics(tmp_path / "nonexistent.jsonl")

    def test_raises_value_error_for_no_step_records(self, tmp_path: pathlib.Path) -> None:
        """load_step_metrics must raise ValueError when no step-level records exist."""
        metrics_path = tmp_path / "metrics.jsonl"
        _write_metrics(metrics_path, [
            _epoch_end_record(step=10, epoch=1),
            _checkpoint_record(step=10, epoch=1),
        ])
        with pytest.raises(ValueError, match="No step-level training records"):
            load_step_metrics(metrics_path)

    def test_raises_value_error_for_empty_file(self, tmp_path: pathlib.Path) -> None:
        """load_step_metrics must raise ValueError for a completely empty file."""
        metrics_path = tmp_path / "metrics.jsonl"
        metrics_path.write_text("")
        with pytest.raises(ValueError, match="No step-level training records"):
            load_step_metrics(metrics_path)


# ---------------------------------------------------------------------------
# TestPlotLoss
# ---------------------------------------------------------------------------


class TestPlotLoss:
    """Verify plot_loss generates a valid PNG from synthetic metrics.jsonl."""

    @pytest.fixture()
    def metrics_file(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Write a synthetic 5-step metrics.jsonl."""
        p = tmp_path / "metrics.jsonl"
        records = [_step_record(i, loss=5.0 - i * 0.3) for i in range(1, 6)]
        records.append(_epoch_end_record(step=5, epoch=1, avg_loss=3.5))
        records.append(_checkpoint_record(step=5, epoch=1))
        _write_metrics(p, records)
        return p

    def test_plot_loss_produces_png(
        self, metrics_file: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """plot_loss must produce a non-empty PNG file."""
        pytest.importorskip("matplotlib", reason="matplotlib not installed")
        from pragma_encoder.evaluation.plot_loss import plot_loss  # noqa: PLC0415
        output = tmp_path / "loss.png"
        plot_loss(metrics_path=metrics_file, output_path=output)
        assert output.exists(), f"loss.png was not created at {output}"
        assert output.stat().st_size > 0, "loss.png is empty"

    def test_plot_loss_creates_parent_dirs(
        self, metrics_file: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """plot_loss must create parent directories of the output path."""
        pytest.importorskip("matplotlib", reason="matplotlib not installed")
        from pragma_encoder.evaluation.plot_loss import plot_loss  # noqa: PLC0415
        output = tmp_path / "subdir" / "deep" / "loss.png"
        assert not output.parent.exists()
        plot_loss(metrics_path=metrics_file, output_path=output)
        assert output.exists(), f"loss.png was not created at {output}"

    def test_plot_loss_cli_end_to_end(
        self, metrics_file: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """plot_loss main() must work via CLI args and exit 0."""
        pytest.importorskip("matplotlib", reason="matplotlib not installed")
        import subprocess  # noqa: PLC0415
        output = tmp_path / "loss_cli.png"
        result = subprocess.run(
            [
                sys.executable, "-m", "pragma_encoder.evaluation.plot_loss",
                "--metrics", str(metrics_file),
                "--output", str(output),
                "--title", "Test Loss Curve",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"plot_loss CLI exited with code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert output.exists(), "plot_loss CLI did not produce loss.png"
        assert "Loss curve written" in result.stdout, (
            f"Expected 'Loss curve written' in stdout. Got: {result.stdout!r}"
        )

    def test_plot_loss_raises_file_not_found(self, tmp_path: pathlib.Path) -> None:
        """plot_loss must raise FileNotFoundError for a missing metrics file."""
        pytest.importorskip("matplotlib", reason="matplotlib not installed")
        from pragma_encoder.evaluation.plot_loss import plot_loss  # noqa: PLC0415
        with pytest.raises(FileNotFoundError):
            plot_loss(
                metrics_path=tmp_path / "nonexistent.jsonl",
                output_path=tmp_path / "loss.png",
            )
