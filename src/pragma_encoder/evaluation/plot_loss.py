"""Loss curve plotting utility for PRAGMA pretraining runs.

Reads a ``metrics.jsonl`` file written by ``pragma_encoder.training.train``
and produces a PNG loss curve.  The JSONL file contains one JSON object per
line; each step-level record has at minimum ``step`` and ``train_loss`` keys.

Usage::

    python -m pragma_encoder.evaluation.plot_loss \\
        --metrics outputs/pragma-s/metrics.jsonl \\
        --output  outputs/pragma-s/loss.png

The output PNG is also written by the ``run_pretraining`` KFP pipeline
component after training completes, and is published to S3 alongside the
model checkpoint.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_step_metrics(metrics_path: Path) -> tuple[list[int], list[float]]:
    """Read step-level (step, train_loss) pairs from metrics.jsonl.

    Skips epoch-end and checkpoint event records (they lack ``train_loss``).

    Args:
        metrics_path: Path to a metrics.jsonl file written by train.py.

    Returns:
        Tuple of (steps, losses) — two parallel lists suitable for plotting.

    Raises:
        FileNotFoundError: If metrics_path does not exist.
        ValueError:        If no step-level records are found.
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found: {metrics_path}")

    steps: list[int] = []
    losses: list[float] = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Step-level records have "train_loss" and no "event" key
            if "train_loss" in record and "event" not in record:
                steps.append(int(record["step"]))
                losses.append(float(record["train_loss"]))

    if not steps:
        raise ValueError(
            f"No step-level training records found in {metrics_path}. "
            "Ensure training ran at least one gradient step."
        )
    return steps, losses


def plot_loss(
    metrics_path: Path,
    output_path: Path,
    title: str = "PRAGMA Pretraining Loss",
) -> None:
    """Generate and save a loss curve PNG from a metrics.jsonl file.

    Args:
        metrics_path: Path to metrics.jsonl produced by train.py.
        output_path:  Destination path for the PNG file.
        title:        Plot title.  Default: "PRAGMA Pretraining Loss".

    Raises:
        FileNotFoundError: If metrics_path does not exist.
        ValueError:        If no step-level records are found.
        ImportError:       If matplotlib is not installed.
    """
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")  # non-interactive backend — safe in container/CI
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_loss. "
            "Install it with: pip install matplotlib"
        ) from exc

    steps, losses = load_step_metrics(metrics_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, linewidth=1.0, color="#1f77b4", alpha=0.9)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Train Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for loss curve generation.

    Example::

        python -m pragma_encoder.evaluation.plot_loss \\
            --metrics outputs/pragma-s/metrics.jsonl \\
            --output  outputs/pragma-s/loss.png
    """
    parser = argparse.ArgumentParser(
        description="Generate a loss curve PNG from a PRAGMA metrics.jsonl file."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics.jsonl written by pragma_encoder.training.train.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the output PNG file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="PRAGMA Pretraining Loss",
        help="Plot title. Default: 'PRAGMA Pretraining Loss'.",
    )
    args = parser.parse_args(argv)

    plot_loss(
        metrics_path=args.metrics,
        output_path=args.output,
        title=args.title,
    )
    print(f"Loss curve written -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
