"""Example 04 — Local training smoke test for PRAGMA workbench.

Runs PRAGMA-S pretraining locally using scripts/train_pragma.py as a
subprocess with a tiny fixture CSV (3 customers, 9 transactions).

This is real local execution — not a dry_run preview, not a cluster job.
The training script runs in a single process (--num-workers 0) for one
step (--max-steps 1) so the smoke test completes quickly.

Use this to verify:
  - The local workbench path end-to-end
  - IBMTabFormerAdapter.prepare() works with a small CSV
  - scripts/train_pragma.py starts and exits cleanly
  - PragmaRun.show_pipeline() correctly labels local execution

Run from the repo root:
    PYTHONPATH=. .venv/bin/python examples/workbench/04_local_training_smoke.py

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path when run directly
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

from src.workbench._api import train_pragma  # noqa: E402

_CSV_PATH   = "tests/fixtures/ibm_tabformer_tiny.csv"
_OUTPUT_DIR = "/tmp/pragma-local-smoke"

print("=" * 60)
print("PRAGMA Workbench — Local Training Smoke Test")
print("=" * 60)
print(f"  Dataset CSV : {_CSV_PATH}")
print(f"  Output dir  : {_OUTPUT_DIR}")
print(f"  Max steps   : 1 (smoke — exits after one optimizer step)")
print()

run = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",
    epochs=1,
    mode="local",
    local_csv_path=_CSV_PATH,
    output_dir=_OUTPUT_DIR,
    max_steps=1,
)

print()
run.show_pipeline()
print()
print("Artifacts:", run.artifacts())
print()

# Verify the train step completed (subprocess exited 0)
train_step = next(s for s in run.steps if s.name == "train")
if train_step.status == "completed":
    print("Smoke test PASSED — local training completed successfully.")
    sys.exit(0)
else:
    print(f"Smoke test FAILED — train step status: {train_step.status!r}")
    print("Check the training output above for errors.")
    sys.exit(1)
