"""Example 05 — Decorator-based PRAGMA pipeline authoring.

Demonstrates the @pragma_pipeline decorator DSL introduced in ADR 004.
Data scientists express training intent in normal Python. The decorated
object captures that intent and can compile it to a KFP v2 pipeline YAML
for submission to OpenShift Pipelines.

Two authoring paths are shown:

  Path A — @pragma_pipeline decorator (recommended for notebooks):

      @pragma_pipeline(name="pragma-s-ibm-tabformer")
      def run():
          ds = dataset("ibm-tabformer", prepare_if_missing=True)
          train(dataset=ds, model_size="S", epochs=1, max_steps=1)

      run.show_pipeline()
      run.compile("pipeline/generated/pragma-s-ibm-tabformer.yaml")

  Path B — train_pragma(mode="pipeline") helper (keyword-argument style):

      run = train_pragma(
          dataset="ibm-tabformer",
          model_size="S",
          epochs=1,
          mode="pipeline",
          max_steps=1,
      )
      run.show_pipeline()

Key properties of both paths (ADR 004 constraints):
  - No training, no S3, no cluster access at authoring time
  - compile() requires kfp; show_pipeline() works without kfp
  - submit() raises NotImplementedError until explicitly implemented
  - dry_run and local modes are unchanged

Run from the repo root:
    PYTHONPATH=. .venv/bin/python examples/workbench/05_decorated_pipeline.py

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/004-workbench-decorated-pipelines.md
"""

import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path when run directly
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from pragma_encoder.workbench import (  # noqa: E402
    pragma_pipeline,
    dataset,
    train,
    train_pragma,
)

# ---------------------------------------------------------------------------
# Path A: @pragma_pipeline decorator
# ---------------------------------------------------------------------------

print("=" * 60)
print("PRAGMA Workbench — Decorator Pipeline Authoring (ADR 004)")
print("=" * 60)
print()
print("Path A: @pragma_pipeline decorator")
print("-" * 40)


@pragma_pipeline(name="pragma-s-ibm-tabformer")
def run():
    ds = dataset("ibm-tabformer", prepare_if_missing=True)
    train(dataset=ds, model_size="S", epochs=1, max_steps=1)


print(f"Decorated object type : {type(run).__name__}")
print(f"Pipeline name         : {run.name}")
print(f"Stages                : {run.stages}")
print()
print("show_pipeline() output:")
print()
run.show_pipeline()
print()

# ---------------------------------------------------------------------------
# submit() is not yet implemented
# ---------------------------------------------------------------------------

print("submit() raises NotImplementedError (compile() first):")
try:
    run.submit()
except NotImplementedError as exc:
    print(f"  NotImplementedError: {exc}")
print()

# ---------------------------------------------------------------------------
# compile() — requires kfp
# ---------------------------------------------------------------------------

_KFP_AVAILABLE = importlib.util.find_spec("kfp") is not None

if _KFP_AVAILABLE:
    output = Path("pipeline/generated/pragma-s-ibm-tabformer.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"compile() — writing KFP YAML to {output}")
    run.compile(str(output))
    size = output.stat().st_size
    print(f"  Written: {output} ({size} bytes)")
else:
    print("compile() — skipped (kfp not installed)")
    print("  Install with: pip install -r requirements.txt")
    print("  Or use the prepared workbench image where kfp is pre-installed.")
print()

# ---------------------------------------------------------------------------
# Path B: train_pragma(mode="pipeline") keyword-argument style
# ---------------------------------------------------------------------------

print("Path B: train_pragma(mode='pipeline')")
print("-" * 40)

run2 = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",
    epochs=1,
    mode="pipeline",
    max_steps=1,
)

print(f"Return type           : {type(run2).__name__}")
print(f"Has compile()         : {callable(getattr(run2, 'compile', None))}")
print(f"Has show_pipeline()   : {callable(getattr(run2, 'show_pipeline', None))}")
print()
print("show_pipeline() output:")
print()
run2.show_pipeline()
print()

# ---------------------------------------------------------------------------
# dry_run is unchanged (ADR 004: existing modes are unaffected)
# ---------------------------------------------------------------------------

print("=" * 60)
print("dry_run and local modes are unchanged (ADR 004)")
print("=" * 60)
print()
print("train_pragma(mode='dry_run') — returns PragmaRun as before:")
from pragma_encoder.workbench._run import PragmaRun  # noqa: E402
dry = train_pragma(dataset="ibm-tabformer", model_size="S", mode="dry_run")
print(f"  type     : {type(dry).__name__}")
print(f"  run_mode : {dry.run_mode}")
print()
dry.show_pipeline()
print()
print("Done. See docs/decisions/004-workbench-decorated-pipelines.md for design rationale.")
