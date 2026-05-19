"""Beginner workbench path — PRAGMA-S on IBM TabFormer data.

This script shows the simplest way to interact with the PRAGMA training API
from a workbench notebook or script. It uses mode="dry_run", which means:

  - No data is downloaded or prepared.
  - No job is submitted to any cluster.
  - No credentials or S3 bucket are required.
  - The returned PragmaRun is a *preview* of what would happen if you ran
    with mode="cluster" or mode="local".

Run from the project root:
    python examples/workbench/01_train_ibm_tabformer.py

Or from inside a JupyterLab cell — copy the code below the sys.path block.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
"""

# ── Project root on sys.path ──────────────────────────────────────────────────
# Needed when running as a plain script outside of the installed package.
# In a workbench pod, src/ is on PYTHONPATH already — this is a no-op.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Imports ───────────────────────────────────────────────────────────────────
from src.workbench import train_pragma

# ─────────────────────────────────────────────────────────────────────────────
# Step 1  —  Call train_pragma with mode="dry_run"
#
# This is safe to run anywhere: on your laptop, in a workbench, or in a CI
# job. It does no work and touches no external systems. The only effect is
# returning a PragmaRun object you can inspect.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("PRAGMA workbench — example 01: IBM TabFormer, single-node")
print("=" * 60)
print()
print("Calling train_pragma(mode='dry_run') ...")
print("  This returns a pipeline preview. No job is submitted.")
print()

run = train_pragma(
    dataset="ibm-tabformer",   # IBM TabFormer credit-card transactions
    model_size="S",            # PRAGMA-S, ~10M parameters (Table 1)
    epochs=10,
    prepare_if_missing=True,   # would fit the tokeniser if this were a real run
    mode="dry_run",            # preview only — no side effects
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2  —  Inspect the pipeline preview
#
# show_pipeline() prints all five §2.4 training stages with their current
# status. In dry_run mode every stage is "pending" and the banner clearly
# states that no training job has been submitted.
# ─────────────────────────────────────────────────────────────────────────────

run.show_pipeline()
print()

# ─────────────────────────────────────────────────────────────────────────────
# Step 3  —  Inspect the placeholder manifest
#
# The dry_run run carries a placeholder DatasetManifest. It reflects the
# dataset and config you requested, but all URIs are placeholders — no data
# has been prepared or uploaded.
# ─────────────────────────────────────────────────────────────────────────────

manifest = run.manifest
print("Dataset manifest (dry-run placeholder):")
print(f"  dataset_name         : {manifest.dataset_name}")
print(f"  dataset_version      : {manifest.dataset_version}  ← 'dry-run' confirms no data was prepared")
print(f"  prepared_prefix_uri  : {manifest.prepared_prefix_uri}")
print(f"  shards               : {len(manifest.shards)} placeholder shard(s)")
print(f"  config.d_model              : {manifest.config.d_model}  (embedding dimension, Table 1)")
print(f"  config.event_encoder_layers : {manifest.config.event_encoder_layers}  (Transformer layers in EventEncoder)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Step 4  —  Metrics and artifacts are empty in dry_run
#
# A real submitted run would populate these as training progresses.
# In dry_run they are always empty — no training happened.
# ─────────────────────────────────────────────────────────────────────────────

print("Metrics (empty — no training was run):", run.metrics())
print("Artifacts (empty — no outputs produced):", run.artifacts())
print()

# ─────────────────────────────────────────────────────────────────────────────
# Next steps
# ─────────────────────────────────────────────────────────────────────────────

print("-" * 60)
print("Next steps:")
print()
print("  To understand what each pipeline stage does, run:")
print("    python examples/workbench/02_understand_pipeline.py")
print()
print("  To see the two-node distributed training topology, run:")
print("    python examples/workbench/03_two_node_training_demo.py")
print()
print("  To submit a real training job (requires an OpenShift AI cluster):")
print("    Change mode='dry_run' to mode='cluster' and ensure your")
print("    pragma-workbench-env Secret is configured with valid S3 credentials.")
print("    The 'cluster' mode is not yet implemented — it will raise")
print("    NotImplementedError until the cluster dispatch component is added.")
print("-" * 60)
