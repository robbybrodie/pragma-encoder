"""Two-node distributed training topology — preview and explanation.

This script demonstrates the two-node (nodes=2) PRAGMA training configuration.
It uses mode="dry_run" by default, which means no job is submitted and no
credentials are required.

What this example covers:
  - How to request a two-node distributed training run
  - What "two-node" means in terms of data flow and PyTorch DDP
  - What is fully supported for fresh (first-run) two-node training
  - What is NOT yet supported: checkpoint resume across pods (TD-006)

Run from the project root:
    python examples/workbench/03_two_node_training_demo.py

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
PyTorchJob manifest: openshift/training/pytorchjob-pragma-s-2node.yaml
Tech debt: docs/tech-debt.md TD-006
"""

# ── Project root on sys.path ──────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Imports ───────────────────────────────────────────────────────────────────
from tools.workbench import train_pragma

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
#
# Change MODE to "cluster" to submit a real job (requires an OpenShift AI
# cluster with pragma-workbench-env Secret configured). The cluster mode is
# not yet implemented and will raise NotImplementedError.
# ─────────────────────────────────────────────────────────────────────────────

MODE = "dry_run"   # safe default — change to "cluster" only on the cluster

print("=" * 60)
print("PRAGMA workbench — example 03: two-node distributed training")
print("=" * 60)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Two-node topology explained
#
# Setting nodes=2 selects the two-node PyTorchJob manifest:
#   openshift/training/pytorchjob-pragma-s-2node.yaml
#
# This creates two pods:
#   Master  — rank 0 — runs torchrun, handles checkpointing and S3 upload
#   Worker  — rank 1 — runs torchrun, participates in gradient averaging
#
# WORLD_SIZE is set to 2 by torchrun (nnodes=2, nproc_per_node=1).
# The training entrypoint (python -m pragma_encoder.training.train) detects WORLD_SIZE > 1 and:
#   - Initialises an NCCL process group across the two pods
#   - Uses DistributedSampler so each pod processes a disjoint data shard
#   - Wraps model and assembler in DistributedDataParallel (DDP)
#   - Gates all logging, checkpointing, and S3 uploads to rank 0 only
#   - Synchronises all ranks with dist.barrier() between epochs
# ─────────────────────────────────────────────────────────────────────────────

print("Requesting two-node training preview (mode='dry_run') ...")
print()

run = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",            # PRAGMA-S (~10M params); use "M" or "L" for larger runs
    epochs=10,
    nodes=2,                   # ← two-node DDP topology
    prepare_if_missing=True,
    mode=MODE,
)

run.show_pipeline()
print()

# ─────────────────────────────────────────────────────────────────────────────
# What is fully supported for two-node fresh training
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Two-node training — supported on fresh runs")
print("=" * 60)
print("""
The following are fully implemented and work correctly when nodes=2:

  Data sharding
    Each pod downloads the full training dataset from S3 independently
    (each pod has its own emptyDir /workspace with no shared filesystem).
    DistributedSampler assigns each pod a disjoint subset of customers,
    so the total effective batch size is batch_size × WORLD_SIZE.

  NCCL gradient averaging
    Gradients are averaged across both pods after each backward pass via
    DistributedDataParallel. Both pods converge on the same model weights.

  Consistent rendezvous
    The Kubeflow Training Operator (KFTO) injects MASTER_ADDR (master pod
    hostname) and MASTER_PORT into every replica. torchrun uses these to
    establish the rendezvous so both pods join the same process group.

  Rank-0-only checkpointing
    Only the Master pod (rank 0) writes checkpoints to disk and uploads
    them to S3. The Worker pod never writes to S3. There is no race
    condition between the two pods on S3 writes.

  Epoch synchronisation
    dist.barrier() is called after each epoch so neither pod begins the
    next epoch before both have finished the current one.
""")

# ─────────────────────────────────────────────────────────────────────────────
# What is NOT supported — TD-006
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Known limitation — TD-006: checkpoint resume across pods")
print("=" * 60)
print("""
IMPORTANT: Resuming a two-node job from a checkpoint is NOT correctly
implemented for the emptyDir storage pattern.

  What happens on --resume with emptyDir:
    1. Rank 0 downloads the checkpoint from S3 to its own pod's emptyDir.
    2. The training script broadcasts "checkpoint found" to all ranks.
    3. Worker rank 1 looks for the checkpoint in its own pod's emptyDir.
       Its emptyDir is empty. It finds nothing.
    4. Rank 0 loads checkpoint weights; rank 1 starts from random weights.
    5. Both pods proceed through the same DDP training loop but with
       diverged starting model states.

  Impact:
    On a genuine first run (no prior checkpoint) --resume is a no-op for
    all ranks and training proceeds correctly. The issue only occurs when
    a checkpoint was saved on a prior run and the job is restarted.

    The two-node demo manifest uses --max-steps 20. In 20 steps it is
    unlikely that a full epoch completes before the job exits, so no
    checkpoint is written and --resume remains a no-op in practice.

  Resolution (TD-006, not yet implemented):
    Replace the "rank 0 downloads; workers look locally" pattern with
    "all ranks download from S3 independently." Each pod would fetch the
    same checkpoint key from S3 at startup, giving all ranks the same
    starting state.

    See docs/tech-debt.md TD-006 for the full description and resolution
    options.

Do NOT remove the --resume flag from a two-node job manifest and assume
this is equivalent to fixing TD-006. The flag is harmless on first runs
and required for any future fault-tolerant implementation.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Comparing single-node and two-node manifests
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Single-node vs. two-node — key differences")
print("=" * 60)
print("""
  Feature                  nodes=1                  nodes=2
  ───────────────────────  ───────────────────────  ───────────────────────
  PyTorchJob spec          Master only              Master + Worker
  Pods created             1                        2
  GPUs used                1                        2
  WORLD_SIZE               1 (no DDP)               2 (DDP via NCCL)
  Data loading             standard DataLoader      DistributedSampler
  Model wrapper            plain nn.Module          DistributedDataParallel
  Checkpointing            rank 0 (always rank 0)   rank 0 (Master pod)
  S3 upload                Master pod               Master pod (rank 0)
  Resume from checkpoint   supported                NOT YET SUPPORTED (TD-006)
  Manifest                 pytorchjob-pragma-s.yaml pytorchjob-pragma-s-2node.yaml
""")

# ─────────────────────────────────────────────────────────────────────────────
# How to submit a real two-node job (when cluster mode is implemented)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("How to submit a real two-node job")
print("=" * 60)
print("""
When the cluster dispatch component is implemented, submitting a two-node
job from the workbench will be:

    run = train_pragma(
        dataset="ibm-tabformer",
        model_size="S",
        epochs=10,
        nodes=2,
        mode="cluster",   # requires OpenShift AI cluster + configured Secret
    )
    run.show_pipeline()

Alternatively, apply the manifest directly:

    oc apply -f openshift/training/pytorchjob-pragma-s-2node.yaml -n pragma-encoder
    oc get pytorchjob pragma-s-pretrain-2node -n pragma-encoder

Prerequisites before submitting:
  1. Upload training data to S3 (scripts/upload_training_data.py)
  2. Ensure pragma-workbench-env Secret is configured in pragma-encoder namespace
  3. Ensure KFTO (Kubeflow Training Operator) is installed on the cluster
  4. Ensure at least two GPU nodes are available

The 'cluster' mode raises NotImplementedError until the dispatch component
is added. This is by design — mode='dry_run' is the safe default.
""")

print(f"Script completed. MODE was: {MODE!r}")
print("No job was submitted. No credentials were required.")
