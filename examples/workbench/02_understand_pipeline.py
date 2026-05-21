"""Understand the PRAGMA training pipeline — five stages explained.

This script walks through each of the five §2.4 training stages and explains
what each one does in data-scientist terms. It uses mode="dry_run" throughout,
so nothing is submitted or executed.

The five stages are:
  1. prepare  — fit the tokeniser and build a DatasetManifest
  2. upload   — move prepared artefacts to S3
  3. submit   — configure and launch the training job
  4. train    — masked event modelling pretraining (§2.3.5)
  5. export   — write the final model checkpoint to S3

Run from the project root:
    python examples/workbench/02_understand_pipeline.py

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
"""

# ── Project root on sys.path ──────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Imports ───────────────────────────────────────────────────────────────────
from tools.openshift_ai.workbench import train_pragma, PIPELINE_STEP_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Obtain a dry-run PragmaRun to inspect
# ─────────────────────────────────────────────────────────────────────────────

run = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",
    epochs=10,
    mode="dry_run",
)

print("=" * 60)
print("PRAGMA training pipeline — five stages explained")
print("=" * 60)
print()
print("Pipeline status (all pending in dry_run mode):")
print()
run.show_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# Stage-by-stage explanation
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("What each stage does")
print("=" * 60)

print("""
Stage 1 — prepare
─────────────────
The prepare stage converts raw transaction data into the format that PRAGMA
expects. Specifically it:

  * Fits a FinancialTokenizerPipeline on 80% of the transaction data.
    This learns the percentile buckets for numerical fields (amounts,
    balances) and the categorical vocabulary for fields like merchant
    category codes and currency.

  * Tokenises every transaction into a (key, value, time) triple per field,
    following the key-value-time representation from §2.2 of the paper.

  * Writes the fitted vocabulary to vocab.pkl and the prepared data to
    one or more CSV/Parquet shards.

  * Returns a DatasetManifest — a small metadata object that records where
    the prepared data lives, how many rows it has, and which PRAGMAConfig
    was used (so tokenisation and training are always consistent).

The DatasetManifest is the canonical contract for all downstream stages.
No stage after prepare reads raw transaction data directly.
""")

print("""
Stage 2 — upload
────────────────
The upload stage copies the prepared data and vocab.pkl from local storage
(or a local path) to S3. This is an idempotent operation — re-running it
does not corrupt or duplicate existing data.

S3 is the durable store for all PRAGMA artefacts:
  Input data  : s3://<bucket>/pragma-encoder/data/<dataset>/
  Checkpoints : s3://<bucket>/pragma-encoder/checkpoints/<variant>/
  Outputs     : s3://<bucket>/pragma-encoder/outputs/<variant>/

Why S3 and not a Kubernetes persistent volume (PVC)?
  PVCs with ReadWriteOnce (RWO) access mode can only be mounted by one
  node at a time. That would force the training job to run on a specific
  node, regardless of where GPU capacity is available. S3 has no such
  constraint — any node can read from it, so the scheduler can place
  training pods freely.

S3 credentials come from the pragma-workbench-env Secret. You do not
set bucket names or keys in this script.
""")

print("""
Stage 3 — submit
────────────────
The submit stage creates a training job on the cluster. PRAGMA uses
Kubeflow Training Operator (KFTO) PyTorchJobs — a Kubernetes resource
that manages one or more PyTorch training processes.

For a single-node run (nodes=1):
  One pod is created. It runs scripts/train_pragma.py directly.

For a two-node run (nodes=2):
  Two pods are created: Master (rank 0) and Worker (rank 1).
  They communicate via NCCL over the cluster network.
  Each pod downloads a copy of the training data from S3 at startup
  (no shared filesystem between pods).

The submit stage does not wait for training to complete. It configures
the job and hands off to the train stage.
""")

print("""
Stage 4 — train
───────────────
This is where PRAGMA actually learns.

PRAGMA uses Masked Event Modelling (MEM, §2.3.5) — a BERT-style
pretraining objective adapted for transaction sequences. Three masking
strategies are applied at random:

  Token masking  — mask individual field value tokens within an event
  Field masking  — mask all tokens for a specific field across events
  Event masking  — mask all tokens for a complete transaction event

For each masked position the model predicts the original token. This
forces the model to learn:
  * The statistical distribution of field values given context
  * Relationships between fields within a transaction
  * Temporal patterns across a customer's transaction history

The training loop runs for the requested number of epochs. Checkpoints
are written to /workspace/outputs/ on the pod and then uploaded to S3
after each epoch, so training can resume from a checkpoint if the pod
is interrupted.

All data loading, masking, and checkpointing is handled by
scripts/train_pragma.py. The submit stage is what launches it.
""")

print("""
Stage 5 — export
────────────────
After training completes, the export stage copies the final checkpoint
and vocabulary from the training pod's local storage to the canonical
S3 export path:

  s3://<bucket>/pragma-encoder/checkpoints/pragma-s/checkpoint_epoch<NNNN>.pt

This is the model that downstream tasks (linear probe, LoRA fine-tuning,
embedding extraction) load via the adaptation API:

  from pragma_encoder.adaptation.probe import EmbeddingProbe
  from pragma_encoder.adaptation.lora import LoRAAdapter

The export stage is a no-op if the training job already uploaded
checkpoints to S3 during training (which train_pragma.py does for
each epoch via --s3-checkpoint-prefix). It exists as an explicit
pipeline stage to signal that the model is ready for downstream use
and to support future post-training cleanup steps.
""")

# ─────────────────────────────────────────────────────────────────────────────
# DatasetManifest — a closer look
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("The DatasetManifest — canonical training data contract")
print("=" * 60)
print()

manifest = run.manifest
print("Fields on a DatasetManifest:")
print(f"  dataset_name        : {manifest.dataset_name!r}")
print(f"  dataset_version     : {manifest.dataset_version!r}  (set by the adapter; 'dry-run' here)")
print(f"  prepared_prefix_uri : {manifest.prepared_prefix_uri!r}")
print(f"  shards              : {manifest.shards}")
print(f"  vocab_uri           : {manifest.vocab_uri}  (S3 path to vocab.pkl after upload)")
print(f"  manifest_uri        : {manifest.manifest_uri}  (S3 path to the manifest JSON)")
print(f"  row_count           : {manifest.row_count}  (total transactions; set after prepare)")
print()
print("The config embedded in the manifest fixes tokenisation to the model size:")
print(f"  max_event_tokens : {manifest.config.max_event_tokens}  (max tokens per event)")
print(f"  max_events       : {manifest.config.max_events}  (max events per customer)")
print(f"  d_model          : {manifest.config.d_model}  (embedding dimension)")
print()
print("The manifest is the only object the train stage needs. If you already")
print("have a prepared dataset in S3, use pragma_train_from_manifest_pipeline(manifest_uri=...)")
print("to run stages 3-5 only (submit -> train -> export), skipping prepare and upload.")
print("The full pragma_pretraining_pipeline always runs all five stages.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline step names — used consistently throughout the codebase
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Stage names are consistent across workbench, pipeline, and PyTorchJob")
print("=" * 60)
print()
print("PIPELINE_STEP_NAMES:", PIPELINE_STEP_NAMES)
print()
print("These same names appear in:")
print("  tools/openshift_ai/workbench/_run.py — PragmaRun.show_pipeline()")
print("  pipeline/components_pragma.py        — PIPELINE_STAGE_NAMES")
print("  pipeline/pragma_pipeline.py  — five pipeline stages")
print("  openshift/training/          — PyTorchJob manifests")
print()
print("Done. See 03_two_node_training_demo.py for distributed topology.")
