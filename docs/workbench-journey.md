# The PRAGMA Workbench Journey

A data-scientist-friendly guide to training, inspecting, and understanding the
PRAGMA foundation model pipeline — from your first workbench call to a
distributed two-node training run.

> Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4

---

## Where does the journey start?

You start inside an **OpenShift AI workbench** — a JupyterLab environment
running on the cluster with GPU access, pre-installed dependencies, and S3
credentials already injected into the pod. You do not need to configure
buckets, endpoints, or API keys manually. They are provided by the cluster
operator via a Kubernetes Secret (`pragma-workbench-env`) that the workbench
reads automatically at startup.

If you are running locally (laptop, CI), the same examples work. The only
difference is that S3 credentials and GPU access are your responsibility to
configure. In `dry_run` mode (the default for all examples) no credentials
are required at all.

---

## Step 1 — Run the beginner example

Open a terminal in your workbench and run:

```bash
python examples/workbench/01_train_ibm_tabformer.py
```

Or paste the key lines into a notebook cell:

```python
from src.workbench import train_pragma

run = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",
    epochs=10,
    prepare_if_missing=True,
    mode="dry_run",
)
run.show_pipeline()
```

You will see output like this:

```
PRAGMA Training Pipeline
==================================================
  [DRY RUN] Pipeline preview — no training job has been submitted.

  [pending  ] prepare  — Validate and prepare dataset (fit tokeniser)
  [pending  ] upload   — Upload prepared artifacts to S3 (idempotent)
  [pending  ] submit   — Submit PyTorchJob to cluster or configure local run
  [pending  ] train    — Pretraining — masked event modelling (§2.3.5)
  [pending  ] export   — Export model checkpoints and outputs to S3
```

The `[DRY RUN]` banner is intentional. Nothing has been submitted or trained.
This is a preview of what the pipeline would do.

---

## Step 2 — The five visible stages

Every PRAGMA training run passes through five stages, always in this order:

| Stage | What it does |
|-------|-------------|
| **prepare** | Fit the tokeniser on the raw transaction data. Build a `DatasetManifest` describing the prepared shards. |
| **upload** | Copy prepared data and vocabulary to S3. This is the only stage that writes to durable storage. Idempotent — safe to re-run. |
| **submit** | Create a training job on the cluster (or configure a local run). For two-node training this creates two pods. |
| **train** | Run masked event modelling pretraining. All five masking strategies from §2.3.5 are active. Checkpoints are uploaded to S3 after each epoch. |
| **export** | Write the final checkpoint and vocabulary to the canonical S3 export path for downstream use (probes, LoRA fine-tuning, embedding extraction). |

These five names are used consistently everywhere: in `PragmaRun.show_pipeline()`,
in the KFP pipeline components (`PIPELINE_STAGE_NAMES`), and in the PyTorchJob
manifests. If you see a stage name in one place you can find it in all the others.

To read a detailed explanation of each stage, run:

```bash
python examples/workbench/02_understand_pipeline.py
```

---

## Step 3 — Understanding `dry_run`

`mode="dry_run"` is the safe default for all examples. It means:

- **No training is submitted.** No job runs on the cluster or locally.
- **No S3 access.** No data is read, written, or uploaded.
- **No credentials are required.** You do not need a bucket name, endpoint, or API key.
- **No DatasetAdapter is called.** The tokeniser is not fitted. No files are created.

What `dry_run` *does* return is a `PragmaRun` object — a pipeline preview. The
`PragmaRun` carries a placeholder `DatasetManifest` (with `dataset_version="dry-run"`)
and all five stages in `"pending"` state. You can inspect it, call
`show_pipeline()`, and see the configuration that *would* be used — but
nothing has happened yet.

```python
# Confirming dry_run produced no real data:
print(run.manifest.dataset_version)   # "dry-run"
print(run.metrics())                  # {} — no training
print(run.artifacts())                # {} — no outputs
```

The `[DRY RUN]` banner in `show_pipeline()` output is there so the preview
cannot be mistaken for a submitted run. A real submitted run would show
`[running]` or `[completed]` against individual stages as training progresses.

**To submit real training**, change `mode="dry_run"` to `mode="cluster"`.
That mode is not yet implemented — it will raise `NotImplementedError` until
the cluster dispatch component is added. This is by design: `dry_run` is the
safe default so examples are always safe to run.

---

## Step 4 — Prove the model can learn: local learning validation

Before submitting anything to a cluster, it is worth confirming that the full
PRAGMA training pipeline — tokeniser → masking → assembler → model → loss —
actually works on your machine and that the model learns (loss decreases).

The local learning validation script does exactly this on synthetic data:

```bash
PYTHONPATH=. python examples/workbench/06_local_learning_validation.py
```

Or with more steps and explicit model size:

```bash
PYTHONPATH=. python examples/workbench/06_local_learning_validation.py \
    --model-size S --max-steps 50
```

**What it does:**

1. Builds a `PRAGMA-S` model and an `EmbeddingAssembler` from the paper-spec config.
2. Generates a synthetic fixed batch (IBM TabFormer-style geometry — no real data, no S3).
3. Runs `--max-steps` Adam optimisation steps, printing loss at 10% intervals.
4. Checks that loss decreased (gradient flow confirmed).
5. Saves a checkpoint to a temporary directory and reloads it.
6. Verifies the reloaded model produces bit-identical `zh` output.
7. Runs a short resume loop from the reloaded checkpoint.
8. Prints a final PASS/FAIL verdict.

Expected output (truncated):

```
============================================================
  PRAGMA LOCAL LEARNING VALIDATION — Readiness Report
============================================================
  Dataset
    name            : ibm-tabformer-synthetic
    path            : (not yet prepared — using synthetic data)
    sequences       : 200
    events (total)  : 2,000
  ...
  Model
    variant         : pragma-s
    trainable params: 10,xxx,xxx
============================================================

  Running 50 steps with Adam lr=1e-4 ...
  Batch: 4 sequences × 10 events × 8 tokens/event

  step    1/50  loss=9.9312  (0.3s)
  step    5/50  loss=9.3141  (1.4s)
  ...
  step   50/50  loss=4.2017  (14.2s)

============================================================
  Training summary
============================================================
  Initial loss (step 1)  : 9.9312
  Final loss   (step  50): 4.2017
  Loss decreased         : YES ✓

  Checkpoint saved       : pragma-s-step-0050.pt  (41,xxx KB)
  Checkpoint reload      : OK ✓

  Resuming for 5 more step(s) from reloaded checkpoint ...
    resume step 1: loss=4.1843
    resume step 5: loss=4.1021

============================================================
  RESULT
============================================================
  [PASS]  Loss is finite
  [PASS]  Loss decreased
  [PASS]  Checkpoint reload OK

  All checks passed. PRAGMA-S is learning correctly.

  Next step: run on real IBM TabFormer data.
    See docs/training-guide.md → Local training — PRAGMA-S

  This was LOCAL LEARNING VALIDATION — no cluster resources were used.
============================================================
```

**This is NOT production training.** It uses synthetic data and a fixed batch
(pure memorisation) to confirm architectural correctness. A real training run
requires real IBM TabFormer data and is documented in `docs/training-guide.md`.

**The four-stage progression:**

| Stage | Mode | Proves | Resources needed |
|-------|------|--------|-----------------|
| `mode="dry_run"` | Preview | Pipeline structure is correct | None |
| `06_local_learning_validation.py` | Local | Model can learn (gradients flow) | CPU only |
| Local training on real data | Local | Tokeniser + real data work | CSV file |
| Cluster training | OpenShift AI | Distributed scaling works | GPU node + S3 + cluster |

Running the local learning validation before cluster training catches
architectural bugs (frozen parameters, detached tensors, broken loss paths)
on a laptop in under a minute — rather than discovering them after hours of
wasted GPU compute.

---

## Step 5 — The DatasetManifest

The `DatasetManifest` is a small metadata object that sits between the prepare
stage and everything downstream. It records:

| Field | Meaning |
|-------|---------|
| `dataset_name` | Which dataset this is (`"ibm-tabformer"`, or a future bank dataset) |
| `dataset_version` | A content hash or label set by the adapter. `"dry-run"` in preview mode. |
| `prepared_prefix_uri` | The S3 path prefix where prepared shards live |
| `shards` | A tuple of `DatasetShard` objects, one per prepared file |
| `vocab_uri` | S3 path to `vocab.pkl` (the fitted tokeniser vocabulary) |
| `config` | The `PRAGMAConfig` used during preparation (fixes `max_event_tokens`, `max_events`, `d_model`) |

The manifest is the *only* input the train stage needs. If you have already
prepared and uploaded a dataset to S3, you can skip the prepare and upload
stages entirely and pass `manifest_uri=` directly to the pipeline or to the
PyTorchJob. Expensive tokeniser fitting does not need to repeat.

Today the only registered adapter is IBM TabFormer (`"ibm-tabformer"` — credit
card transactions from the IBM public research dataset). The adapter pattern is
designed so that future bank transaction datasets slot in as new adapters
without changing the pipeline, the manifest format, or the training script.
New datasets register one class in `src/data/adapters/__init__.py`.

---

## Step 6 — How storage works

This is the most common source of confusion, so it is explained explicitly.

### S3 — durable source of truth

S3-compatible object storage (the same bucket your workbench pod uses) is where
all durable artefacts live:

```
s3://<bucket>/pragma-encoder/data/<dataset>/      ← prepared training data
s3://<bucket>/pragma-encoder/checkpoints/<model>/ ← training checkpoints
s3://<bucket>/pragma-encoder/outputs/<model>/     ← final model outputs
```

Every piece of data that matters survives in S3. A training pod that crashes
and restarts will re-download from S3 and continue from the latest checkpoint.
No data is permanently lost when a pod exits.

### emptyDir — ephemeral runtime scratch

Each training pod has a local `/workspace` directory backed by `emptyDir` — a
temporary disk allocation that is created when the pod starts and destroyed when
the pod exits. It is never shared between pods.

```
/workspace/repo/   ← git clone of this repository (fetched at startup)
/workspace/data/   ← training data staged from S3 (fetched at startup)
/workspace/outputs/← checkpoints written during training (uploaded to S3)
```

The init container populates `/workspace` before training begins. It clones the
repository from GitHub and downloads the training data from S3. The main
training container then runs entirely from `/workspace` — it never reads from
S3 during training, only writes checkpoints back to it.

When the pod exits, `/workspace` disappears. This is fine because S3 is the
durable store. The local workspace is runtime scratch only.

### Why no PVCs?

A Kubernetes PersistentVolumeClaim (PVC) with ReadWriteOnce (RWO) access mode
can only be attached to one node at a time. If training data lived on a RWO PVC,
the training job would have to run on whichever node the PVC happened to be
attached to — even if that node had no GPU available. This is called a
scheduling constraint.

By putting data in S3 (which any node can read) and using `emptyDir` for the
local workspace, the scheduler is free to place training pods on any available
GPU node. For two-node training this is essential — the two pods may end up on
completely different physical nodes.

PVCs may appear elsewhere in the platform (for the data science pipeline server,
for example), but they are not part of the PRAGMA training data path.

---

## Step 7 — Two-node distributed training

To see the two-node topology in action, run:

```bash
python examples/workbench/03_two_node_training_demo.py
```

Setting `nodes=2` in `train_pragma()` selects the two-node PyTorchJob manifest
(`openshift/training/pytorchjob-pragma-s-2node.yaml`). This creates two pods:

- **Master** (rank 0) — coordinates rendezvous, handles logging and checkpointing
- **Worker** (rank 1) — participates in gradient averaging, skips checkpointing

Both pods run `torchrun --nnodes=2 --nproc_per_node=1`, giving a `WORLD_SIZE`
of 2. The training script detects this automatically and:

- Uses `DistributedSampler` to split the training data between pods
- Wraps the model in `DistributedDataParallel` (DDP) so gradients are averaged
  across both pods after each backward pass
- Synchronises all pods at the end of each epoch with a barrier
- Gates all checkpointing and S3 uploads to rank 0 (the Master pod) only

**What is fully supported on a fresh two-node run:**

| Capability | Status |
|-----------|--------|
| NCCL gradient averaging across pods | Supported |
| Disjoint data sharding via DistributedSampler | Supported |
| Rank-0-only checkpointing (no S3 write races) | Supported |
| Epoch barrier synchronisation | Supported |

**What is not yet supported — TD-006:**

Checkpoint resume across pods is not correctly implemented for the per-pod
`emptyDir` storage pattern. If a two-node job is interrupted *after* a
checkpoint has been saved and then restarted, rank 0 downloads the checkpoint
from S3 but the Worker pod looks for it in its own (empty) local workspace and
finds nothing. Rank 0 resumes; rank 1 starts from scratch. Model states diverge.

For the demo manifest (`--max-steps 20`, first run), this does not matter: no
checkpoint is written before the job exits, so `--resume` is always a no-op.
The limitation only affects genuine restarts of long-running two-node jobs.

The resolution (all ranks download the checkpoint from S3 independently) is
described in `docs/tech-debt.md` under **TD-006**. It is not yet implemented.

---

## Step 8 — What Argo CD does (and does not do)

**Argo CD** is a GitOps tool that the platform team uses to keep the cluster
configuration in sync with this repository. When someone merges a change to a
manifest in `openshift/gitops/`, Argo CD detects the difference and applies the
updated Kubernetes resources automatically.

Argo CD deploys:
- The workbench pod (JupyterLab environment)
- The data science pipeline server (KFP)
- Namespace configuration, RBAC, and secrets scaffolding
- The network and storage infrastructure for the pipeline

**Argo CD does not:**
- Store training data or model artefacts (S3 does)
- Submit training jobs (you or KFP do)
- Run the tokeniser or prepare datasets
- Track model checkpoints or metrics

When you call `train_pragma()` from a workbench cell, Argo CD is not involved.
You are talking directly to the cluster API (for `mode="cluster"`) or to your
local Python process (for `mode="local"` or `mode="dry_run"`). Argo CD is
background infrastructure that ensures the workbench you are sitting in exists
and is correctly configured.

Think of Argo CD as the facilities team that built and maintains the laboratory.
You are the scientist who runs experiments inside it.

---

## Step 9 — What the workbench API does

The workbench API (`src/workbench`) is the thin layer that data scientists
interact with. Its public surface is small:

```python
from src.workbench import train_pragma, PragmaRun, PIPELINE_STEP_NAMES
```

`train_pragma()` takes four key arguments a data scientist cares about:

| Argument | Meaning |
|----------|---------|
| `dataset` | Which dataset to use (`"ibm-tabformer"` today) |
| `model_size` | `"S"` (~10M params), `"M"` (~100M), or `"L"` (~1B) |
| `epochs` | How many training epochs to run |
| `nodes` | `1` = single GPU, `2` = two-GPU distributed training |

And one argument that controls safety:

| Argument | Meaning |
|----------|---------|
| `mode` | `"dry_run"` = preview only (default in examples); `"cluster"` = submit to OpenShift AI; `"auto"` = detect environment |

It returns a `PragmaRun` — a result object you can inspect:

```python
run.show_pipeline()   # print all five stages with their current status
run.metrics()         # training loss, epoch, step (empty until training starts)
run.artifacts()       # S3 URIs for checkpoint, model, vocab (empty until exported)
run.manifest          # DatasetManifest describing the prepared dataset
```

The API deliberately keeps cluster details out of the data scientist's path.
You do not write `oc apply` commands or edit YAML. The underlying PyTorchJob
manifests exist in `openshift/training/` for operators and engineers who need
to inspect or modify the cluster configuration, but data scientists interact
through `train_pragma()` and inspect results through `PragmaRun`.

---

## Quick reference — examples

| Example | What it covers | How to run |
|---------|---------------|------------|
| `examples/workbench/01_train_ibm_tabformer.py` | Beginner path: `train_pragma` + `show_pipeline` | `python examples/workbench/01_train_ibm_tabformer.py` |
| `examples/workbench/02_understand_pipeline.py` | Five-stage walkthrough: what each stage does | `python examples/workbench/02_understand_pipeline.py` |
| `examples/workbench/03_two_node_training_demo.py` | Two-node topology, DDP, and TD-006 honest callout | `python examples/workbench/03_two_node_training_demo.py` |
| `examples/workbench/06_local_learning_validation.py` | Proves PRAGMA can learn: loss decreases, gradients flow, checkpoint save/reload | `PYTHONPATH=. python examples/workbench/06_local_learning_validation.py` |

Examples 01–03 use `mode="dry_run"` by default. They require no credentials
and submit no jobs. They are safe to run anywhere.

Example 06 runs a short local training loop on synthetic data (CPU only, no
credentials, no cluster). It is safe to run anywhere and completes in under a
minute on a laptop.

---

## What to do next

| Goal | Action |
|------|--------|
| Preview the pipeline for any model size | Change `model_size="S"` to `"M"` or `"L"` in any example |
| Prove the model can learn (before real training) | `PYTHONPATH=. python examples/workbench/06_local_learning_validation.py` |
| Prepare data locally (no cluster) | Run `src/data/fit_tokenizer.py` + `scripts/upload_training_data.py` |
| Submit a real single-node training job | Follow `docs/training-guide.md` → Cluster training — PRAGMA-S |
| Submit a real two-node training job | Apply `openshift/training/pytorchjob-pragma-s-2node.yaml` (see TD-006 note above) |
| Use PRAGMA embeddings for downstream tasks | See `src/adaptation/probe.py` (linear probe) and `src/adaptation/lora.py` (LoRA) |
| Understand the paper-to-code mapping | Read `docs/paper-to-code.md` |

---

## Honest status summary

| Capability | Status |
|-----------|--------|
| `mode="dry_run"` — pipeline preview | **Complete** |
| Local learning validation (synthetic data, CPU) | **Complete** — `examples/workbench/06_local_learning_validation.py` |
| IBM TabFormer dataset adapter | **Complete** |
| DatasetManifest and DatasetShard | **Complete** |
| PragmaRun inspection API | **Complete** |
| KFP pipeline components (five stages) | **Complete** (stubs — `upload`, `submit`, `run_pretraining`, `export` raise `NotImplementedError`) |
| `mode="cluster"` — cluster dispatch | **Not yet implemented** |
| `mode="local"` — local subprocess dispatch | **Not yet implemented** |
| Single-node training via PyTorchJob | **Complete** — `pytorchjob-pragma-s.yaml` |
| Two-node fresh training via PyTorchJob | **Complete** — `pytorchjob-pragma-s-2node.yaml` |
| Two-node checkpoint resume | **Not yet correct** (TD-006) |
