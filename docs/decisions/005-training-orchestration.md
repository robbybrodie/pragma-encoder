# Decision 005: Training Orchestration

Status: Accepted
Date: 2026-05-17 (supersedes embedded Decision 005 in old ADR 001)
Paper reference: Section 2.4 (Training Infrastructure)
Supersedes: embedded "Decision 005: NeMo for Training Orchestration" in the original ADR 001 file

---

## Context

The PRAGMA paper (§2.4) describes training infrastructure at Revolut scale:
LMDB user index, Parquet event shards, dynamic batching, varlen FlashAttention,
and 16–32× H100 distributed runs.

For this open-source implementation, a decision was needed on:
1. Which training loop framework to use
2. How to handle distributed training
3. How to deploy to an OpenShift AI cluster
4. How to orchestrate multi-stage pipeline workflows
5. Where to store datasets, checkpoints, and artifacts

An earlier draft of this codebase listed "NeMo AutoModel" as the training
orchestration choice. **NeMo AutoModel is not currently used.** The codebase
uses a custom PyTorch training loop. This ADR corrects the record.

---

## Decision

### 1. Custom PyTorch training loop (implemented)

`scripts/train_pragma.py` implements the pre-training loop directly in PyTorch.
No external training framework (NeMo, PyTorch Lightning, Hugging Face Trainer)
is used for pre-training.

The script handles:
- Dataset loading via `src/data/pragma_dataset.py`
- Sequence packing (`src/training/packing.py`)
- MLM masking (`src/masking/strategy.py`)
- Forward pass through `PRAGMA.forward()`
- MLM loss computation
- Gradient updates (AdamW)
- Checkpoint save/load
- DDP-aware logging (rank 0 only)

### 2. torchrun / DistributedDataParallel (implemented)

`scripts/train_pragma.py` supports both single-process and multi-process
distributed training via DDP.

When launched via `torchrun` (as KFTO does), the script detects
`RANK` / `LOCAL_RANK` / `WORLD_SIZE` environment variables and initialises
`torch.distributed` with the `nccl` backend.

`DistributedSampler` ensures each rank receives a disjoint shard of the
dataset per epoch.

### 3. KFTO PyTorchJob for OpenShift distributed execution (implemented)

Distributed training on the OpenShift AI cluster uses the Kubeflow Training
Operator (KFTO) `PyTorchJob` resource (`kubeflow.org/v1`).

GA API only. Do not use TrainJob (Tech Preview as of RHOAI 2.x).

Manifests:
- `openshift/training/pytorchjob-pragma-s.yaml` — single-node (1 GPU)
- `openshift/training/pytorchjob-pragma-m.yaml` — PRAGMA-M
- `openshift/training/pytorchjob-pragma-s-2node.yaml` — two-node DDP demo

Each pod: init container clones the repo and downloads data from S3 into an
`emptyDir` scratch volume. Training writes checkpoints to the `emptyDir` and
uploads to S3. No PVC for canonical data (see ADR 003).

### 4. KFP SDK v2 / OpenShift Pipelines for workflow orchestration (implemented)

Multi-stage training workflows (prepare → upload → submit → train → export)
are expressed as KFP v2 pipeline components and compiled to pipeline YAML.

The five-stage pipeline is defined in `pipeline/pragma_pipeline.py` and the
workbench authoring surface in `tools/workbench/` (ADR 003, ADR 004).

KFP remains an optional dependency for local development and unit tests.

### 5. S3-compatible object storage (implemented)

All canonical data (prepared datasets, checkpoints, artifacts) is stored in
S3-compatible object storage. `boto3` is used for all S3 operations.

No PVC is used as canonical storage for datasets or model outputs.
Temporary scratch storage uses `emptyDir` inside PyTorchJob pods.

---

## NeMo AutoModel — not currently used

**NeMo AutoModel is not currently used in this codebase.**

The earlier embedded "Decision 005" recorded NeMo as the training orchestration
choice. This was a planning assumption that was not implemented. The actual
training loop in `scripts/train_pragma.py` uses pure PyTorch.

NeMo remains a viable future option for scaling to larger GPU counts or for
adopting NeMo's optimised kernels (varlen FlashAttention, sequence parallelism).
Any adoption of NeMo AutoModel requires:
1. A new ADR superseding this one
2. A prototype demonstrating NeMo compatibility with the PRAGMA model architecture
3. Validation that the MLM training objective is correctly expressed in NeMo

Do not introduce NeMo into `scripts/train_pragma.py` or `src/` without a new ADR.

---

## Consequences

Enables:
- Training loop is fully transparent and modifiable
- No framework compatibility constraints on model architecture
- Local training without any cluster or framework installation (`python scripts/train_pragma.py`)
- Distributed training via standard `torchrun` (KFTO injects `MASTER_ADDR`, `MASTER_PORT`)
- Checkpoint resume from S3

Constrains:
- Optimiser, precision, and batching are managed manually (no framework helpers)
- Checkpoint format is PyTorch native (`.pt` files), not NeMo format
- Varlen FlashAttention kernel is not currently implemented (standard padded attention used)
  See §2.4 for the throughput benefit; implement as a later optimisation

Known limitations:
- Multi-node checkpoint resume assumes shared filesystem (TD-006 in tech-debt.md)
  Preferred fix: all ranks download checkpoint from S3 independently

---

## What future sessions must not contradict

- Do not introduce NeMo, PyTorch Lightning, or Hugging Face Trainer for pre-training
  without a new ADR that supersedes this one.
- Do not use PyTorchJob TrainJob API (Tech Preview); use `kubeflow.org/v1` PyTorchJob.
- Do not use PVCs as canonical data storage (S3 only).
- Use KFP SDK v2 for pipeline components (not v1).

---

## References

- Paper: §2.4 (Training Infrastructure)
- `scripts/train_pragma.py`
- `openshift/training/pytorchjob-pragma-s.yaml`
- `openshift/training/pytorchjob-pragma-s-2node.yaml`
- `pipeline/pragma_pipeline.py`, `pipeline/components_pragma.py`
- ADR 003: `docs/decisions/003-workbench-training-api.md`
- TD-006: `docs/tech-debt.md` (multi-node checkpoint resume)
