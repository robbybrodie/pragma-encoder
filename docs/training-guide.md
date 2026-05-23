# Training Guide

How to pretrain the PRAGMA encoder on transaction data.

> Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4

---

## Overview

PRAGMA pretraining uses a masked event modelling (MEM) objective — BERT-style,
not causal. Three masking strategies are applied per batch: token masking,
field masking, and event masking (Section 2.3.5).

Training can run:
- **Locally** — single process, any GPU/MPS/CPU
- **On OpenShift AI** — KFTO PyTorchJob, single-node (PRAGMA-S) or multi-node DDP (PRAGMA-M)

The canonical training entrypoint is `python -m pragma_encoder.training.train`
(equivalently `pragma-encoder-train`). The compatibility wrapper
`scripts/train_pragma.py` is also available in the training image.
No code change is needed between local and cluster runs.

---

## Storage model

For cluster training, all storage decisions — S3 vs emptyDir vs PVC — are
governed by a single rule set. Before modifying any training manifest, read:

**[docs/openshift-storage-pattern.md](openshift-storage-pattern.md)**

Short version: S3 is the source of truth. emptyDir is runtime scratch only.
No dataset PVCs. Credentials come from the OpenShift AI Connection (represented
in this deployment by the `pragma-workbench-env` Secret).

---

## Model sizes

| Variant | Parameters | Nodes × GPUs | Use case |
|---------|-----------|--------------|----------|
| PRAGMA-S | ~10M | 1 × 1 | Research, development, smoke tests |
| PRAGMA-M | ~100M | 4 × 4 = 16 | Larger datasets, production pretraining |
| PRAGMA-L | ~1B | — | Aspirational; not currently manifested |

Start with PRAGMA-S. It trains on a single GPU and validates the pipeline end-to-end.

---

## Prerequisites

### 1. Fit the tokeniser

The tokeniser pipeline must be fitted on the training data before training:

```bash
python -m pragma_encoder.data.fit_tokenizer \
    --csv-path data/tabformer/card_transaction.v1.csv \
    --output   data/tabformer/vocab.pkl
```

This produces `vocab.pkl`, which the training script loads at startup.

### 2. Upload training data to S3

For any cluster job (both PRAGMA-S and PRAGMA-M), upload the data to S3 first.
This only needs to be done once — the operation is idempotent.

```bash
# Export credentials (same keys as pragma-workbench-env secret — native RHOAI S3 Connection schema):
export AWS_S3_BUCKET=<bucket>
export AWS_S3_ENDPOINT=https://<host>   # full URL including scheme
export AWS_ACCESS_KEY_ID=<access-key>
export AWS_SECRET_ACCESS_KEY=<secret-key>

# Dry run first to verify paths:
python scripts/upload_training_data.py --dry-run

# Upload:
python scripts/upload_training_data.py \
    --csv-path   data/tabformer/card_transaction.v1.csv \
    --vocab-path data/tabformer/vocab.pkl
```

S3 destination paths (relative to bucket root):
```
pragma-encoder/data/tabformer/card_transaction.v1.csv
pragma-encoder/data/tabformer/vocab.pkl
```

---

## Checkpoint modes

`pragma_encoder` supports two checkpoint/resume modes. The mode is selected
automatically based on environment:

| Mode | When active | What it requires |
|---|---|---|
| **Local** (default) | No `AWS_S3_BUCKET`/`AWS_S3_ENDPOINT`, or no `--s3-checkpoint-prefix` | Only `--output-dir` |
| **S3** (platform/durable) | Both `AWS_*` env vars set and `--s3-checkpoint-prefix` non-empty | S3 credentials in env |

**Local mode is the default.** The wheel works fully without S3, OpenShift,
or any platform credentials. This is the correct mode for laptop development.

**S3 mode is for cluster/durable storage.** On OpenShift AI, an S3 Connection
injects `AWS_*` env vars into pods. Combined with `--s3-checkpoint-prefix`,
checkpoints are uploaded to S3 after each epoch and can be resumed across pod
restarts.

`build_checkpoint_store(output_dir, s3_prefix)` in
`pragma_encoder.training.checkpoints` selects the correct adapter:
- Returns `LocalCheckpointStore` when S3 config is absent or `s3_prefix` is empty.
- Returns `S3CheckpointStore` when both S3 env vars and a non-empty prefix are present.

---

## Local training (single process)

```bash
python -m pragma_encoder.training.train \
    --model-variant pragma-s \
    --csv-path  data/tabformer/card_transaction.v1.csv \
    --vocab-path data/tabformer/vocab.pkl \
    --output-dir outputs/pragma-s \
    --epochs 10 \
    --batch-size 32
```

To resume from a previous run:
```bash
python -m pragma_encoder.training.train \
    --model-variant pragma-s \
    --csv-path  data/tabformer/card_transaction.v1.csv \
    --vocab-path data/tabformer/vocab.pkl \
    --output-dir outputs/pragma-s \
    --resume \
    --epochs 10 \
    --batch-size 32
```

`--resume` scans `--output-dir` for `checkpoint_epoch*.pt` files and loads the
lexicographically latest one. If no checkpoint exists, training starts fresh.
No S3 credentials are required for local resume.

---

## Cluster training — PRAGMA-S (single node)

```bash
# Verify prerequisites:
oc get crd pytorchjobs.kubeflow.org
oc get secret pragma-workbench-env -n pragma-encoder

# Submit job:
oc apply -f openshift/training/pytorchjob-pragma-s.yaml -n pragma-encoder

# Monitor:
oc get pytorchjob pragma-s-pretrain -n pragma-encoder
oc logs -f -l job-name=pragma-s-pretrain,replica-type=master -n pragma-encoder

# Init container logs (data staging phase):
oc logs -c stage-data \
  $(oc get pod -n pragma-encoder -l job-name=pragma-s-pretrain -o name | head -1) \
  -n pragma-encoder
```

The job:
1. Init container clones the repo from GitHub and downloads data from S3
2. Training runs with `--checkpoint-every 1` (checkpoint each epoch)
3. Each checkpoint is uploaded to S3 immediately after saving
4. On completion, all outputs are uploaded to `pragma-encoder/outputs/pragma-s/`

If the pod is restarted (OOM, preemption, node failure), it re-runs the init
container (fresh data from S3) and resumes training from the latest S3 checkpoint.

---

## Cluster training — PRAGMA-M (4 nodes × 4 GPUs, DDP)

PRAGMA-M requires 4 GPU nodes. The job uses PyTorch DDP via `torchrun`.
All 4 pods run identical init containers (each node needs its own data copy).

```bash
# Verify prerequisites:
oc get crd pytorchjobs.kubeflow.org
oc get secret pragma-workbench-env -n pragma-encoder

# Submit job:
oc apply -f openshift/training/pytorchjob-pragma-m.yaml -n pragma-encoder

# Monitor all pods:
oc get pytorchjob pragma-m-pretrain -n pragma-encoder
oc logs -f -l job-name=pragma-m-pretrain,replica-type=master -n pragma-encoder
oc logs -f -l job-name=pragma-m-pretrain,replica-type=worker  -n pragma-encoder
```

**DDP topology:**
- 1 Master pod (KFTO RANK=0) + 3 Worker pods (KFTO RANK=1,2,3)
- Each pod runs `torchrun --nproc_per_node=4`
- Total: 16 GPU processes, WORLD_SIZE=16
- Per-process rank 0 (on master pod, local_rank 0) handles all checkpointing and S3 uploads

**Checkpoint resilience:**
- `restartPolicy: OnFailure` — any failed pod is restarted automatically
- Init container re-downloads code and data from S3 on restart
- Training starts with `--resume`: downloads latest checkpoint from
  `pragma-encoder/checkpoints/pragma-m/` in S3 and continues from that epoch
- Workers load the checkpoint too (broadcast via barrier in the script)

**S3 paths for PRAGMA-M:**
```
pragma-encoder/checkpoints/pragma-m/checkpoint_epoch<NNNN>.pt   ← mid-training
pragma-encoder/outputs/pragma-m/<all output files>              ← post-training
```

---

## Checkpoint behaviour

### Checkpoint naming

Files are named `checkpoint_epoch<NNNN>.pt` (zero-padded epoch number).
Lexicographic order equals epoch order, so the latest checkpoint is always
the last file alphabetically in `--output-dir`.

### Local mode (default — no S3)

| Event | What happens |
|-------|-------------|
| End of each epoch (rank 0) | `checkpoint_epoch<NNNN>.pt` written to `--output-dir` |
| `--resume` on next run | Latest `.pt` in `--output-dir` is loaded; training continues from that epoch |
| No checkpoint in `--output-dir` | Training starts from epoch 0 |

No S3 credentials required.  No `--s3-checkpoint-prefix` needed.

### S3 mode (platform/durable — OpenShift AI or explicit S3 config)

| Event | What happens |
|-------|-------------|
| End of each epoch (rank 0) | Checkpoint saved locally, then uploaded to S3 under `--s3-checkpoint-prefix` |
| Pod restart | Init container runs fresh; `--resume` selects latest S3 key and ALL ranks download independently |
| Training complete (rank 0 / master pod) | All outputs uploaded to S3 |

Requires `AWS_S3_BUCKET`, `AWS_S3_ENDPOINT`, and `--s3-checkpoint-prefix`.

### Resume logic

```
resolve_resume_checkpoint(output_dir, s3_prefix, rank, distributed, device)
  │
  ├─ S3 config present AND s3_prefix non-empty
  │    └─ Download latest S3 checkpoint (all ranks download independently — TD-006 fix)
  │
  └─ Otherwise (local mode)
       └─ Scan output_dir for checkpoint_epoch*.pt → return lexicographically latest
```

---

## Metadata and artifact traceability

`pragma-encoder-train` writes `metadata.json` to `--output-dir` on completion.

### metadata.json structure

```json
{
  "timestamp": "<ISO-8601 UTC>",
  "model_variant": "pragma-s",
  "epochs_completed": 10,
  "global_step": 12345,
  "dataset": {
    "dataset_name": "<original dataset reference — S3 URI or label>",
    "csv_staging_path": "<local path used for this run — ephemeral>",
    "limit_rows": 0
  },
  "output_dir": "<absolute path to output directory>",
  "final_checkpoint": "<path or S3 key of final checkpoint>",
  "args": { ... }
}
```

### dataset_name vs csv_staging_path

| Field | What it records | When to use it |
|---|---|---|
| `dataset.dataset_name` | Original dataset reference: S3 URI, dataset label, or csv_path for local runs | Trace back to the source of truth |
| `dataset.csv_staging_path` | Local path where the CSV was staged for this run (may be ephemeral `/tmp/`) | Debug staging failures; not the dataset source |

For local runs without `--dataset-name`, `dataset_name` falls back to the `--csv-path` value.

For pipeline runs, `run_pretraining` passes the original S3 URI via `--dataset-name`
so `dataset_name` always points to the original source — not the staging path in
the component pod's `/tmp/` scratch directory.

### Credential safety

The `args` section in `metadata.json` is filtered by `_METADATA_ALLOWED_ARG_KEYS`
(an explicit allowlist). AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
are env vars — they are never CLI args and therefore never appear in `metadata.json`.

Excluded from `args`: `csv_path`, `vocab_path`, `output_dir`, `device`, `resume`.
These are staging paths or runtime-specific values — not reproducibility-relevant
configuration.

### KFP component scratch / staging contract

In the `run_pretraining` KFP component:
- Input data (CSV, vocab) is downloaded to `scratch_dir` (default: `/tmp/pragma-run`).
- `scratch_dir` is a configurable parameter — tests redirect it to a tmpdir;
  future confidential runtimes can restrict it to attested memory regions.
- It is pod-local ephemeral storage, not a shared PVC.
- `metadata.json` is uploaded to S3 at `{output_prefix}/{variant}/metadata.json`
  alongside the checkpoint — making it durable, not ephemeral.

---

## Masking strategy

PRAGMA uses three masking strategies applied per batch (Section 2.3.5):

```python
from src.masking import MaskingStrategy

masker = MaskingStrategy(config)
masked_ids, target_ids, mask = masker.forward(token_ids, key_ids)
```

Mixture probabilities are controlled by `PRAGMAConfig`:
- `token_mask_prob = 0.15` — individual token masking
- `event_mask_prob = 0.10` — full event masking
- `key_mask_prob   = 0.10` — field masking

---

## Monitoring training loss

Training metrics to watch:
- **MLM loss** — should decrease from ~log(vocab_size) toward < 3.0
- **Per-epoch average loss** — logged by rank 0 at end of each epoch
- **Step loss** — logged every `--log-every` steps (default: 50)

In DDP mode, only rank 0 emits logs. Watch the master pod logs.
