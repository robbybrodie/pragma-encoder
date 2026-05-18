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

The training script (`scripts/train_pragma.py`) handles both modes automatically.
No code change is needed between local and cluster runs.

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
python src/data/fit_tokenizer.py \
    --csv-path data/tabformer/card_transaction.v1.csv \
    --output   data/tabformer/vocab.pkl
```

This produces `vocab.pkl`, which the training script loads at startup.

### 2. Upload training data to S3

For any cluster job (both PRAGMA-S and PRAGMA-M), upload the data to S3 first.
This only needs to be done once — the operation is idempotent.

```bash
# Export credentials (same keys as pragma-workbench-env secret):
export MODEL_REGISTRY_BUCKET=<bucket>
export MODEL_REGISTRY_ENDPOINT=<host-without-scheme>
export MODEL_REGISTRY_ACCESS_KEY=<access-key>
export MODEL_REGISTRY_SECRET_KEY=<secret-key>

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

## Local training (single process)

```bash
python scripts/train_pragma.py \
    --model-variant pragma-s \
    --csv-path  data/tabformer/card_transaction.v1.csv \
    --vocab-path data/tabformer/vocab.pkl \
    --output-dir outputs/pragma-s \
    --epochs 10 \
    --batch-size 32
```

To resume from a previous run:
```bash
python scripts/train_pragma.py \
    --model-variant pragma-s \
    --csv-path  data/tabformer/card_transaction.v1.csv \
    --vocab-path data/tabformer/vocab.pkl \
    --output-dir outputs/pragma-s \
    --resume \
    --epochs 10 \
    --batch-size 32
```

`--resume` checks `--output-dir` for the latest `checkpoint_epoch*.pt` and
loads it automatically. No epoch argument changes are needed.

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

| Event | What happens |
|-------|-------------|
| End of each epoch (rank 0) | Checkpoint saved locally to `--output-dir` |
| After local save (if `--s3-checkpoint-prefix` set) | Checkpoint uploaded to S3 |
| Pod restart | Init container runs fresh; `--resume` downloads latest S3 checkpoint |
| Training complete (rank 0 / master pod) | All outputs uploaded to S3 |

Checkpoint files follow the naming convention: `checkpoint_epoch<NNNN>.pt`

Resume logic:
1. Check `--output-dir` for the latest local `checkpoint_epoch*.pt`
2. If not found and `--s3-checkpoint-prefix` is set, download latest from S3
3. If found (local or S3), load weights, optimizer state, and resume from that epoch

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
