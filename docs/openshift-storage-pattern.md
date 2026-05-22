# OpenShift Storage Pattern

Canonical reference for how this project manages data on OpenShift AI.

> This document is the single source of truth for storage decisions.
> `docs/training-guide.md` and `docs/deployment.md` link here rather than
> repeating these rules.

---

## The four rules

### 1. Argo CD manages desired state — not ephemeral jobs

Argo CD syncs:
- Namespace, RBAC, ServiceAccounts (`openshift/gitops/`)
- Sealed Secrets (`openshift/gitops/secrets/`)
- Workbench Notebook CR (`openshift/gitops/workbench/`)
- DSPA pipeline server (`openshift/gitops/pipeline/`)

Argo CD does **not** sync training jobs. PyTorchJob manifests live in
`openshift/training/` and are applied manually:

```bash
oc apply -f openshift/training/pytorchjob-pragma-s.yaml -n pragma-encoder
```

Rationale: a training job is a one-shot execution, not desired state. Syncing
it via Argo CD would re-trigger training on every reconciliation loop.

### 2. S3-compatible object storage is the source of truth

All durable artifacts — training datasets, tokeniser vocabularies, model
checkpoints, and final outputs — live in S3.

Nothing that needs to survive a pod restart may be stored only on a local
volume. If it is not in S3, it does not exist.

| Artifact | S3 path (relative to bucket root) |
|---|---|
| TabFormer CSV | `pragma-encoder/data/tabformer/card_transaction.v1.csv` |
| Fitted tokeniser vocab | `pragma-encoder/data/tabformer/vocab.pkl` |
| PRAGMA-S checkpoints | `pragma-encoder/checkpoints/pragma-s/checkpoint_epoch<NNNN>.pt` |
| PRAGMA-M checkpoints | `pragma-encoder/checkpoints/pragma-m/checkpoint_epoch<NNNN>.pt` |
| PRAGMA-S outputs | `pragma-encoder/outputs/pragma-s/` |
| PRAGMA-M outputs | `pragma-encoder/outputs/pragma-m/` |

### 3. emptyDir is runtime scratch only

`emptyDir` volumes are ephemeral local execution storage. They:
- Are created when the pod starts and destroyed when it exits
- Cannot be used as canonical storage for any artifact
- Are the correct choice for scratch space, code clones, and data staging caches

Both training manifests use two `emptyDir` volumes:

| Volume name | Mount | Purpose |
|---|---|---|
| `workspace` | `/workspace` | Code clone + staged data; populated by init container |
| `dshm` | `/dev/shm` | Expanded shared memory for PyTorch DataLoader workers |

These are intentionally ephemeral. On pod restart (OOM, preemption, node
failure), the init container re-downloads code from GitHub and data from S3
before training starts again.

### 4. RWO PVCs must not be used as canonical training dataset storage

ReadWriteOnce PVCs are bound to a single node. Using one as the training data
store creates a hard scheduling constraint: the pod must land on the same node
as the PVC. This prevents multi-node jobs and conflicts with GPU-topology-aware
scheduling.

This project explicitly removed all dataset and output PVCs from the training
manifests. The prior design had both a `workbench-data` PVC (code) and a
`model-output` PVC (artifacts). Both were replaced by `emptyDir` + S3.

---

## Data flow

```
Before the job:
  Local workstation
    upload_training_data.py ──────────────────────► S3
                                                    pragma-encoder/data/tabformer/

During the job (each pod, each restart):
  Init container
    GitHub ──git clone──────────────────────────► /workspace/repo/   (emptyDir)
    S3 ─────download_file───────────────────────► /workspace/data/   (emptyDir)

  Main container (reads /workspace populated by init)
    Training ...
    rank 0 ──upload checkpoint after each epoch──► S3
                                                    pragma-encoder/checkpoints/
    rank 0 ──upload all outputs after training──► S3
                                                    pragma-encoder/outputs/

After the job:
  S3 ──────────────────────────────────────────► downstream consumers
```

---

## Credentials

All S3 access uses credentials supplied by the OpenShift AI Connection. In this
deployment those credentials are stored in the `pragma-workbench-env` Secret (a
test fixture/default managed as a Sealed Secret in
`openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml`).

Required keys (native OpenShift AI S3 Connection schema):

| Key | Description | Required |
|---|---|---|
| `AWS_S3_BUCKET` | S3 bucket name | yes |
| `AWS_S3_ENDPOINT` | S3 endpoint URL (e.g. `https://s3.us-west-2.amazonaws.com`) | yes |
| `AWS_ACCESS_KEY_ID` | S3 access key ID | yes |
| `AWS_SECRET_ACCESS_KEY` | S3 secret access key | yes |
| `AWS_DEFAULT_REGION` | S3 region (e.g. `us-west-2`) | no |

Source of truth: `oc get cm s3 -n redhat-ods-applications -o yaml`.
These are the env var names the RHOAI dashboard injects into pods when a Connection is attached.

> **TD-010 Resolved (2026-05-22):** The SealedSecret was re-sealed with native `AWS_*` key names.
> The `MODEL_REGISTRY_*` fallback code has been removed from the codebase.
> See `docs/tech-debt.md §TD-010`.

The secret is referenced via `envFrom.secretRef` in both the init container
and main container of every training manifest. Credentials are never
hardcoded in manifests or scripts.

To create or rotate the secret:
```bash
# Edit openshift/secrets/workbench-secret.yaml (NOT committed)
# Seal it:
kubeseal --scope namespace-wide \
  --namespace pragma-encoder \
  --format yaml \
  < openshift/secrets/workbench-secret.yaml \
  > openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml
```

---

## Failure semantics

S3 download or upload failure must exit non-zero so the pod does not proceed
with missing or partial data.

Every init container and upload script uses `set -euo pipefail`. The boto3
calls (`download_file`, `upload_file`) raise exceptions on failure, which
propagate as non-zero exit codes under `python3 -c`. The KFTO launcher then
marks the pod as failed and triggers a restart (per `restartPolicy: OnFailure`).

The `scripts/upload_training_data.py` script also validates local file
existence before attempting S3 uploads, and exits with code 1 on error.

---

## S3 path conventions

All paths follow this structure:

```
<bucket-root>/
  pragma-encoder/
    data/
      <dataset-name>/        ← input datasets (immutable once uploaded)
        *.csv
        vocab.pkl
    checkpoints/
      <variant>/             ← mid-training checkpoints (overwritten each run)
        checkpoint_epoch<NNNN>.pt
    outputs/
      <variant>/             ← post-training artifacts (overwritten each run)
        *
```

Rules:
- `data/` is **write-once**: upload before training, never overwrite during training.
- `checkpoints/` is **overwritten per run**: each new training run replaces previous checkpoints.
- `outputs/` is **overwritten per run**: final artifacts from the most recent completed run.

If you need to preserve outputs across runs, copy them to a versioned prefix
(e.g., `pragma-encoder/outputs/pragma-s/runs/2026-05-18/`) before re-running.

---

## Future-facing: new dataset convention

When adding a second dataset (e.g., synthetic transactions, a different
financial domain):

1. **Upload to a new prefix** — do not overwrite the TabFormer data:
   ```
   pragma-encoder/data/<new-dataset-name>/
   ```

2. **Add a new vocab** — the tokeniser must be fitted separately for each dataset:
   ```
   pragma-encoder/data/<new-dataset-name>/vocab.pkl
   ```

3. **Do not modify the existing PyTorchJob manifests** — copy and specialise:
   ```
   openshift/training/pytorchjob-pragma-s-<dataset-name>.yaml
   ```
   Update only the `S3_STAGE_SCRIPT` file list and the `--csv-path`/`--vocab-path`
   training script arguments. The storage pattern (emptyDir, secretRef, set -euo
   pipefail) must be preserved unchanged.

4. **Update `upload_training_data.py`** or write a new upload script for the
   new dataset. The existing script is TabFormer-specific (`_S3_PREFIX` constant).

5. **Do not break TabFormer training** — the existing manifests and S3 paths
   are the live training path and must not be modified as part of adding a new dataset.

---

## Manifest review checklist

Before submitting a new or modified PyTorchJob manifest, verify:

- [ ] No `persistentVolumeClaim` referencing a dataset or output PVC
- [ ] `emptyDir` used for all local scratch volumes
- [ ] `emptyDir` volumes annotated with `# ephemeral` comments or equivalent
- [ ] S3 credentials via `envFrom.secretRef: pragma-workbench-env` only — no hardcoded keys
- [ ] Init container command uses `set -euo pipefail`
- [ ] S3 download script exits non-zero on failure (boto3 raises; `python3 -c` propagates)
- [ ] Post-training upload script exits non-zero on failure
- [ ] `restartPolicy: OnFailure` set on all replica specs
- [ ] Training command includes `--resume` to pick up S3 checkpoint on restart
