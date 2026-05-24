# Workbench-to-Pipeline-to-Model-Publication — IBM TabFormer

How to author, compile, and run the PRAGMA pretraining pipeline for IBM
TabFormer card transaction data using an OpenShift AI Workbench.

> Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
> ADR: docs/decisions/003-workbench-training-api.md

---

## Overview

The PRAGMA pretraining pipeline follows the five §2.4 training infrastructure stages:

```
Stage 1 — prepare_dataset   : Fit tokeniser via IBMTabFormerAdapter
Stage 2 — upload_artifacts  : Upload prepared CSV + vocab to S3
Stage 3 — submit_pytorchjob : Configure KFTO PyTorchJob (stub — non-blocking)
Stage 4 — run_pretraining   : Run pragma_encoder.training.train (MEM objective)
Stage 5 — export_checkpoint : Publish checkpoint + metrics to S3 export prefix
```

The pipeline is authored from the Workbench Notebook, compiled to KFP v2
YAML, and submitted to OpenShift AI Data Science Pipelines (DSPA). All five
stages run inside component pods using the PRAGMA training image — not the
Workbench image. Data flows via S3 (OpenShift AI Connection), not PVCs.

---

## Prerequisites

### Platform substrate (ArgoCD-managed)

The following must be deployed in your namespace before running the pipeline:

| Resource | Kind | Purpose |
|----------|------|---------|
| `pragma-workbench-env` | `Secret` (sealed) | S3 credentials via AWS_* env vars |
| DSPA instance | `DataSciencePipelinesApplication` | KFP v2 API server |
| `pragma-encoder-workbench` | Workbench | Notebook environment |
| PRAGMA training image | `ImageStream` | Component pod base image |

S3 credentials (from OpenShift AI Connection) must include:
- `AWS_S3_BUCKET`
- `AWS_S3_ENDPOINT`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

### Data

IBM TabFormer card transaction CSV must be available in S3 at the path the
`IBMTabFormerAdapter` expects, or at a local path for `mode="local"` runs.

Expected file: `card_transaction.v1.csv`

---

## Workbench authoring

### 1. Import and preview (dry_run)

```python
from tools.workbench import train_pragma

# Preview pipeline shape and config — no S3, no cluster, no training
run = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",
    mode="dry_run",
)
run.show_pipeline()
```

Expected output:
```
PRAGMA Pretraining Pipeline — dry_run
  ✓ prepare   Prepare dataset via DatasetAdapter (dry_run)
  ● upload    Upload prepared artifacts to S3
  ● submit    Configure and submit KFTO PyTorchJob
  ● train     Execute pretraining (masked event modelling)
  ● export    Export model checkpoint to S3
```

### 2. Author a KFP pipeline

```python
from tools.workbench import train_pragma

# Returns a PragmaPipeline (no training, no S3, no cluster)
pipeline = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",
    epochs=10,
    mode="pipeline",
)

# Inspect pipeline structure
pipeline.show_pipeline()

# Compile to KFP v2 YAML
pipeline.compile("pipeline/pragma_tabformer_pipeline.yaml")
```

### 3. Submit to OpenShift AI (DSPA)

```python
from tools.workbench._submit import (
    get_dspa_endpoint,
    get_service_account_token,
    make_kfp_client,
    upload_pipeline,
    submit_pipeline_run,
    wait_for_run_terminal,
)

# Construct KFP client (SA token comes from workbench pod mount)
endpoint = get_dspa_endpoint()
token = get_service_account_token()
client = make_kfp_client(host=endpoint, token=token, verify_ssl=False)

# Upload compiled pipeline
pipeline_id = upload_pipeline(
    client=client,
    yaml_path="pipeline/pragma_tabformer_pipeline.yaml",
    pipeline_name="pragma-tabformer-s",
)

# Submit a Run with bounded smoke parameters
run_id = submit_pipeline_run(
    client=client,
    pipeline_id=pipeline_id,
    run_name="pragma-tabformer-smoke",
    params={
        "dataset_name": "ibm-tabformer",
        "model_size":   "S",
        "max_steps":    2,       # 2 gradient steps — fast smoke
        "limit_rows":   5,       # 5 customers — fast data prep
        "batch_size":   1,       # CPU-safe
        "device":       "cpu",
        "run_name":     "smoke",
    },
)
print(f"Run submitted: {run_id}")

# Wait for terminal state
final_state = wait_for_run_terminal(client=client, run_id=run_id, timeout_seconds=300)
print(f"Run state: {final_state}")  # → SUCCEEDED
```

---

## Pipeline parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | (required) | Adapter registry key. Use `"ibm-tabformer"`. |
| `model_size` | `"S"` | PRAGMA model variant: `"S"` (~10M), `"M"` (~100M), `"L"` (~1B). |
| `epochs` | `10` | Pretraining epochs. Use `1` for smoke runs. |
| `nodes` | `1` | Training nodes. `1` = single-node; `2` = 2-node DDP via PyTorchJob. |
| `max_steps` | `0` | Stop after N gradient steps (0 = run all epochs). Use `2` for smoke. |
| `limit_rows` | `0` | Cap dataset to N customers (0 = all). Use `5` for smoke. |
| `batch_size` | `32` | Training batch size. Use `1` for CPU/smoke runs. |
| `device` | `"auto"` | Device: `"auto"` (CUDA > MPS > CPU), `"cuda"`, `"cpu"`. |
| `output_prefix` | `"pragma-encoder/runs"` | S3 key prefix for all run artifacts. |
| `run_name` | `""` | Optional run label in S3 artifact paths and `metadata.json`. |

---

## What the pipeline produces

After a successful run, the following artifacts are available in S3:

```
pragma-encoder/runs/pragma-{s,m,l}/
    checkpoint_epoch1.pt     — model checkpoint (PyTorch state dict)
    metrics.jsonl            — per-step loss, LR, timestamps
    loss.png                 — loss curve (requires matplotlib in component image)
    vocab.pkl                — fitted tokenizer vocabulary
    metadata.json            — training configuration and artifact paths

pragma-encoder/runs/export/pragma-{s,m,l}/
    checkpoint_epoch1.pt     — checkpoint copy at canonical export prefix
    metrics.jsonl            — metrics copy
    loss.png                 — loss curve copy
    vocab.pkl                — vocab copy
    metadata.json            — metadata copy
    export_manifest.json     — JSON manifest of all artifact S3 URIs
```

`export_manifest.json` is the artifact publication record. It is NOT a
registration with the OpenShift AI Model Registry API (see TD-013 in
`docs/tech-debt.md` for the Model Registry integration roadmap).

---

## Local smoke (without cluster)

For fast local iteration:

```python
from tools.workbench import train_pragma

run = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",
    mode="local",
    local_csv_path="data/tabformer/card_transaction.v1.csv",
    output_dir="outputs/pragma-local/smoke",
    epochs=1,
    max_steps=2,
)
run.show_pipeline()
# → Stage 4 (train) should be "completed"
```

Local mode: no S3, no cluster, no distributed training. Single-node only.

---

## Monitoring and troubleshooting

**DSPA UI**: Navigate to OpenShift AI → Data Science Pipelines → Runs.
Select the Run ID to see per-component logs.

**Common failures**:

| Symptom | Likely cause |
|---------|-------------|
| `prepare_dataset` fails with `KeyError: 'ibm-tabformer'` | Adapter not registered. Check `src/pragma_encoder/data/adapters/__init__.py`. |
| `run_pretraining` fails with `EnvironmentError: AWS_S3_BUCKET not set` | S3 credentials not injected into component pod. Check `pragma-workbench-env` Secret. |
| `run_pretraining` fails with `FileNotFoundError: CSV not found` | `manifest_uri` points to a location where the CSV was not uploaded. Run `prepare_dataset` + `upload_artifacts` first. |
| Pipeline Run stuck in `RUNNING` for >10 minutes | Component pod image pull failing. Check `PRAGMA_TRAINING_IMAGE` and ImageStream. |
| `export_checkpoint` skips S3 export | `checkpoint_uri` is a local path (S3 not configured). Normal for CPU/local runs. |

**Artifact verification** after a successful run:

```bash
# Check export manifest
oc exec -n pragma-encoder deployment/pragma-encoder-workbench -- \
  aws s3 ls s3://<bucket>/pragma-encoder/runs/export/pragma-s/

# View export_manifest.json
oc exec -n pragma-encoder deployment/pragma-encoder-workbench -- \
  aws s3 cp s3://<bucket>/pragma-encoder/runs/export/pragma-s/export_manifest.json -
```

---

## Architecture notes

- **Component image**: `pragma-encoder-training:latest` — NOT the workbench image.
  Set `PRAGMA_TRAINING_IMAGE` or `PRAGMA_KFP_COMPONENT_IMAGE` before compiling.
- **No PVCs**: All data flows through S3 (ephemeral component pod scratch via `/tmp/pragma-run`).
- **Scratch dir**: `run_pretraining` stages downloads to `scratch_dir` (default `/tmp/pragma-run`).
  Override with `PRAGMA_SMOKE_BATCH_SIZE=...` at compile time for integration tests.
- **S3 credentials**: Native AWS_* env vars from OpenShift AI Connection (`pragma-workbench-env`).
  The pipeline does not reference Connection names — it reads env vars from the process.
- **Confidential computing seam**: AWS_* env vars can be substituted by attestation-based
  secret release (Trustee / CoCo) without changing any pipeline code.

---

## Integration test

Level 7 cluster integration test:

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_OPENSHIFT_AI_PIPELINE_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_07_tabformer_pipeline_runtime.py -v
```

Local pre-flight checks (no cluster, no IBM TabFormer data required):

```bash
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_07_tabformer_pipeline_runtime.py::TestTabFormerPipelineLocalPrereqs -v
```

---

## Related documentation

- `docs/training-guide.md` — general training guide (local + cluster)
- `docs/openshift-storage-pattern.md` — S3 storage decision (no PVCs)
- `docs/openshift-ai-primitives.md` — platform primitive map
- `docs/decisions/003-workbench-training-api.md` — workbench API ADR
- `docs/tech-debt.md` — TD-013 (Model Registry integration roadmap)
- `tests/openshift/README.md` — integration test maturity levels
