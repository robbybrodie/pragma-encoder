# Decision 003: Workbench-Led Training API

Status: Accepted
Date: 2026-05-19
Paper reference: Section 2.4 (Training Infrastructure)

---

## Context

The PRAGMA paper (Section 2.4) describes training infrastructure
at Revolut scale: LMDB user index, Parquet event shards, dynamic
batching, varlen FlashAttention, and 16–32× H100 distributed runs.

For this open-source implementation, training uses a custom PyTorch loop
(`scripts/train_pragma.py`) with torchrun/DDP on KFTO PyTorchJob (ADR 005).
Two interaction modes existed before this ADR:

1. **Script mode** — `python scripts/train_pragma.py` (local and
   cluster runs, already implemented)
2. **Pipeline stub mode** — `pipeline/components_pragma.py` with
   four NotImplementedError scaffolds

Neither mode is accessible from a workbench notebook without
writing OpenShift YAML or knowing the CLI arguments. Data
scientists cannot launch, inspect, or monitor training from the
workbench without this gap.

---

## Decision

Expose a `train_pragma()` Python function as the primary workbench
entry point for launching PRAGMA pretraining:

```python
from src.workbench import train_pragma

run = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",
    epochs=10,
    nodes=1,
    mode="dry_run",        # or "local", "pipeline", "auto", "cluster"
    local_csv_path=None,   # required for mode="local"
    output_dir=None,       # optional; defaults to outputs/pragma-local/<ts>
    max_steps=None,        # optional cap on training steps
    prepare_if_missing=True,
)
run.show_pipeline()   # prints the 5 pipeline steps and their status
run.metrics()         # returns training metrics dict
run.artifacts()       # returns S3 artifact URIs dict
```

### Execution modes

| Mode | Status | Behaviour |
|------|--------|-----------|
| `"dry_run"` | Implemented | Returns a `PragmaRun` preview immediately. No side effects: no adapter.prepare(), no subprocess, no S3. Useful for previewing pipeline shape and validating configuration. |
| `"local"` | Implemented | Runs `scripts/train_pragma.py` as a subprocess via `torchrun`. Requires `local_csv_path`. Writes checkpoints to `output_dir`. `nodes > 1` is not supported for local mode (raises `NotImplementedError`). |
| `"pipeline"` | Implemented | Returns a `PragmaPipeline` object capturing training intent. No training, no S3, no cluster at call time. Use `.compile(path)` to produce KFP YAML, `.show_pipeline()` to inspect stages. |
| `"auto"` | Implemented | Detects environment: if running inside an OpenShift pod, delegates to `"cluster"`; otherwise delegates to `"local"`. |
| `"cluster"` | **Not yet implemented** | Will submit a KFTO PyTorchJob via `oc apply`. Currently raises `NotImplementedError` directing users to `dry_run` or `local`. |

This is an implementation decision. The paper does not prescribe a
Python API. The design is guided by Section 2.4's pipeline stages:

1. Prepare dataset (fit tokeniser, validate)
2. Upload to S3 (idempotent)
3. Submit PyTorchJob (or run locally)
4. Monitor until completion
5. Export outputs to S3

### Supporting components

**`DatasetManifest`** (§2.4 data storage)
A frozen dataclass that describes where a prepared dataset lives
in S3. Carries the `PRAGMAConfig` so truncation limits travel
with the data reference. Replaces scattered S3 path strings.

**`DatasetAdapter`** (§2.1 dataset)
Abstract protocol for dataset-specific preparation logic. Each
adapter transforms a raw dataset source (CSV, Parquet, LMDB)
into a `DatasetManifest`. `IBMTabFormerAdapter` is the first
implementation (wraps `src/data/fit_tokenizer.py` and
`scripts/upload_training_data.py`). Future bank transaction
adapters implement the same protocol without changing
`train_pragma()`.

**`PragmaRun`**
A result object that wraps a submitted (or completed) training
job. Exposes `show_pipeline()`, `metrics()`, and `artifacts()`.
Does not hide the pipeline: `show_pipeline()` lists the five
pipeline steps with status and, if KFP is available, emits the
KFP run URL.

---

## Dataset adapter boundary

The adapter boundary decouples training from dataset format:

```
DatasetAdapter.prepare(config) → DatasetManifest
```

`train_pragma()` receives a `DatasetManifest` — it never knows
whether the data came from a CSV, Parquet, or LMDB source.

When Westpac supplies bank transaction data in their format,
a new `BankTransactionAdapter` is added. Nothing else changes.

---

## Two-node training

`nodes=2` in `train_pragma()` selects
`openshift/training/pytorchjob-pragma-s-2node.yaml`
(Master=1 pod + Worker=1 pod, each with 1 GPU).

This manifest follows the same storage pattern as the single-node
manifest: emptyDir scratch, S3 data source, `restartPolicy:
OnFailure`, init container clones repo and downloads data.

The existing `scripts/train_pragma.py` already supports
`--nnodes` via torchrun. No changes to the training script.

---

## Consequences

Enables:
- Data scientists launch training from the workbench notebook
- Pipeline steps are visible (not hidden behind YAML)
- Dataset adapter boundary makes future bank data integration
  a localised change
- Two-node DDP demonstration without RWO PVC scheduling conflicts

Constrains:
- `src/workbench/` depends on `src/data/` and `src/model/config.py`
  only — no circular imports introduced
- KFP remains an optional dependency (`pip install pragma[workbench]`)
  — `train_pragma()` works without KFP (local mode fallback)
- All S3 credentials come from `pragma-workbench-env` Secret only
  (openshift-storage-pattern.md rule 4)
- DatasetManifest is immutable once created — not modified by training

What future sessions must not contradict:
  Do not merge DatasetAdapter into train_pragma().
  Do not hardcode S3 paths — all paths via DatasetManifest.
  Do not require KFP for local development or unit tests.
  Do not use RWO PVCs for dataset or output storage.

---

## Dependency graph (this ADR)

```
src/model/config.py          → (no deps)
src/data/dataset_manifest.py → config
src/data/adapters/base.py    → dataset_manifest
src/data/adapters/ibm_tabformer.py → adapters/base, dataset_manifest
src/workbench/_run.py        → dataset_manifest
src/workbench/_api.py        → _run, adapters, config
pipeline/                    → src/ only (kfp optional import guard)
examples/workbench/          → src/workbench
```

---

## paper-to-code.md additions (mandatory after implementation)

Section 2.4 block to add:

| Paper element | This repo |
|---|---|
| Two-level data storage (LMDB + Parquet) | `src/data/dataset_manifest.py::DatasetManifest` (S3-backed; LMDB/Parquet are adapter-internal) |
| Dataset preparation pipeline | `src/data/adapters/ibm_tabformer.py::IBMTabFormerAdapter` |
| Training pipeline stages (§2.4) | `src/workbench/_api.py::train_pragma` |
| Pipeline visibility | `src/workbench/_run.py::PragmaRun` |

---

## References

- Paper: Section 2.4 (Training Infrastructure)
- Paper: Section 2.1 (Dataset)
- ADR 005: Training Orchestration (`docs/decisions/005-training-orchestration.md`)
- docs/openshift-storage-pattern.md
- docs/training-guide.md
