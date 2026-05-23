# PRAGMA Encoder — Architecture

This document is the high-level architecture anchor for the repository.
It covers two related concerns:

1. **Platform and deployment architecture** — how the package, image, and
   OpenShift AI platform relate to each other and who owns what.
2. **Model architecture** — how the PRAGMA paper maps to code.

For the model architecture jump to [PRAGMA Model Architecture](#pragma-model-architecture).

---

## Purpose

`pragma-encoder` is a platform-neutral Python training package. Its core
wheel (`pragma_encoder`) contains model code, training logic, and storage
semantics. It can be run:

- locally with `python -m pragma_encoder.training.train` or `pragma-encoder-train`
- in any container with PyTorch installed
- on OpenShift AI as a Kubeflow Training Operator (KFTO) PyTorchJob

The wheel does not contain any OpenShift AI, Kubernetes, or KFP code. Platform
integration lives in `tools/workbench/` and `pipeline/`, which are
not packaged into the wheel. The wheel is portable; the platform layer is
environment-specific.

---

## Layer model

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Workbench authoring                                         │
  │  (OpenShift AI Jupyter — tools/workbench/)     │
  └────────────────────────────┬────────────────────────────────┘
                               │ notebook → pipeline code → PR
  ┌────────────────────────────▼────────────────────────────────┐
  │  Git / PR promotion                                          │
  │  (review, CI, wheel build)                                   │
  └────────────────────────────┬────────────────────────────────┘
                               │ pip install pragma-encoder-*.whl
  ┌────────────────────────────▼────────────────────────────────┐
  │  pragma_encoder wheel                                        │
  │  (portable Python — model, training, storage adapter)        │
  └────────────────────────────┬────────────────────────────────┘
                               │ RUN pip install in Dockerfile
  ┌────────────────────────────▼────────────────────────────────┐
  │  Training image                                              │
  │  (executable container — wheel + PyTorch/CUDA stack)         │
  └────────────────────────────┬────────────────────────────────┘
                               │ openshift/gitops/ synced by ArgoCD
  ┌────────────────────────────▼────────────────────────────────┐
  │  ArgoCD-declared OpenShift AI platform state                 │
  │  (namespace, RBAC, secrets, DSPA, workbench, image builds)   │
  └────────────────────────────┬────────────────────────────────┘
                               │ RHOAI operator reconciles
  ┌────────────────────────────▼────────────────────────────────┐
  │  OpenShift AI GUI / pipeline / runtime surface               │
  │  (Workbench, DSPA, HardwareProfiles, pipeline definitions)   │
  └────────────────────────────┬────────────────────────────────┘
                               │ dynamic — created at run time
  ┌────────────────────────────▼────────────────────────────────┐
  │  Runtime jobs and artifacts                                  │
  │  (PyTorchJob pods, checkpoints in S3, pipeline runs, logs)   │
  └─────────────────────────────────────────────────────────────┘
```

Key properties of each layer:

| Layer | Key property |
|---|---|
| `pragma_encoder` wheel | Portable Python behaviour — no platform dependency |
| Training image | Executable runtime — not a source checkout |
| ArgoCD / `openshift/gitops/` | Declared platform substrate — stable long-lived state |
| OpenShift AI GUI/runtime | User-facing platform constructs — managed by RHOAI |
| Runtime jobs/checkpoints/logs | Dynamic — created and destroyed at execution time |

---

## Separation of concerns

| Layer | Owns | Must not own |
|---|---|---|
| `src/pragma_encoder` | Model code; training loop; checkpoint/resume semantics; storage adapter interface and S3/local adapters; installed CLI entrypoint (`pragma-encoder-train`) | OpenShift AI Workbench lifecycle; Kubernetes Secrets or Connections; HardwareProfiles; PyTorchJob / TrainJob YAML; ArgoCD resources; namespace / RBAC / SA lifecycle; KFP registration |
| Training image | Runtime dependencies; PyTorch/CUDA stack; installed `pragma_encoder` wheel; executable environment for `pragma-encoder-train` | Customer credentials; OpenShift AI resources; GitOps state; source checkout |
| `tools/workbench/` | Convenience helpers for Workbench, KFP, and DSPA use; platform-aware UX (`train_pragma`, `pragma_pipeline` DSL) | Being imported by or packaged into `pragma_encoder`; core training logic |
| `pipeline/` | KFP SDK v2 pipeline and component definitions; compiled IR YAML; smoke pipeline | Model implementation; training loop; checkpoint semantics |
| `openshift/` | Platform examples, aligned manifests, and runtime/test resources; Dockerfile for training image | GitOps desired-state management (belongs in `openshift/gitops/`) |
| `openshift/gitops/` | Declared platform desired state suitable for ArgoCD sync (namespace, RBAC, secrets, DSPA, workbench, image builds) | Ad hoc runtime jobs (PyTorchJob runs, pipeline runs) |
| `tests/` | Package/unit tests; packaging/wheel tests; pipeline compile tests | OpenShift cluster access; S3 credentials; GPU; registry access |
| `tests/openshift/` | Opt-in cluster runtime tests; static manifest fixture tests; image contract tests | Default test-run inclusion (all gated by `RUN_OPENSHIFT_TESTS=1`) |
| OpenShift AI runtime | Workbench Jupyter surface; DSPA pipeline server; HardwareProfile scheduling; pipeline run execution | Model implementation; training logic; checkpoint semantics |
| ArgoCD | Long-lived stable cluster resources (namespace, RBAC, secrets, DSPA, workbench, image builds) | Ad hoc runtime jobs; pipeline runs; exploratory notebook state |

---

## One OpenShift AI primitive pathway

There is one platform-facing pathway: **OpenShift AI primitives**.

The RHOAI GUI is the preferred visible surface for data scientists and operators,
but it is not a separate architecture. Other surfaces interact with the same
underlying primitives:

| Surface | Role |
|---|---|
| RHOAI GUI | User-facing surface for workbenches, pipelines, hardware profiles, connections |
| Workbench / KFP SDK | Author and submit pipeline code; interact with DSPA API |
| ArgoCD | Declare and promote stable platform substrate into the cluster |
| `oc` / `kubectl` | Direct access to the same OpenShift/Kubernetes primitives |
| CI/CD | Validate, build, and promote images and compiled pipeline artifacts |
| Tests | Verify primitives and contracts through opt-in cluster integration tests |

`pragma_encoder` does not define a deployment abstraction of its own. It is a
portable Python wheel that consumes platform-supplied configuration through
standard env vars. Platform resources are OpenShift AI primitives — not custom
pragma abstractions.

Runtime jobs (PyTorchJob training runs, pipeline runs) are dynamic objects. A
pipeline run launched from the RHOAI GUI and a pipeline run submitted by
`kfp.Client()` from the Workbench are the same OpenShift AI primitive. ArgoCD
does not own them by default.

**Placement rule:**

| What it is | Where it belongs |
|---|---|
| User/platform-facing resource | Map to an OpenShift AI/OpenShift primitive |
| Application behaviour | Keep in the wheel (`pragma_encoder`) |
| Executable runtime | Put in the training image |
| Declared platform state | Promote through ArgoCD (`openshift/gitops/`) |
| Execution instance | Treat as runtime/dynamic — not GitOps-managed |

---

## Workbench-to-wheel promotion workflow

Notebooks are the discovery and authoring surface. During exploration they
may mix concerns — that is acceptable. What is not acceptable is allowing a
notebook to remain the owner of those concerns in production.

> **Rule:** Notebooks are allowed to mix concerns during discovery.
> They are not allowed to remain the owner of those concerns in production.

Reusable behaviour must be promoted to the correct layer:

```
  notebook experiment
      │
      │  extract reusable model/training logic
      ▼
  src/pragma_encoder
      │
      │  tests written first (TDD), PR review
      ▼
  CI wheel build  (pragma_encoder-*.whl)
      │
      │  RUN pip install in Dockerfile.training
      ▼
  training image
      │
      │  KFTO PyTorchJob applies image + env
      ▼
  OpenShift AI runtime
```

Where code belongs after promotion:

| Concern | Destination |
|---|---|
| Reusable model / training behaviour | `src/pragma_encoder` |
| Repeatable workflow / pipeline stages | `pipeline/` |
| Platform configuration / manifests | `openshift/` or `openshift/gitops/` |
| Official wheel | Built by CI — not manually from a notebook |
| Workbench convenience helpers | `tools/workbench/` |

---

## Wheel-to-image-to-runtime flow

```
  source (src/pragma_encoder)
      │
      │  python -m pip wheel .
      ▼
  pragma_encoder-*.whl   (portable, no platform code)
      │
      │  RUN pip install pragma_encoder-*.whl  (Dockerfile.training)
      ▼
  training image
      │
      │  image contract: pragma-encoder-train --help exits 0
      ▼
  KFTO PyTorchJob   (openshift/training/pytorchjob-pragma-s.yaml)
      │
      │  entrypoint: pragma-encoder-train  or  python -m pragma_encoder.training.train
      ▼
  checkpoint / model artifacts in S3
```

The pipeline and PyTorchJob manifests reference the image by tag.
They must not assume any source-tree paths (`/workspace/src/`, `PYTHONPATH=.`,
or `python scripts/...`). The installed entrypoint is the contract.

---

## CPU/GPU selection model

The `--device` flag on `pragma-encoder-train` accepts application-level intent:

| Value | Meaning |
|---|---|
| `auto` (default) | Use CUDA if available, then MPS, then CPU |
| `cuda` | Require CUDA; fail if absent |
| `cpu` | Force CPU execution |
| `mps` | Use Apple Metal Performance Shaders |

OpenShift AI HardwareProfile controls actual resource allocation — it
schedules the pod onto a node with the requested GPU resources and exposes
that hardware to the container. The training code then validates what it
received and selects accordingly.

OpenShift does not override `--device`. The platform supplies hardware;
the command supplies application intent. A job scheduled onto a GPU node
with `--device auto` will use CUDA. The same image with `--device cpu` will
use CPU regardless of available hardware.

---

## Storage and checkpointing

Checkpoint and resume semantics live in `pragma_encoder.training.checkpoints`.
The module implements a `CheckpointStore` Protocol with two concrete adapters:

| Mode | Adapter | When active |
|---|---|---|
| **Local** (default) | `LocalCheckpointStore` | `AWS_S3_BUCKET`/`AWS_S3_ENDPOINT` absent, **or** `--s3-checkpoint-prefix` is empty |
| **S3** (platform/durable) | `S3CheckpointStore` | Both `AWS_*` env vars set **and** `--s3-checkpoint-prefix` non-empty |

**Local mode is the default.** The wheel works fully on a laptop with only
`--output-dir`. No S3 credentials, no OpenShift Secret, no Connection required.
`LocalCheckpointStore.put()` is a documented no-op — the training loop writes
the file to `output_dir` directly; `latest_key()` discovers it via lexicographic
sort of `checkpoint_epoch<NNNN>.pt` files.

**S3 mode is the platform/durable adapter.** On OpenShift AI, a Connection
(annotated Kubernetes Secret) injects the native `AWS_*` env vars into pods.
Combined with `--s3-checkpoint-prefix`, checkpoints are uploaded after each
epoch and can be resumed across pod restarts.

```
build_checkpoint_store(output_dir, s3_prefix)
  ├─ AWS_S3_BUCKET + AWS_S3_ENDPOINT set AND s3_prefix non-empty
  │    └─► S3CheckpointStore   (boto3 — durable, fault-tolerant)
  └─ Otherwise
       └─► LocalCheckpointStore  (filesystem — laptop/dev default)
```

The wheel reads native OpenShift AI S3 Connection environment variables when
S3 mode is active:

```
AWS_S3_BUCKET          (required)
AWS_S3_ENDPOINT        (required — full URL including scheme)
AWS_ACCESS_KEY_ID      (optional)
AWS_SECRET_ACCESS_KEY  (optional)
```

Schema source of truth: `oc get cm s3 -n redhat-ods-applications -o yaml`.
The wheel must not create Connections, Secrets, or S3 buckets. The
`pragma-workbench-env` Secret is the test fixture/default representing that
connection in this deployment — it is not the product abstraction.

| Concern | Owner |
|---|---|
| Checkpoint / resume logic | `pragma_encoder.training.checkpoints` |
| Local filesystem read/write | `LocalCheckpointStore` — default, no credentials needed |
| S3 read/write operations | `S3CheckpointStore` — platform/durable adapter |
| S3 credentials supply | OpenShift AI Connection → pod env vars |
| `pragma-workbench-env` Secret | Test fixture / platform default — not package code |
| Bucket creation | Platform operator (not the wheel) |

**RHOAI 3.4 path (current):** PyTorchJob with application-managed checkpoint/resume
over S3. Each pod downloads data from S3 at startup into `emptyDir`; the master pod
(rank 0) uploads checkpoints to S3 after each epoch. Training can resume from a
checkpoint if the pod restarts.

**KFP component staging (`run_pretraining`):** S3 data is downloaded to a configurable
`scratch_dir` (default `/tmp/pragma-run`, passed as `--scratch-dir` if overriding).
This is pod-local ephemeral storage — not a shared PVC. `--dataset-name` records the
original S3 URI in `metadata.json` separately from the local staging path
(`csv_staging_path`).  `metadata.json` is uploaded to S3 alongside the checkpoint.

**Kubeflow Trainer v2 / TrainJob:** Tech Preview in RHOAI 3.4. Evaluate separately
before adopting — not the current production path.

See `docs/openshift-storage-pattern.md` for the full storage decision record.

---

## OpenShift AI alignment

> **Platform version:** RHOAI 3.4.0 is the currently installed version
> (upgraded from 2.25.6 via controlled re-installation on 2026-05-23).
> The primitive map below reflects 3.4 semantics.

RHOAI 3.4 primitives mapped to this repo:

| RHOAI primitive | Status | Notes |
|---|---|---|
| Data Science Project | GA | `pragma-encoder` namespace |
| Workbench | GA | `openshift/gitops/workbench/notebook.yaml` |
| Connection (object storage) | GA | Supplies native `AWS_*` env vars (`AWS_S3_BUCKET`, `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) |
| HardwareProfile | GA | `infrastructure.opendatahub.io` API group in 3.4 (was `dashboard.opendatahub.io`) |
| Data Science Pipeline (DSPA) | GA | KFP v2; `openshift/gitops/pipeline/dspa.yaml` |
| PyTorchJob (`kubeflow.org/v1`) | GA | Current production training runtime |
| TrainJob / Kubeflow Trainer v2 | Tech Preview in 3.4 | Forward path — evaluate separately before adopting |
| Model serving | GA | KServe `InferenceService`; not yet wired for PRAGMA |

**GA APIs only** — the production training path uses `kubeflow.org/v1` PyTorchJob.
TrainJob is Tech Preview in RHOAI 3.4 and must not be the current production path.

Full primitive map: `docs/openshift-ai-3.3-alignment.md`
(Note: document describes 3.3 primitives; 3.4 is the installed version — primitives are compatible)

---

## ArgoCD / GitOps boundary

ArgoCD owns declared platform substrate — long-lived, stable cluster objects
that form the environment. It does not own ad hoc runtime objects.

```
  ArgoCD syncs openshift/gitops/
      ├── namespace, RBAC, Sealed Secrets      ← stable
      ├── ImageStream, BuildConfig (image)     ← stable
      ├── DSPA (pipeline server)               ← stable
      └── Workbench Notebook                   ← stable

  NOT owned by ArgoCD:
      ├── PyTorchJob runs                      ← dynamic, one-shot
      ├── Pipeline runs                        ← dynamic
      └── Exploratory notebook state           ← user workspace
```

Pipeline authoring lifecycle:

| Stage | Owner | Location |
|---|---|---|
| Pipeline Python source | Git | `pipeline/*.py` |
| Compiled pipeline IR YAML | Git (when promoted) | `pipeline/generated/` |
| Pipeline definition registration | RHOAI GUI or API | Not in Git by default |
| Pipeline run | RHOAI runtime | Dynamic — not Git-managed |

A Workbench-authored KFP decorated pipeline can be compiled to IR YAML and
committed. That compiled artifact can be GitOps-managed. Individual pipeline
runs are runtime/dynamic and are not committed.

Full ownership table and lifecycle: `docs/openshift-gitops-argocd.md`

---

## Testing model

Tests are layered. Default `pytest tests/` must complete offline with no
cluster, no credentials, and no GPU.

| Layer | Test files | Run by default | Requires |
|---|---|---|---|
| Package / unit | `tests/test_*.py` | Yes | Python + deps |
| Packaging / wheel | `tests/test_packaging.py`, `tests/test_platform_neutral_wheel.py` | Yes | Built wheel in `dist/` |
| Image contract (static) | `tests/test_image_contract.py` | Yes | Repo file reads only |
| OpenShift AI fixtures (static) | `tests/test_openshift_ai_fixtures.py` | Yes | Fixture YAML files |
| Pipeline compile (static) | `tests/openshift/test_02_pipeline_compile.py` | Yes (with `RUN_OPENSHIFT_TESTS=1`) | kfp (optional) |
| Image contract (runtime) | `tests/openshift/test_02b_image_contract_runtime.py` | Opt-in | Live cluster, training image |
| Pipeline smoke run | `tests/openshift/test_03_pipeline_smoke_run.py` | Opt-in | Live cluster, DSPA |
| PyTorchJob smoke | `tests/openshift/test_03b_training_job_smoke.py` | Opt-in | Live cluster, training image |
| S3 checkpoint/resume | `tests/openshift/test_05_s3_checkpoint_resume.py` | Opt-in | Live cluster, S3 |
| GPU training smoke | `tests/openshift/test_06_gpu_training_smoke.py` | Opt-in | Live cluster, GPU node |

Default tests must not require:
- OpenShift cluster
- S3 credentials
- GPU
- Container registry access
- ArgoCD server

Opt-in tests are gated by environment variables (`RUN_OPENSHIFT_TESTS=1`,
`RUN_PYTORCHJOB_TESTS=1`, `RUN_OPENSHIFT_S3_RESUME_SMOKE=1`, etc.).

---

## Current status

| Item | Status |
|---|---|
| `pragma_encoder` wheel | Built and installed by training image |
| `pragma_encoder.workbench` | Removed from wheel (TD-009 resolved) |
| Workbench helper code | `tools/workbench/` |
| Training CLI entrypoint | `pragma-encoder-train` / `python -m pragma_encoder.training.train` |
| Training image | Installs wheel; verified by image contract tests |
| S3 checkpoint/resume | Proven end-to-end (Level 5 test) |
| KFP pipeline code | Does not assume source-tree paths |
| Metadata contract | Hardened: `csv_staging_path`, `dataset_name`, `_safe_args_for_metadata` allowlist |
| `run_pretraining` scratch dir | Configurable `scratch_dir` parameter (not hardcoded `/tmp/pragma-run`) |
| metadata.json S3 upload | Uploaded alongside checkpoint in S3 |
| Platform | RHOAI 3.4.0 (upgraded from 2.25.6 on 2026-05-23) |
| TD-009 | Resolved 2026-05-21 |
| Test suite | 835+ passed (as of 2026-05-23) |

Next major milestone: Workbench-to-pipeline user flow end-to-end — compiled
pipeline registration in RHOAI GUI and evaluation of Kubeflow Trainer v2
for RHOAI 3.4 (TrainJob is Tech Preview; evaluate before adopting).

---

## PRAGMA Model Architecture

Maps the PRAGMA paper to this repository's implementation.

> Ostroukhov, M. et al. (2026). PRAGMA: Revolut Foundation Model.
> arXiv:2604.08649v1

### Overview

PRAGMA is an **encoder-only** foundation model for financial transaction data.
It is NOT a decoder-only model. It uses **bidirectional attention** throughout.

```
Input: Customer profile + transaction history
         │
         ├─► Profile State Encoder ──► [USR] ──────────────────┐
         │                                                       │ z = [USR : EVT₁ : EVT₂ : ...]
         └─► Event Encoder (×N events) ──► [EVT] reprs ────────┘
                                                                 │
                                           History Encoder (bidirectional self-attention,
                                                            [USR] at position 0)
                                                                 │
                                           Contextualised history (Section 2.3.4)
                                                                 │
                                       ┌─────────────────────────┴─────────────────────────┐
                                   Linear probe                               LoRA fine-tuning
                                  (Section 3.1.1)                           (Section 3.1.2)
```

### Paper section → code mapping

| Paper section | Description | Code |
|---|---|---|
| 2.2 | Key-value-time tokenisation | `src/pragma_encoder/tokenizer/` |
| 2.2 | Numerical (percentile buckets) | `src/pragma_encoder/tokenizer/numerical.py` |
| 2.2 | Categorical (single token) | `src/pragma_encoder/tokenizer/categorical.py` |
| 2.2 | Textual (BPE subword) | `src/pragma_encoder/tokenizer/textual.py` |
| 2.2 | Temporal (log-seconds + calendar) | `src/pragma_encoder/tokenizer/temporal.py` |
| 2.2 | Unified pipeline | `src/pragma_encoder/tokenizer/pipeline.py` |
| 2.3.1 | Architecture hyperparameters | `src/pragma_encoder/model/config.py` |
| 2.3.2 | Profile State Encoder | `src/pragma_encoder/encoders/profile_state_encoder.py` |
| 2.3.3 | Event Encoder | `src/pragma_encoder/encoders/event_encoder.py` |
| 2.3.3 | Calendar embeddings | `src/pragma_encoder/encoders/event_encoder.py` |
| 2.3.4 | History Encoder | `src/pragma_encoder/encoders/history_encoder.py` |
| 2.3.4 | RoPE positional encoding | `src/pragma_encoder/encoders/rope.py` |
| 2.3.5 | Three-strategy masking | `src/pragma_encoder/masking/strategy.py` |
| 2.3.5 | MLM head | `src/pragma_encoder/model/mlm_head.py` |
| 2.3.5 | MLM loss with label smoothing | `src/pragma_encoder/training/objective.py` |
| 2.4 | Sequence packing | `src/pragma_encoder/training/packing.py` |
| 2.4 | Dynamic batching | `src/pragma_encoder/training/batching.py` |
| 3 | Full PRAGMA model | `src/pragma_encoder/model/pragma.py` |
| 3.1.1 | Linear embedding probe | `src/pragma_encoder/adaptation/probe.py` |
| 3.1.2 | LoRA fine-tuning | `src/pragma_encoder/adaptation/lora.py` |
| 3 | Evaluation metrics | `src/pragma_encoder/evaluation/metrics.py` |
| 3 | Downstream evaluator | `src/pragma_encoder/evaluation/downstream.py` |

### Comparison with NVIDIA blueprint (adjacent repo)

| Property | PRAGMA (this repo) | NVIDIA blueprint (adjacent) |
|---|---|---|
| Architecture | Encoder-only | Decoder-only (GPT-style) |
| Attention | Bidirectional | Causal (left-to-right) |
| Objective | Masked modelling | Causal language modelling |
| Tokenisation | Key-value-time | Tabular text serialisation |
| Positional encoding | RoPE | Absolute (learned) |
| Encoders | 3 separate (profile/event/history) | 1 monolithic |
| Fine-tuning | LoRA via PEFT | Full fine-tuning |
| Scale (paper) | 207B tokens, 10M–1B params | Varies |

### Key architectural constraints

Non-negotiable from the paper:

1. **Encoder-only** — Never use a causal attention mask.
2. **Bidirectional** — All three encoders see the full context.
3. **RoPE** — Not sinusoidal, not absolute positional embeddings.
4. **Three separate encoders** — Profile, Event, History have separate weights.
5. **Key-value-time tokenisation** — Not text serialisation.
6. **Masked modelling** — Not causal language modelling.
