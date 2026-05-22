# OpenShift AI 3.3 Alignment

> **Note (2026-05-22):** The `MODEL_REGISTRY_*` env var naming described in this document
> has been superseded. All code, manifests, and secrets now use the native RHOAI S3
> Connection schema (`AWS_*` keys). See `docs/tech-debt.md §TD-010` for the resolution.

This document maps the `pragma-encoder` repository to Red Hat OpenShift AI (RHOAI) 3.3
primitives. It is the authoritative statement of:

- what RHOAI 3.3 provides and owns
- what `pragma_encoder` provides and owns
- what the training image provides
- how the tests are structured
- where OpenShift/RHOAI-facing configs and fixtures live
- which platform primitives correspond to current repo constructs

**Source of truth:** [RHOAI 3.3 documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/)

---

## RHOAI 3.3 Primitive Map

### 1. Data Science Project

| Field | Value |
|---|---|
| What it is | An OpenShift namespace labelled as a Data Science Project by the RHOAI dashboard |
| API | Standard Kubernetes `Namespace` — no separate CRD; label: `opendatahub.io/dashboard: "true"` |
| RHOAI 3.3 status | GA |
| How users interact | Created via RHOAI dashboard; contains workbenches, connections, pipelines, storage |

**Mapping to pragma-encoder:**
The `pragma-encoder` namespace on the cluster is the Data Science Project.
ArgoCD manages the long-lived substrate in that namespace.
Tests target `PRAGMA_TEST_NAMESPACE=pragma-encoder`.

**Owner:** OpenShift AI / OpenShift. Not owned by `pragma_encoder`.

---

### 2. Workbench

| Field | Value |
|---|---|
| What it is | A Jupyter Notebook server running inside the Data Science Project namespace |
| API | `notebooks.kubeflow.org/v1` (Kubeflow Notebook CRD, managed by RHOAI) |
| RHOAI 3.3 status | GA |
| Config | `openshift/gitops/workbench/notebook.yaml` |

**Mapping to pragma-encoder:**
The workbench runs the `pragma-encoder-workbench` custom image (built from
`openshift/notebook-image/Dockerfile`). Data scientists author pipelines,
inspect models, and launch training from the workbench using:

```python
from tools.workbench import train_pragma
```

The workbench tooling lives in `tools/workbench/` — it is importable
when `PYTHONPATH=.` is set (repo root on `sys.path`) but is **not** part of the
installed `pragma_encoder` wheel.

The workbench is the user's interactive surface. It consumes `pragma_encoder`
as a library. It does not define the cluster execution layer.

**Owner:** OpenShift AI. The workbench image is built and maintained by this repo.
`pragma_encoder` provides the Python library installed into the workbench.
`tools/workbench/` provides the workbench-facing helper layer.

---

### 3. Connection / Data Connection

| Field | Value |
|---|---|
| What it is | An OpenShift AI-managed Kubernetes Secret that injects object-storage credentials |
| API | Kubernetes `Secret` with annotations `opendatahub.io/managed: "true"` and `opendatahub.io/connection-type: s3` |
| RHOAI 3.3 status | GA |
| Env vars injected | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`, `AWS_S3_ENDPOINT`, `AWS_DEFAULT_REGION` |

**Mapping to pragma-encoder:**

The `pragma-workbench-env` Secret in this repo is a **test fixture/default**
that represents what an RHOAI object-storage Connection would inject into
workbench pods or training pods.

The `pragma_encoder` checkpoint code reads native RHOAI S3 Connection env vars:
```
AWS_S3_BUCKET          — bucket name (required)
AWS_S3_ENDPOINT        — S3 endpoint URL including scheme (required)
AWS_ACCESS_KEY_ID      — access key (optional)
AWS_SECRET_ACCESS_KEY  — secret key (optional)
```

Schema source of truth: `oc get cm s3 -n redhat-ods-applications -o yaml`.
The `MODEL_REGISTRY_*` naming used previously (pre-TD-010) has been fully removed.
See [env var contract](#env-var-contract) below.

**Owner:** OpenShift AI / OpenShift owns the Connection and credential injection.
`pragma_encoder` owns the checkpoint logic that consumes the injected env vars.
`pragma-workbench-env` is a test fixture/default — not the product abstraction.
See `tests/openshift/fixtures/` for RHOAI-aligned examples.

---

### 4. Cluster Storage

| Field | Value |
|---|---|
| What it is | PVC-backed persistent storage attached to workbenches and pipelines |
| API | Standard Kubernetes `PersistentVolumeClaim` |
| RHOAI 3.3 status | GA |

**Mapping to pragma-encoder:**

PVCs are **intentionally absent** from the training PyTorchJob manifests.
All canonical data (CSV, vocab, checkpoints, model artifacts) is stored in S3.
`emptyDir` is used as ephemeral scratch inside each pod.

This is documented in ADR 005 and the PyTorchJob manifest comments.
Avoiding PVCs eliminates the single-node scheduling constraint caused by
`ReadWriteOnce` PVC semantics.

Workbench-attached PVCs (workspace PVC) are managed by RHOAI and are separate
from training data storage.

**Owner:** OpenShift AI manages PVCs. `pragma_encoder` explicitly does not use
PVCs for training data; S3 is canonical.

---

### 5. Data Science Pipeline

| Field | Value |
|---|---|
| What it is | KFP v2-backed repeatable ML workflow orchestration |
| API | `DataSciencePipelinesApplication` CRD (`datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io`) |
| RHOAI 3.3 status | GA |
| Runtime | KFP v2 SDK; components are decorated Python functions |

**Mapping to pragma-encoder:**

`pipeline/components_pragma.py` defines the five-stage PRAGMA training pipeline
as KFP v2 decorated components:

1. Prepare dataset (fit tokenizer, validate)
2. Upload to S3 (idempotent)
3. Submit PyTorchJob
4. Monitor until completion
5. Export outputs to S3

`pipeline/pragma_smoke_pipeline.py` compiles a smoke pipeline for testing.

The DSPA instance in the cluster is managed by ArgoCD via
`openshift/gitops/pipeline/dspa.yaml`. The pipeline connects to it via
`PRAGMA_DSPA_ENDPOINT`.

Level 3 tests (`tests/openshift/test_03_pipeline_smoke_run.py`) validate
DSPA connectivity and KFP v2 pipeline smoke.

**Owner:** OpenShift AI owns the pipeline runtime (DSPA, KFP v2 server).
`pragma_encoder` and `pipeline/` own the pipeline component logic.

---

### 6. Hardware Profile

| Field | Value |
|---|---|
| What it is | Predefined CPU/GPU/resource shapes for workbenches and training |
| API | `HardwareProfile` kind, `apiVersion: infrastructure.opendatahub.io/v1` |
| RHOAI 3.3 status | **GA** (replaces deprecated Accelerator Profiles and Container Size selector) |
| How selected | Users choose a Hardware Profile in the RHOAI dashboard when creating a workbench or training job |

**Mapping to pragma-encoder:**

Hardware Profiles are the **RHOAI 3.3 construct for CPU/GPU/resource selection**.
CPU and GPU resource requests in training jobs should be driven by a Hardware Profile,
not hardcoded in Python code or test manifests.

Current state: resource requests are hardcoded in:
- `openshift/training/pytorchjob-pragma-s.yaml` (`cpu: "8"`, `memory: 64Gi`, `nvidia.com/gpu: "1"`)
- `openshift/training/pytorchjob-pragma-s-2node.yaml`
- `tests/openshift/test_04_pytorchjob_smoke.py` (smoke: `cpu: "500m"`, `memory: "2Gi"`)
- `tests/openshift/test_05_s3_checkpoint_resume.py` (smoke: `cpu: "500m"`, `memory: "2Gi"`)

For production training jobs, the Hardware Profile should determine the resource
shape. For smoke/test jobs, the lightweight CPU-only resource requests are
acceptable (they do not represent a production hardware choice).

Example Hardware Profile YAML files are in `tests/openshift/fixtures/`.

**KFP pipeline run limitation (RHOAI 3.3):**
KFP component pods (Level 3 DSPA pipeline runs) cannot directly select a HardwareProfile
via pipeline parameters. `HardwareProfile.spec.identifiers` applies to workbench
notebooks (Notebook CR) and KFTO training jobs, not to KFP component pods.
KFP component pod resources are set via `@dsl.component` resource limits or cluster
defaults. The smoke pipeline (`pipeline/pragma_smoke_pipeline.py`) correctly omits
HardwareProfile selection — this is the expected design, not a gap.

**Owner:** OpenShift AI / RHOAI administrators define Hardware Profiles.
`pragma_encoder` Python code must not decide GPU vs CPU deployment shape.
Training manifests should reference Hardware Profile labels or node selectors
where appropriate.

---

### 7. Custom Workbench Image / Runtime Image

| Field | Value |
|---|---|
| What it is | Custom container image added to RHOAI for use as a notebook or training runtime |
| API | OpenShift `ImageStream`, or direct registry URL configured in RHOAI settings |
| RHOAI 3.3 status | GA |

**Mapping to pragma-encoder:**

Two custom images are built and managed by this repo:

| Image | Purpose | Dockerfile |
|---|---|---|
| `pragma-encoder-workbench` | Notebook image for data scientists | `openshift/notebook-image/Dockerfile` |
| `pragma-encoder-training` | Training runtime image | `openshift/training/Dockerfile.training` |

The training image installs the `pragma_encoder` wheel (`pip install --no-deps dist/pragma_encoder-*.whl`)
and copies `scripts/` into the image. It does not contain the source tree.

Built via `oc start-build pragma-encoder-training --from-dir=.`.
The image contract is verified by `tests/openshift/test_02b_image_contract_runtime.py`.

**Owner:** This repo maintains both images. RHOAI imports them via ImageStream or registry.
`pragma_encoder` provides the installable Python wheel; the image provides the runtime.

---

### 8. Distributed Workloads — PyTorchJob (GA)

| Field | Value |
|---|---|
| What it is | N-node distributed PyTorch training on Kubernetes |
| API | `PyTorchJob`, `apiVersion: kubeflow.org/v1` (Kubeflow Training Operator v1) |
| RHOAI 3.3 status | **GA** |
| Operator | KFTO (Kubeflow Training Operator), managed by RHOAI |

This is the **current GA primitive for distributed training in RHOAI 3.3**.

**Mapping to pragma-encoder:**

Production manifests: `openshift/training/pytorchjob-pragma-s.yaml`,
`pytorchjob-pragma-m.yaml`, `pytorchjob-pragma-s-2node.yaml`.

Test smoke: Level 4 (`tests/openshift/test_04_pytorchjob_smoke.py`),
Level 5 (`tests/openshift/test_05_s3_checkpoint_resume.py`).

All test-created PyTorchJobs carry test labels for cleanup isolation.
`pragma_encoder` training code is indifferent to whether it runs in a
PyTorchJob, a batch Job, or locally — it reads standard `torchrun` env vars
(`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).

**Owner:** OpenShift AI / Kubernetes owns PyTorchJob execution.
`pragma_encoder` owns the training logic; it does not render or submit jobs.

---

### 9. Kubeflow Trainer v2 / TrainJob (Tech Preview)

| Field | Value |
|---|---|
| What it is | Unified distributed training API replacing per-framework CRDs |
| API | `TrainJob`, `apiVersion: trainer.kubeflow.org/v1alpha1` |
| RHOAI 3.3 status | **Technology Preview** (introduced in RHOAI 3.2 TP) |
| Also introduces | `TrainingRuntime`, `ClusterTrainingRuntime` |

**RHOAI 3.3 status: Tech Preview. Do not use in production.**

Kubeflow Trainer v2 replaces separate per-framework CRDs (`PyTorchJob`,
`TFJob`, etc.) with a unified `TrainJob` resource. A Python SDK allows
programmatic job creation.

**Mapping to pragma-encoder:**

The repo currently uses `PyTorchJob` (`kubeflow.org/v1`) — the GA API.
`TrainJob` is documented here as the **forward-looking direction to evaluate**
once it reaches GA in a future RHOAI release.

Do not introduce `TrainJob` into production manifests or CI tests until it
is GA in RHOAI. When evaluating TrainJob, place examples under
`tests/openshift/fixtures/trainjob-example.yaml`.

See ADR 005 (`docs/decisions/005-training-orchestration.md`) for the
GA-only API constraint.

---

### 10. TrainingRuntime / ClusterTrainingRuntime

| Field | Value |
|---|---|
| What it is | Reusable training configuration blueprints used by Kubeflow Trainer v2 |
| API | `TrainingRuntime` / `ClusterTrainingRuntime`, `apiVersion: trainer.kubeflow.org/v1alpha1` |
| RHOAI 3.3 status | **Technology Preview** (part of Kubeflow Trainer v2) |

Not used by this repo. Future consideration alongside TrainJob.

---

### 11. Model Serving / Deployed Model

| Field | Value |
|---|---|
| What it is | Online model inference endpoint |
| API | `InferenceService`, `apiVersion: serving.kserve.io/v1beta1` (KServe) |
| RHOAI 3.3 status | GA (KServe); ModelMesh also available |
| Config | `openshift/serving/` |

**Mapping to pragma-encoder:**

PRAGMA produces embedding vectors, not predictions. Embedding extraction is
currently a script/batch path, not an online inference endpoint.

`openshift/serving/` contains serving config scaffolding for future use.
No current tests cover model serving.

**Owner:** OpenShift AI owns serving runtime. Future work.

---

## Responsibility Boundaries

### OpenShift AI / OpenShift owns

| Concern | Primitive |
|---|---|
| Project / namespace boundary | Data Science Project (labelled Namespace) |
| Workbench / IDE environment | Workbench (`notebooks.kubeflow.org/v1`) |
| Object storage credential injection | Connection (annotated Kubernetes Secret) |
| Persistent scratch storage | Cluster Storage (PVC) |
| Pipeline orchestration runtime | Data Science Pipeline (DSPA / KFP v2) |
| CPU/GPU/resource shape selection | Hardware Profile (`infrastructure.opendatahub.io/v1`) |
| Distributed job scheduling | Kubeflow Training Operator / PyTorchJob (`kubeflow.org/v1`) |
| Service accounts, RBAC, namespace policy | OpenShift / ArgoCD |
| GPU scheduling, node affinity | OpenShift scheduler + Hardware Profile |
| Image selection at runtime | RHOAI dashboard / manifest |
| Model serving / inference | KServe `InferenceService` |

### `pragma_encoder` wheel owns

| Concern | Code |
|---|---|
| Model architecture | `src/pragma_encoder/model/` |
| Training loop | `pragma_encoder.training.train` (module) |
| Checkpoint save and local load | `src/pragma_encoder/training/checkpoints.py` |
| S3 upload / download / list / latest discovery | `src/pragma_encoder/training/checkpoints.py` |
| Distributed rank-safe resume | `src/pragma_encoder/training/checkpoints.py` |
| Tokenizer, encoder, masking, MLM | `src/pragma_encoder/` |
| CLI / module entrypoints | `pragma-encoder-train`, `python -m pragma_encoder.training.train`, `python -m pragma_encoder.data.fit_tokenizer` |
| Pipeline component logic | `pipeline/components_pragma.py` |

### Training image owns

| Concern | How |
|---|---|
| Python runtime | Inherited from workbench base image |
| PyTorch / CUDA / dependencies | Inherited from workbench base image |
| `pragma_encoder` wheel | `pip install --no-deps dist/pragma_encoder-*.whl` |
| Training script entrypoint | `pragma-encoder-train` (console script); `scripts/train_pragma.py` compatibility wrapper |

### Tests own

| Concern | Location |
|---|---|
| Package/unit/wheel tests | `tests/` (no OpenShift dependency) |
| S3 checkpoint/resume unit tests | `tests/test_checkpoint_resume.py` |
| OpenShift substrate checks | `tests/openshift/test_0*` |
| Image contract | `tests/openshift/test_02b_image_contract_runtime.py` |
| Pipeline compile | `tests/openshift/test_02_pipeline_compile.py` |
| Runtime smoke tests | `tests/openshift/test_03b_*, test_04_*, test_05_*, test_06_*` |
| RHOAI-aligned fixture examples | `tests/openshift/fixtures/` |

---

## Env Var Contract

`pragma_encoder` training code reads the following env vars from the pod environment.
In production, these are injected by an OpenShift AI Connection. The
`pragma-workbench-env` Secret is the test fixture/default representing that
connection in this deployment.

### S3 / Object Storage

| Env var | Source | Description |
|---|---|---|
| `MODEL_REGISTRY_BUCKET` | Secret / Connection | S3 bucket name |
| `MODEL_REGISTRY_ENDPOINT` | Secret / Connection | S3 endpoint URL (e.g. `https://s3.example.com`) |
| `MODEL_REGISTRY_ACCESS_KEY` | Secret / Connection | S3 access key ID |
| `MODEL_REGISTRY_SECRET_KEY` | Secret / Connection | S3 secret access key |

**Alignment gap:** Standard RHOAI object-storage Connections inject `AWS_*` env vars
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`, `AWS_S3_ENDPOINT`,
`AWS_DEFAULT_REGION`). The current code uses `MODEL_REGISTRY_*` naming.

When migrating to a formal RHOAI Connection, either:
- Create the Connection with custom key names (`MODEL_REGISTRY_*`), or
- Update `checkpoints.py` to read `AWS_S3_BUCKET` / `AWS_S3_ENDPOINT` etc.

This is a future alignment task, not a blocker for current testing.

### Pipeline / DSPA

| Env var | Source | Description |
|---|---|---|
| `PRAGMA_DSPA_ENDPOINT` | Set at pipeline submission time | KFP v2 API endpoint URL |

### Training job (injected by KFTO via torchrun)

| Env var | Source | Description |
|---|---|---|
| `RANK` | KFTO / torchrun | Global rank of this process |
| `LOCAL_RANK` | KFTO / torchrun | Local rank within this node |
| `WORLD_SIZE` | KFTO / torchrun | Total number of processes |
| `MASTER_ADDR` | KFTO | Master node hostname |
| `MASTER_PORT` | KFTO | Master node port |

`pragma_encoder` reads these via `torch.distributed` init — no custom code needed.

---

## `pragma-workbench-env` Secret — Role and Status

The `pragma-workbench-env` Kubernetes Secret is:

- A **test fixture/default** representing what an RHOAI object-storage Connection
  would inject into workbench and training pods
- Created manually (from template) or via Sealed Secrets in GitOps
- Consumed by training pods via `envFrom.secretRef`
- **Not the product abstraction** — it is a stand-in until formal RHOAI Connections
  are configured for this project

An RHOAI-aligned Connection YAML example is at:
`tests/openshift/fixtures/object-storage-connection.yaml.template`

The template `openshift/secrets/workbench-secret.template.yaml` documents the
expected keys. The sealed production Secret is at
`openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml`.

**Secret data is never printed, logged, or asserted on in tests.**

---

## Storage Adapter Boundary

`pragma_encoder` separates storage transport from checkpoint semantics via the
`CheckpointStore` Protocol in `src/pragma_encoder/training/checkpoints.py`.

### What this boundary enforces

| Layer | What it owns |
|---|---|
| `CheckpointStore` (Protocol) | Interface: `latest_key()`, `fetch()`, `put()` |
| `LocalCheckpointStore` | Filesystem operations only — no S3 calls |
| `S3CheckpointStore` | boto3 S3 operations — no local filesystem assumptions beyond `output_dir` |
| `build_checkpoint_store()` | Factory: reads `MODEL_REGISTRY_*` env vars, returns correct adapter |
| `resolve_resume_checkpoint()` | Distributed rank coordination — calls storage adapter, does not know transport details |
| `train.py` | Calls adapter `put()` after each checkpoint save; calls `resolve_resume_checkpoint()` on resume |

### What the boundary prevents

- Training code does not contain S3-specific logic — it calls the `CheckpointStore` interface
- The `CheckpointStore` adapters do not know which Kubernetes Secret supplied `MODEL_REGISTRY_*`
- `pragma-workbench-env` (the Secret name) does not appear in `src/pragma_encoder` — it is a
  platform fixture (`openshift/secrets/`, `tests/openshift/fixtures/`) only
- `LocalCheckpointStore.put()` is a deliberate no-op — the training loop saves files locally
  before calling `put()`, so there is nothing to persist again

### Storage adapter selection logic

```
build_checkpoint_store(output_dir, s3_prefix)
  ├── MODEL_REGISTRY_BUCKET set AND MODEL_REGISTRY_ENDPOINT set AND s3_prefix non-empty
  │   └── returns S3CheckpointStore(config, s3_prefix)
  └── otherwise
      └── returns LocalCheckpointStore(output_dir)
```

An empty `s3_prefix` disables S3 even when env vars are present. This is the
correct single-node / CI / local development behaviour.

### `pragma-workbench-env` is a platform fixture, not a product abstraction

The `pragma-workbench-env` Secret is:
- Defined in `openshift/secrets/workbench-secret.template.yaml`
- Referenced in `tests/openshift/fixtures/` as the cluster-side Secret name
- **Not referenced in `src/pragma_encoder/`** — the package reads `MODEL_REGISTRY_*`
  env vars from the process environment and does not care which Secret provided them

Mechanical enforcement: `TestPlatformNameBoundary` in `tests/test_checkpoint_resume.py`
scans all `src/pragma_encoder/*.py` files and fails if `pragma-workbench-env` appears.

---

## Hardware Profile — Current and Target State

### Current state

Resource requests are hardcoded in YAML manifests:
- Production PyTorchJobs: `cpu: "8"`, `memory: 64Gi`, `nvidia.com/gpu: "1"`
- Smoke test jobs: `cpu: "500m"`, `memory: "2Gi"` (CPU only, no GPU)

### Target state (RHOAI 3.3 alignment)

Production training jobs should reference an RHOAI Hardware Profile.
The Hardware Profile defines the approved resource shape; the manifest
references it by label or node selector.

Example Hardware Profile YAML files are in `tests/openshift/fixtures/`:
- `hardware-profile-cpu-smoke.yaml` — lightweight CPU shape for smoke tests
- `hardware-profile-gpu-pragma-s.yaml` — GPU shape for PRAGMA-S production training

Until Hardware Profiles are configured in the cluster, the hardcoded resource
requests in the production manifests are acceptable. Smoke test resource
requests (`cpu: "500m"`, `memory: "2Gi"`) are intentionally lightweight
and do not represent a hardware profile choice.

**`pragma_encoder` Python code must not decide CPU vs GPU deployment shape.**
Hardware selection belongs to the OpenShift AI platform layer.

---

## Data Science Pipelines — Current and Target State

### Current state

The five-stage pipeline is expressed as KFP v2 components in
`pipeline/components_pragma.py`. The pipeline compiles to YAML and can be
uploaded to the DSPA API.

Level 3 tests validate DSPA connectivity and run the full end-to-end pipeline smoke
(`TestKFPPipelineSmoke`). The Level 3 smoke (`pragma_smoke_training_pipeline`) is
implemented and ready to run against a cluster with `RUN_OPENSHIFT_PIPELINE_SMOKE=1`.

### Target state

The Data Science Pipeline is an OpenShift AI primitive (managed by the DSPA).
Submitting KFP pipelines via the RHOAI dashboard, the KFP SDK, or the workbench
API are all surfaces over this same primitive — there is one platform pathway.
The pipeline accepts these parameters:

| Parameter | Description |
|---|---|
| `training_image` | RHOAI training image (from image registry) |
| `hardware_profile` | Hardware Profile name or resource shape |
| `object_storage_connection` | RHOAI Connection name for S3 |
| `checkpoint_prefix` | S3 key prefix for checkpoints |
| `nnodes` | Number of training nodes |
| `resume` | Resume from latest checkpoint (`true`/`false`) |
| `max_steps` | Maximum training steps |
| `batch_size` | Batch size per rank |
| `dataset_location` | S3 URI of prepared dataset |

This is a **contract note**, not a new framework. These parameters describe
what a workbench or pipeline would pass to the training infrastructure.
They are not owned by `pragma_encoder`.

---

## Distributed Training — PyTorchJob vs TrainJob Direction

| Approach | API | RHOAI 3.3 status | Repo status |
|---|---|---|---|
| **PyTorchJob** | `kubeflow.org/v1` | **GA** | Current — all tests and manifests use this |
| **TrainJob** (Kubeflow Trainer v2) | `trainer.kubeflow.org/v1alpha1` | **Tech Preview** | Not used — see ADR 005 |

Do not remove PyTorchJob tests before there is a GA TrainJob replacement.

When TrainJob reaches GA in a future RHOAI release:
1. Write a new ADR superseding ADR 005
2. Add TrainJob examples under `tests/openshift/fixtures/`
3. Implement Level 4b tests against TrainJob
4. Keep PyTorchJob tests until TrainJob smoke is green

---

## What Belongs Where — Rules

### Must stay in `pragma_encoder` (wheel)

- Model architecture code
- Training loop
- Checkpoint save/load and S3 operations
- Distributed rank-safe resume
- Tokenizer, encoder, masking
- `fit_tokenizer` module
- Pipeline component logic
- Unit tests for all of the above

### Must stay under `tests/openshift/`

- Cluster substrate checks (Levels 0–1)
- Image contract tests (Level 2b)
- Pipeline compile/runtime tests (Level 2–3)
- PyTorchJob smoke tests (Level 4)
- S3 checkpoint/resume smoke (Level 5)
- GPU training smoke (Level 6)
- RHOAI fixture examples (`tests/openshift/fixtures/`)

### Must NOT go into the wheel

- OpenShift AI Connections
- Hardware Profiles
- Data Science Project definitions
- Service accounts
- Namespace assumptions
- PyTorchJob / TrainJob YAML
- KFP pipeline runtime binding
- Cluster-specific config
- Secret references or credential handling beyond env var reads

### Must NOT go into generic `tests/`

- OpenShift cluster checks
- PyTorchJob manifest tests
- Image contract checks
- Runtime smoke tests requiring cluster access

---

## Platform Neutrality

### Core wheel is platform-neutral

The `pragma_encoder` wheel is fully platform-neutral. All subpackages contain
no platform-specific code (TD-009 resolved — 2026-05-21):

| Subpackage | Platform-neutral? | Notes |
|---|---|---|
| `pragma_encoder.tokenizer` | Yes | Pure Python + numpy |
| `pragma_encoder.encoders` | Yes | Pure PyTorch |
| `pragma_encoder.masking` | Yes | Pure PyTorch |
| `pragma_encoder.model` | Yes | Pure PyTorch |
| `pragma_encoder.adaptation` | Yes | PyTorch + PEFT |
| `pragma_encoder.training` | Yes | PyTorch + boto3 (S3 is platform-neutral storage) |
| `pragma_encoder.evaluation` | Yes | Pure Python |
| `pragma_encoder.data` | Yes | Python + pandas |

Platform terms in core module docstrings (e.g. "KFTO PyTorchJob" in `train.py`,
"Kubernetes Secret" in `checkpoints.py`) are **explanatory context only**. They
describe the deployment environment; they do not create platform dependencies.

### Workbench tooling lives outside the wheel (TD-009 resolved)

The platform-aware workbench helpers have been moved to `tools/workbench/`
— a directory that is **not packaged into the wheel** (`[tool.setuptools.packages.find]
where = ["src"]` excludes `tools/`). This fully resolves TD-009.

`tools/workbench/` reads Kubernetes SA tokens, constructs DSPA endpoint
URLs from namespace information, and wraps `kfp.Client` for pipeline submission.
It is platform-aware by design and is explicitly excluded from the core wheel.

The boundary is enforced mechanically:

1. **`import pragma_encoder.workbench` raises `ModuleNotFoundError`** — verified by
   `tests/test_platform_neutral_wheel.py::TestWorkbenchRemovedFromWheel`.

2. **`kfp` and `kfp-kubernetes` are NOT in `pyproject.toml` dependencies** — neither
   in `[project.dependencies]` nor in `[project.optional-dependencies]`. They are
   listed in `openshift/notebook-image/requirements.txt` for the workbench image only.

3. **`tests/test_platform_neutral_wheel.py` enforces the full boundary** mechanically:
   - No kfp import statements in any core modules
   - No Kubernetes pod paths (`/var/run/secrets/kubernetes.io/`) in core source
   - No OpenShift CRD resource names in core source (ArgoCD, InferenceService, etc.)
   - `import pragma_encoder.workbench` raises `ModuleNotFoundError`
   - `tools.workbench` is importable when `PYTHONPATH=.`

### Module classification

| Module | Purpose | Platform-neutral? |
|---|---|---|
| `pragma_encoder.training.train` | Training entrypoint + DDP loop | Yes |
| `pragma_encoder.training.checkpoints` | Checkpoint semantics + storage adapters | Yes |
| `pragma_encoder.model` | PRAGMA architecture | Yes |
| `pragma_encoder.tokenizer` | Key-value-time tokenisation | Yes |
| `pragma_encoder.encoders` | Three-encoder architecture | Yes |
| `pragma_encoder.masking` | Three-strategy MEM masking | Yes |
| `tools.workbench._submit` | DSPA/KFP endpoint + auth | No — K8s paths, kfp.Client |
| `tools.workbench._api` | Cluster/local mode dispatch | Partial — checks KUBERNETES_SERVICE_HOST |
| `tools.workbench._run` | PragmaRun result object | Yes — pure Python |
| `tools.workbench._decorators` | Pipeline authoring DSL | Mostly — platform refs in docstrings only |
| `tools.workbench._intent` | Intent value objects | Yes — pure Python |

TD-009 in `docs/tech-debt.md` is **resolved**. The workbench helpers were moved to
`tools/workbench/` and removed from the `pragma_encoder` distribution.

---

## References

- [RHOAI 3.3 documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/)
- [RHOAI 3.3 release notes](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/html/release_notes/index)
- [Kubeflow Trainer v2 blog](https://www.redhat.com/en/blog/resilient-model-training-red-hat-openshift-ai-kubeflow-trainer)
- ADR 005: `docs/decisions/005-training-orchestration.md`
- ADR 003: `docs/decisions/003-workbench-training-api.md`
- OpenShift storage pattern: `docs/openshift-storage-pattern.md`
- Image contract: `docs/openshift-image-contract.md`
- Tests overview: `tests/openshift/README.md`
- Fixture examples: `tests/openshift/fixtures/README.md`
- Platform boundary tests: `tests/test_platform_neutral_wheel.py`
- Storage adapter boundary tests: `tests/test_checkpoint_resume.py` — `TestPlatformNameBoundary`
- Tech debt register: `docs/tech-debt.md` — TD-009
