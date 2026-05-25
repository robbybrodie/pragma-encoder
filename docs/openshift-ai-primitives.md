# OpenShift AI Primitive Pathway — PRAGMA Encoder

This document is the authoritative primitive map for the `pragma-encoder` project
on Red Hat OpenShift AI (RHOAI). It answers: **what OpenShift AI/OpenShift
primitive does each concern map to, who owns it, does ArgoCD manage it, and
where does it live in this repo?**

---

## One platform pathway

There is one platform-facing pathway:

```
Git
  -> ArgoCD / OpenShift GitOps
  -> declared OpenShift AI / OpenShift primitives
  -> OpenShift AI GUI / Workbench / KFP SDK / ArgoCD / oc / CI / tests
  -> dynamic runtime jobs and artifacts
```

**The GUI, Workbench SDK, KFP SDK, ArgoCD, `oc`, CI/CD, and tests are all
surfaces over the same underlying OpenShift AI primitives.** They do not
define separate deployment paths.

ArgoCD promotes **stable declared resources** into the cluster. The RHOAI
dashboard (GUI), Workbench, `oc`, and tests interact with those same resources.

---

## Primitive Map

| Concern | OpenShift AI / OpenShift primitive | Current repo file(s) | ArgoCD owned? | Status | Notes |
|---|---|---|---|---|---|
| **Data Science Project** | `Namespace` with `opendatahub.io/dashboard: "true"` | `openshift/gitops/namespace.yaml` | Yes — wave −1 | GA | RHOAI dashboard lists this namespace as a Data Science Project |
| **Workbench / Notebook** | `notebooks.kubeflow.org/v1 Notebook` | `openshift/gitops/workbench/notebook.yaml` | Yes — wave 4 | GA | Runs custom workbench image; clones repo into PVC on first start |
| **Object storage** | `Connection` (annotated `Secret` with `opendatahub.io/connection-type: s3`) | `tests/openshift/fixtures/object-storage-connection.yaml.template` | ArgoCD owns template / SealedSecret; not raw credentials | GA | Canonical `AWS_*` key names — aligned with native RHOAI S3 Connection schema (TD-010 resolved) |
| **Object storage credentials** | SealedSecret → Secret (via Sealed Secrets operator) | `openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml` | Yes — wave 0 | GA | Production credentials sealed with kubeseal; never committed as plaintext |
| **Registry pull secret** | `kubernetes.io/dockerconfigjson` Secret (SealedSecret) | `openshift/gitops/secrets/registry-pull-secret.sealed.yaml` | Yes — wave 0 | GA | NGC registry credentials for nvcr.io base images |
| **RBAC / ServiceAccounts** | `ServiceAccount`, `ClusterRole`, `RoleBinding` | `openshift/gitops/rbac/` | Yes — wave 0 | GA | Includes anyuid SCC grant for training pods, ArgoCD admin binding |
| **CPU resource shape** | `HardwareProfile` (`infrastructure.opendatahub.io/v1`) | `tests/openshift/fixtures/hardware-profile-cpu-smoke.yaml` | RHOAI admin applies; example only in fixtures | GA (RHOAI 3.4) | Lightweight shape for smoke tests; no GPU |
| **GPU resource shape** | `HardwareProfile` (`infrastructure.opendatahub.io/v1`) | `tests/openshift/fixtures/hardware-profile-gpu-pragma-s.yaml` | RHOAI admin applies; example only in fixtures | GA (RHOAI 3.4) | PRAGMA-S shape: 1 GPU, 8 CPU, 64Gi RAM per node |
| **Data Science Pipeline** | `DataSciencePipelinesApplication` | `openshift/gitops/pipeline/dspa.yaml` | Yes — wave 3 | GA | KFP v2 server; pipeline components defined in `pipeline/` |
| **Pipeline source** | Python (KFP SDK v2 decorators) | `pipeline/components_pragma.py`, `pipeline/pragma_smoke_pipeline.py` | No — Python source, not cluster resource | GA | Compiled to YAML; registered with DSPA via API |
| **Pipeline compiled artifact** | Pipeline definition/version IR YAML | `pipeline/dist/` (generated) | Optionally — when promoted to `openshift/gitops/pipelines/` | GA | Not committed by default; promoted via GitOps as team choice |
| **Pipeline run** | KFP PipelineRun (dynamic) | — | **No** — dynamic runtime object | GA | Created by RHOAI GUI, KFP API, or workbench SDK; not ArgoCD-owned by default |
| **Workbench image** | `BuildConfig` + `ImageStream` (OpenShift S2I) | `openshift/gitops/notebook-image/` | Yes — wave 2 | GA | Builds custom JupyterLab image from `openshift/notebook-image/Dockerfile` |
| **Training image** | OCI image (built by `oc start-build`) | `openshift/training/Dockerfile.training` | No — built manually or by CI | GA | Installs `pragma_encoder` wheel; used by all training jobs |
| **Distributed training (current project path)** | `PyTorchJob` (`kubeflow.org/v1`) | `openshift/training/pytorchjob-pragma-s.yaml`, `pytorchjob-pragma-s-2node.yaml`, `pytorchjob-pragma-m.yaml` | No — one-shot runtime job | current proven project path; verify CRD availability on target cluster | All manifests use wheel-based entrypoint; no git clone; upstream v1 source removed Feb 2025 — verify `oc api-resources \| grep kubeflow` before Level 5 tests |
| **Distributed training (forward evaluation path)** | `TrainJob` (`trainer.kubeflow.org/v1alpha1`) | `tests/openshift/fixtures/trainjob-example.yaml` | No — example only | **Technology Preview** in RHOAI 3.4 unless GA confirmed | available as Kubeflow Trainer v2; evaluate platform-native checkpointing before replacing current PyTorchJob path |
| **Model serving** | `InferenceService` (`serving.kserve.io/v1beta1`) + `ServingRuntime` | `openshift/serving/inference-service.yaml`, `openshift/serving/serving-runtime.yaml` | Yes, when serving is standardized | GA (KServe) | Future work; currently embedding extraction is batch-only |
| **GitOps controller** | ArgoCD `Application` (`argoproj.io/v1alpha1`) | `openshift/argocd/application.yaml` | Bootstrapped manually once; then self-managed | GA | Syncs `openshift/gitops/` to `pragma-encoder` namespace |

---

## ArgoCD Ownership Classification

ArgoCD owns **stable, declared platform state**. It does not own dynamic
runtime objects.

### ArgoCD owns (by default)

| Resource | Reason |
|---|---|
| Data Science Project namespace | Stable platform boundary |
| RBAC, ServiceAccounts | Stable security posture |
| Connection templates / SealedSecrets | Stable credential scaffolding |
| HardwareProfile examples (when promoted) | Approved compute shapes |
| DataSciencePipelinesApplication | Stable pipeline runtime |
| Workbench Notebook CR + PVC | Stable authoring environment |
| BuildConfig + ImageStream | Stable image build pipeline |
| ServingRuntime / InferenceService (when standardized) | Stable serving config |
| Promoted compiled pipeline artifacts | Optional stable promoted artifacts |

### ArgoCD does NOT own (by default)

| Resource | Reason |
|---|---|
| Ad hoc PipelineRuns | Dynamic execution objects |
| Individual training runs | One-shot user-initiated jobs |
| Transient PyTorchJobs / TrainJobs | Runtime dispatch objects |
| Pods, logs | Ephemeral runtime state |
| Checkpoints, experiment outputs | S3-resident training artifacts |
| Exploratory notebook state | User workspace, not platform config |

ArgoCD **may own** examples or templates for runtime resources (e.g. a reference
PyTorchJob YAML), but those must be clearly marked as examples/templates, not
as dynamic run objects.

---

## Object Storage — Connection Contract

**Primitive:** OpenShift AI Connection (annotated Kubernetes Secret).

On OpenShift AI, object storage is supplied through an **OpenShift AI Connection**.
The Connection materializes as a Kubernetes Secret annotated with:

```yaml
opendatahub.io/managed: "true"
opendatahub.io/connection-type: s3
```

RHOAI injects the Secret into workbench pods and training pods as environment
variables.

### Naming alignment (TD-010 resolved 2026-05-22)

The training code reads native RHOAI S3 Connection env var names:
`AWS_S3_BUCKET`, `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.

The SealedSecret was re-sealed with these canonical keys. The legacy
`MODEL_REGISTRY_*` fallback code has been removed from the codebase.

### `pragma_encoder` boundary rule

- `pragma_encoder` consumes env vars from the process environment.
- `pragma_encoder` does not create or own Connections, Secrets, buckets,
  namespaces, or credential lifecycle.
- The Secret name `pragma-workbench-env` does not appear in `src/pragma_encoder/`.
- The Connection fixture `tests/openshift/fixtures/object-storage-connection.yaml.template`
  is an **example only** — never committed with real credentials.

---

## Compute / Hardware — HardwareProfile Contract

**Primitive:** `HardwareProfile` (`infrastructure.opendatahub.io/v1`), GA in RHOAI 3.4.

HardwareProfile is the OpenShift AI primitive for CPU/GPU/resource shape selection.
It replaces deprecated Accelerator Profiles and the Container Size selector.

### Fixtures in this repo

| File | Purpose | GPU? |
|---|---|---|
| `tests/openshift/fixtures/hardware-profile-cpu-smoke.yaml` | Lightweight CPU shape for smoke tests | No |
| `tests/openshift/fixtures/hardware-profile-gpu-pragma-s.yaml` | PRAGMA-S production shape (1 GPU/node) | Yes — `nvidia.com/gpu` |

Raw `nvidia.com/gpu` resource requests appear in runtime manifests and test
fixtures as implementation detail. They should align with an approved
HardwareProfile when the profile is applied to the cluster.

### `pragma_encoder` boundary rule

- `pragma_encoder` Python code does not decide GPU vs CPU deployment shape.
- Training code validates the device it receives (`--device auto` selects the
  best available device at runtime).
- HardwareProfile is a platform resource; it does not appear in `src/pragma_encoder/`.

---

## Distributed Training — PyTorchJob vs TrainJob

| Approach | API | RHOAI 3.4 status | Repo status |
|---|---|---|---|
| **PyTorchJob** | `kubeflow.org/v1` | Current proven project path — verify availability and support status on the target RHOAI 3.4 cluster before relying on it | Current — all production manifests and tests |
| **TrainJob** (Kubeflow Trainer v2) | `trainer.kubeflow.org/v1alpha1` | **Technology Preview** in RHOAI 3.4 unless GA is confirmed for the target deployment | Evaluation example — `tests/openshift/fixtures/trainjob-example.yaml` |

**PyTorchJob is the current proven project path. TrainJob is available in RHOAI 3.4 as Technology Preview; do not use in production until GA is confirmed in the target deployment.**

### PyTorchJob upstream deprecation notice (2026-05-24)

The upstream Kubeflow Training Operator v1 source code has been **removed** from
the `kubeflow/trainer` repository (Feb 2025, PR #2389). Kubeflow docs redirect
all v1 documentation pages to v2 with deprecation warnings.

**RHOAI 3.4 cluster verification required before running Level 5 cluster tests:**

```bash
oc api-resources | grep kubeflow    # must show pytorchjobs
oc api-resources | grep trainer     # should show trainjobs (Tech Preview)
```

If `kubeflow.org/v1` CRDs are absent from the cluster, Level 5 cluster tests
will fail and urgent migration to TrainJob is required. See TD-012 in
`docs/tech-debt.md` for the full evaluation plan.

**Do not assume either conclusion — verify on the cluster first.**

### TrainJob evaluation (TD-012)

TrainJob is being evaluated as the forward platform path for RHOAI 3.4+.
Key findings from the initial evaluation (`docs/rhoai-3.4-trainjob-checkpointing.md`):

- **Resilient checkpointing** (JIT + periodic) requires HuggingFace trainers for SDK-native support
- **PRAGMA custom loop gaps:** no SIGTERM handler, no checkpoint-dir env reading, PVC backend conflicts ADR 003
- **S3 checkpoint backend** for TrainJob is unconfirmed from official docs — [VERIFY]
- **Level 5** (S3 checkpoint/resume) remains valid for PyTorchJob and custom loop paths

When TrainJob reaches GA: write a new ADR superseding ADR 005, add Level 4b/5b
cluster smoke tests, and keep PyTorchJob tests until TrainJob smoke is green.

---

## Platform Neutrality — Wheel Boundary

The `pragma_encoder` wheel is fully platform-neutral. No platform resource
(YAML manifests, cluster primitives, Kubernetes API references, credential names)
belongs inside the wheel.

**Must NOT go into `src/pragma_encoder/` (the wheel):**

- OpenShift AI Connections or Secret names
- HardwareProfile definitions
- Data Science Project definitions
- Service accounts, namespace assumptions
- PyTorchJob / TrainJob / KFP YAML
- KFP runtime binding, DSPA endpoint references
- Cluster-specific configuration
- Secret references beyond reading standard env vars

**Enforced by:** `tests/test_platform_neutral_wheel.py` (static, no cluster).

---

## Fixture and Manifest Locations

| Purpose | Location |
|---|---|
| Declared platform state (ArgoCD-synced) | `openshift/gitops/` |
| ArgoCD Application definition | `openshift/argocd/` |
| Production training manifests (PyTorchJob) | `openshift/training/` |
| Production serving manifests | `openshift/serving/` |
| Credential templates (never populated) | `openshift/secrets/` |
| Test fixtures and primitive examples | `tests/openshift/fixtures/` |
| Platform contract tests (static) | `tests/test_openshift_ai_primitive_contract.py` |
| Platform neutrality tests (static) | `tests/test_platform_neutral_wheel.py` |
| OpenShift runtime tests (cluster required) | `tests/openshift/test_0*.py` |

---

## Surface Map

All surfaces operate over the same OpenShift AI primitives:

| Surface | Role |
|---|---|
| RHOAI dashboard (GUI) | User-facing: create workbenches, connections, pipeline runs, serving endpoints |
| Workbench / JupyterLab | Authoring: write pipeline code, inspect models, launch training via `train_pragma()` |
| KFP SDK | Programmatic pipeline compile and submission |
| ArgoCD / OpenShift GitOps | Promotion: keeps declared platform state in sync with Git |
| `oc` / `kubectl` | Direct cluster operations: apply manifests, inspect resources, debug |
| CI/CD | Automation: build images, run tests, promote artifacts |
| Tests | Validation: static contract tests + opt-in cluster smoke tests |

The platform contract is the same OpenShift AI primitives regardless of which
surface is used.

---

> Reference: `docs/openshift-ai-3.3-alignment.md` for RHOAI 3.3/3.4 primitive
> detail, responsibility boundaries, and env var contract.
> Note: that document was written for RHOAI 3.3 and carries a historical banner;
> the primitives it describes are compatible with the currently installed RHOAI 3.4.
>
> Reference: `docs/openshift-gitops-argocd.md` for ArgoCD ownership table and
> pipeline authoring lifecycle.
