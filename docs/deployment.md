# PRAGMA Encoder — OpenShift AI Deployment Guide

Deploys the PRAGMA encoder workbench and training infrastructure onto Red Hat
OpenShift AI (RHOAI) 3.3 via ArgoCD GitOps.

---

## Architecture

For the separation-of-concerns layer model and ownership table, see
`docs/architecture.md`. This document covers bootstrap steps, sync wave
order, and cluster-specific prerequisites.

```
Git (pragma-implementation branch)
  └── ArgoCD Application (openshift-gitops)
        └── openshift/gitops/   ← auto-synced, recurse: true
              ├── namespace.yaml          wave -1
              ├── rbac/                   wave  0
              │   ├── service-accounts.yaml      (training SA + anyuid SCC)
              │   ├── argocd-admin.yaml           (bootstrap: ArgoCD admin)
              │   ├── argocd-sealedsecret-access.yaml (bootstrap: SealedSecret RBAC)
              │   └── sa-imagepull-links.yaml     (default+builder imagePullSecrets)
              ├── secrets/                wave  0
              │   ├── registry-pull-secret.sealed.yaml   → pragma-registry
              │   └── workbench-runtime-secret.sealed.yaml → pragma-workbench-env
              ├── notebook-image/         wave  1–2
              │   ├── imagestream.yaml    wave  1
              │   └── buildconfig.yaml   wave  2  (auto-triggers image build)
              ├── pipeline/               wave  3
              │   └── dspa.yaml          (KFP 2.5, S3 external storage)
              └── workbench/             wave  4
                  └── notebook.yaml      (GPU workbench + init container git clone)
```

**Secrets** (Bitnami Sealed Secrets — encrypted in Git):

| Sealed file | Decrypts to | Contents |
|---|---|---|
| `registry-pull-secret.sealed.yaml` | `pragma-registry` | NGC dockercfg for nvcr.io |
| `workbench-runtime-secret.sealed.yaml` | `pragma-workbench-env` | AWS_* S3 credentials (OpenShift AI Connection) |

---

## Prerequisites

| Requirement | Check |
|---|---|
| `oc` CLI, cluster-admin | `oc whoami` → `kube:admin` |
| OpenShift GitOps operator | `oc get pods -n openshift-gitops` |
| Bitnami Sealed Secrets controller | `oc get pods -n kube-system -l name=sealed-secrets-controller` |
| GPU node Ready | `oc get nodes -l nvidia.com/gpu.present=true` |
| S3 bucket accessible | `nemo-tfm-pipelines-e17c4943` in `us-west-2` |

---

## One-time Bootstrap

Bootstrap must be run **once** before ArgoCD takes over. After that, all changes
are driven by Git pushes to `pragma-implementation`.

```bash
# From repo root, logged in as cluster-admin:
chmod +x openshift/scripts/bootstrap-project.sh
./openshift/scripts/bootstrap-project.sh
```

### What the script does

| Step | Command | Why |
|---|---|---|
| 1 | `oc apply -f openshift/gitops/namespace.yaml` | Creates the `pragma-encoder` project |
| 2 | `oc apply -f openshift/gitops/rbac/argocd-admin.yaml` | Gives ArgoCD admin rights in the namespace — **must exist before ArgoCD syncs** |
| 2 | `oc apply -f openshift/gitops/rbac/argocd-sealedsecret-access.yaml` | Lets ArgoCD manage SealedSecret resources |
| 3 | `oc apply -f openshift/gitops/secrets/*.sealed.yaml` | Decrypts secrets before ArgoCD's first sync |
| 4 | `oc apply -f openshift/argocd/application.yaml` | Registers the Application CR — ArgoCD auto-sync starts |

### Why this order matters

ArgoCD's application controller SA needs namespace-admin access **before** it
can apply the wave-0 RBAC resources (which are inside the namespace). If you
apply `application.yaml` first, the first sync will fail with a 403 on
`argocd-admin-pragma-encoder` RoleBinding — because ArgoCD doesn't yet have
permission to create it. The bootstrap script resolves this chicken-and-egg.

---

## Sync Wave Order

ArgoCD applies resources in wave order, waiting for each wave to be healthy
before proceeding:

| Wave | Resources | Depends on |
|---|---|---|
| -1 | `namespace.yaml` | — |
| 0 | RBAC, SealedSecrets | namespace |
| 1 | `imagestream.yaml` | namespace |
| 2 | `buildconfig.yaml` (triggers image build) | imagestream |
| 3 | `dspa.yaml` (KFP server) | secrets (wave 0) |
| 4 | `notebook.yaml` + PVC | image build (wave 2), secrets (wave 0) |

The image build (wave 2) takes 5–15 minutes. The Notebook pod (wave 4) won't
start until the `pragma-encoder-workbench:latest` ImageStreamTag exists. ArgoCD
marks the Notebook as `Progressing` and the Notebook controller retries
until the image is ready.

---

## Monitoring

```bash
# ArgoCD sync status
oc get applications.argoproj.io pragma-encoder -n openshift-gitops

# ArgoCD UI URL
oc get route openshift-gitops-server -n openshift-gitops -o jsonpath='{.spec.host}'

# Image build progress
oc get builds -n pragma-encoder
oc logs -f build/pragma-encoder-workbench-1 -n pragma-encoder

# All pods
oc get pods -n pragma-encoder

# Notebook pod (init container clones Git repo on first start)
oc logs -f -c git-clone \
  $(oc get pod -n pragma-encoder -l app=pragma-encoder-workbench -o name) \
  -n pragma-encoder

# Events (useful when pods are pending)
oc get events -n pragma-encoder --sort-by=.lastTimestamp | tail -20
```

---

## ArgoCD and Pipeline Authoring Lifecycle

ArgoCD owns the DSPA (wave 3 — the KFP pipeline server). It does **not** own
pipeline Python source, compiled IR YAML, pipeline definitions/versions, or
pipeline runs. Those belong to the Workbench authoring lifecycle.

Pipelines are authored as Python using KFP SDK v2 decorators — not hand-written
YAML. The Workbench is the primary authoring surface. The RHOAI GUI is the
runtime surface for definitions, versions, and runs.

**[docs/openshift-gitops-argocd.md](openshift-gitops-argocd.md)** — full
ownership table, pipeline authoring lifecycle, and GitOps promotion path.

---

## Storage model

Training data, checkpoints, and outputs are stored in S3. Training pods use
`emptyDir` scratch volumes — no dataset PVCs. See:

**[docs/openshift-storage-pattern.md](openshift-storage-pattern.md)**

That document is the canonical reference for all storage decisions (S3 paths,
credential management, failure semantics, RWO PVC prohibition, emptyDir usage).

---

## Running a Training Job

The PyTorchJob is **not** auto-synced by ArgoCD (it's a one-shot job, not
a long-running resource). Apply manually when ready.

For the full procedure (data upload, monitoring, checkpoint resilience) see
**[docs/training-guide.md](training-guide.md)**. Quick reference:

```bash
# 1. Upload training data to S3 (once, idempotent):
python scripts/upload_training_data.py \
  --csv-path  data/tabformer/card_transaction.v1.csv \
  --vocab-path data/tabformer/vocab.pkl

# 2. Verify the workbench image exists:
oc get istag pragma-encoder-workbench:latest -n pragma-encoder

# 3. Submit the PRAGMA-S training job (single GPU):
oc apply -f openshift/training/pytorchjob-pragma-s.yaml -n pragma-encoder

# 4. Monitor:
oc get pytorchjob pragma-s-pretrain -n pragma-encoder
oc logs -f -l job-name=pragma-s-pretrain,replica-type=master -n pragma-encoder
```

The job uses no dataset PVCs. The init container downloads data from S3 into
`emptyDir`. On restart, it re-downloads everything and resumes from the latest
S3 checkpoint.

---

## Teardown

```bash
# Stop ArgoCD reconciliation
oc delete application pragma-encoder -n openshift-gitops

# Remove all resources
oc delete project pragma-encoder
```

---

## Tech Debt

| ID | Issue | Resolution |
|---|---|---|
| TD-001 | S3 bucket `nemo-tfm-pipelines-e17c4943` shared with nemo-tfm | Create dedicated `pragma-encoder-pipelines-<suffix>` bucket; re-seal secrets |
| TD-002 | NGC API key shared with nemo-tfm | Generate dedicated NGC key; re-seal |

See `docs/tech-debt.md` for full details.
