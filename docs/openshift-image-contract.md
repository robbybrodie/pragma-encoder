# PRAGMA OpenShift Image Contract

This document describes the two-image model used by the PRAGMA training stack
on OpenShift AI and the responsibilities of each image.

---

## Two-image model

```
┌─────────────────────────────────────────┐
│  PRAGMA Workbench Image                 │
│  (pragma-encoder-workbench)             │
│                                         │
│  Purpose: author, compile, submit       │
│  pipelines from the JupyterLab UI.      │
│                                         │
│  Built from:                            │
│    openshift/notebook-image/Dockerfile  │
│  Base: RHOAI S2I generic data-science   │
│                                         │
│  Includes:                              │
│    Python runtime + PyTorch             │
│    pandas, numpy, scikit-learn          │
│    kfp >= 2                             │
│    kfp-kubernetes >= 1.2               │
│    tokenizers, transformers, peft       │
│    JupyterLab                           │
│                                         │
│  Does NOT include:                      │
│    pragma_encoder wheel                 │  ← installed in training image only
└─────────────────────────────────────────┘
              ↑ FROM (base)
┌─────────────────────────────────────────┐
│  PRAGMA Training / Component Image      │
│  (pragma-encoder-training)              │
│                                         │
│  Purpose: execute KFP component pods,   │
│  OpenShift Jobs, and PyTorchJob         │
│  containers.                            │
│                                         │
│  Built from:                            │
│    openshift/training/Dockerfile.training│
│  Base: pragma-encoder-workbench         │
│                                         │
│  Adds (installed at build time):        │
│    pragma_encoder wheel (pip install)   │
│    scripts/ → train_pragma.py wrapper   │
│                                         │
│  Inherits from workbench:               │
│    All Python deps (kfp, kfp-kubernetes,│
│    PyTorch, pandas, tokenizers, …)      │
│                                         │
│  Must NOT include:                      │
│    Runtime git clone                    │  ← wheel is installed at build time
└─────────────────────────────────────────┘
```

---

## Why two images?

### Workbench image: authoring environment, no pragma_encoder wheel

The workbench image provides the Python runtime environment for authoring
pipelines in JupyterLab. The `pragma_encoder` wheel is **not** installed in
the workbench image. Workbench users access the library via `PYTHONPATH=.`
(source tree on `sys.path`) during development.

### Training image: wheel installed, no runtime clone

KFP component pods run in their own image — they do **not** inherit the
workbench pod's source tree or filesystem. The training image installs the
`pragma_encoder` wheel via `pip install --no-deps dist/pragma_encoder-*.whl`
at image build time, so component pods can `import pragma_encoder` without
any network access.

Training entrypoints available in the image:
- `pragma-encoder-train` (console script, installed by wheel)
- `python -m pragma_encoder.training.train` (module invocation)
- `scripts/train_pragma.py` (compatibility wrapper, copied at build time)

**Runtime git clone is not used** because:

1. It requires network egress from the component pod to GitHub.
2. It couples the component pod's execution to GitHub availability.
3. It means a git push can silently change running pipeline behaviour.
4. The training image build step provides a proper image digest audit trail.

---

## Image name environment variables

All three variables are read at compile time (module import) and can be
overridden without code changes.

| Variable | Purpose | Default |
|---|---|---|
| `PRAGMA_KFP_COMPONENT_IMAGE` | Override the image used by all KFP component pods. Highest priority. | — |
| `PRAGMA_TRAINING_IMAGE` | Training image URI. Used as component image when `PRAGMA_KFP_COMPONENT_IMAGE` is not set. | `image-registry.openshift-image-registry.svc:5000/pragma-encoder/pragma-encoder-training:latest` |
| `PRAGMA_IMAGE_PULL_SECRET_NAME` | Name of the OpenShift pull secret to attach to component pods (future: kfp-kubernetes secret injection). | — |

Resolution order for KFP component base image:

```
PRAGMA_KFP_COMPONENT_IMAGE  →  PRAGMA_TRAINING_IMAGE  →  default
```

Set before compiling the pipeline (before `import pipeline.components_pragma`):

```bash
export PRAGMA_TRAINING_IMAGE=image-registry.openshift-image-registry.svc:5000/pragma-encoder/pragma-encoder-training:latest
export PRAGMA_KFP_COMPONENT_IMAGE=...   # optional override
```

---

## kfp-kubernetes: workbench only

`kfp-kubernetes` is a KFP SDK extension for Kubernetes-native features:
secret injection (`use_secret_as_env`), PVC mounting, tolerations, etc.

**kfp-kubernetes belongs in the workbench/compile environment, not in the
component image's required dependencies.** The workbench uses kfp-kubernetes
at pipeline _compile_ time to annotate component pods. The component pods
themselves do not import kfp-kubernetes.

**Installation:**

```bash
# Workbench image — already included in openshift/notebook-image/requirements.txt
pip install kfp-kubernetes>=1.2

# kfp is also an optional dependency:
pip install kfp
```

**Guard behaviour:**

Functions that use kfp-kubernetes features call `_require_kfp_kubernetes()`
before attempting any import. If kfp-kubernetes is absent, a friendly
`ImportError` is raised with install instructions. This guard is **only**
invoked when secret injection is explicitly enabled — it is never called
unconditionally at module load time.

---

## Image build commands

### Workbench image

```bash
# Triggered by ArgoCD on push to main (openshift/gitops/notebook-image/buildconfig.yaml).
# Manual trigger:
oc start-build pragma-encoder-workbench -n pragma-encoder

# Verify:
oc get istag pragma-encoder-workbench:latest -n pragma-encoder
```

### Training / component image

```bash
# Run from repo root — Dockerfile.training installs the wheel from dist/:
oc start-build pragma-encoder-training --from-dir=. --follow -n pragma-encoder

# Verify:
oc get istag pragma-encoder-training:latest -n pragma-encoder
```

---

## What component pods import

KFP v2 serializes the component function body (Python source) into the
pipeline YAML at compile time. Each component pod receives:

1. A self-contained `ephemeral_component.py` (from the YAML).
2. The training image as the execution environment.

The component bodies import from `pragma_encoder.*` — the wheel installed in
the training image. The `pipeline/` directory is **not** in the training
image; it is only used in the workbench environment at compile time.

---

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
