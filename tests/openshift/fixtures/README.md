# tests/openshift/fixtures/

This directory contains **OpenShift AI 3.3-aligned fixture examples and test fixtures**.

These files are:
- Example YAML for RHOAI platform constructs
- Reference fixtures documenting the expected platform configuration
- Used by OpenShift integration tests as static validation targets
- NOT part of the `pragma_encoder` wheel

---

## Contents

| File | RHOAI Primitive | Description |
|---|---|---|
| `object-storage-connection.yaml.template` | Connection | RHOAI object-storage Connection that injects S3 credentials into pods |
| `hardware-profile-cpu-smoke.yaml` | Hardware Profile | CPU-only shape for smoke/test workloads |
| `hardware-profile-gpu-pragma-s.yaml` | Hardware Profile | GPU shape for PRAGMA-S production training |
| `trainjob-example.yaml` | TrainJob (Tech Preview) | Example of the future Kubeflow Trainer v2 API — not for production use |

---

## What these are NOT

These files are **not** deployed by the `pragma_encoder` package.
They are **not** cluster configuration managed by ArgoCD.
They are **reference fixtures** — the platform configuration that must exist
on the cluster for training and testing to work.

Actual deployed platform config lives in `openshift/gitops/`.

---

## Boundary

OpenShift AI / OpenShift owns:
- Data Science Projects
- Workbenches
- Connections
- Hardware Profiles
- Data Science Pipelines
- Service accounts
- Namespace/RBAC policy
- GPU scheduling
- Image selection at runtime
- Model serving

`pragma_encoder` owns:
- Model code
- Training code
- Checkpoint save/load and S3 logic
- CLI/module entrypoints
- Pure Python behaviour

Do not move these fixture files into the `pragma_encoder` package.
Do not make the wheel create, read, or reference these RHOAI constructs.

---

## Related documentation

- `docs/openshift-ai-3.3-alignment.md` — full primitive map and responsibility table
- `openshift/secrets/workbench-secret.template.yaml` — test Secret representing a Connection
- `tests/openshift/README.md` — OpenShift test suite overview
