# PRAGMA Encoder — Validated Architecture Summary

> **Status:** Validated milestone — 2026-05-25
> **Commit tested:** `fca41d7` on `main`
> **Platform:** RHOAI 3.4.0 on `api.ocpai-demo.sandbox3438.opentlc.com`

---

## Summary

A platform-neutral Python training package can be built into a wheel-based
image, run through OpenShift AI/OpenShift primitives, consume
Connection-backed object storage, execute through PyTorchJob, checkpoint and
resume through S3, and use GPU runtime — without leaking platform concerns
into the wheel.

---

## What was validated

| Area | Result | Evidence |
|---|---|---|
| Wheel build | Passed | `pragma_encoder-0.1.0-py3-none-any.whl` built cleanly from `python -m build` |
| Wheel excludes workbench helper | Confirmed | `workbench` absent from wheel zip; `ModuleNotFoundError` on import; enforced by `test_platform_neutral_wheel.py` |
| Installed training entrypoint | Confirmed | `pragma-encoder-train` and `python -m pragma_encoder.training.train` registered by wheel; verified at image build time |
| Training image installs wheel | Passed | `oc start-build pragma-encoder-training-3` — all 9 Dockerfile steps passed; digest `sha256:cd44ed5d…` |
| OpenShift image contract | 8/8 passed | `test_02b_image_contract_runtime.py` — including live pod-exec of `import pragma_encoder` and `checkpoints` module |
| PyTorchJob runtime | Passed (42 s) | `test_04_pytorchjob_smoke.py` — job submitted, ran, completed on `kubeflow.org/v1` CRD |
| Level 5 S3 checkpoint/resume | Passed (69 s) | `test_05_s3_checkpoint_resume.py` — Run 1 uploaded checkpoint; Run 2 discovered latest key and resumed |
| GPU runtime | Passed (18 s) | `test_06_gpu_training_smoke.py` — single-node GPU PyTorchJob; CUDA visible; compute capability 8.9 |
| Static preflight | 225 passed, 8 skipped | ruff clean; `test_packaging`, `test_platform_neutral_wheel`, `test_openshift_ai_primitive_contract`, `test_openshift_ai_fixtures`, `test_pipeline_components`, `test_smoke_pipeline` |
| OpenShift AI primitive mapping | Documented and tested | `docs/openshift-ai-3.3-alignment.md`; `test_openshift_ai_primitive_contract.py` (72 tests) |
| ArgoCD declared-state boundary | Documented and tested | `docs/openshift-gitops-argocd.md`; ArgoCD owns `openshift/gitops/`; does not own PyTorchJob runs, pipeline runs, or checkpoints |
| Platform-neutral wheel boundary | Enforced mechanically | `test_platform_neutral_wheel.py` — no kfp/K8s imports; no platform YAML; `pragma_encoder.workbench` not importable from wheel |

---

## Architecture shape

```
src/pragma_encoder
    -> wheel (pragma_encoder-0.1.0-py3-none-any.whl)
    -> training image (pip install --no-deps wheel in Dockerfile.training)
    -> ArgoCD-declared OpenShift AI/OpenShift primitives (openshift/gitops/)
    -> OpenShift AI / OpenShift runtime (RHOAI 3.4.0)
    -> PyTorchJob / GPU / S3 checkpoint-resume
```

```
tools/openshift_ai/workbench/
    -> helper tooling only (train_pragma(), pipeline DSL, DSPA submit)
    -> importable via PYTHONPATH=. from repo root
    -> not packaged into the wheel
    -> not required for training image or PyTorchJob execution
```

---

## Separation of concerns

| Layer | Owns |
|---|---|
| `pragma_encoder` wheel | Model/training/checkpoint/storage-adapter behaviour; `pragma-encoder-train` entrypoint |
| Training image | Executable runtime — wheel installed into site-packages; PyTorch/CUDA stack from workbench base |
| OpenShift AI | User/platform primitives — Connection, HardwareProfile, DSPA, Workbench, PyTorchJob scheduling |
| ArgoCD | Declared platform state — namespace, RBAC, Sealed Secrets, DSPA, Workbench, BuildConfig/ImageStream |
| Runtime | Dynamic jobs and artifacts — PyTorchJob runs, pipeline runs, checkpoints, logs (not GitOps-managed) |
| `tools/openshift_ai/workbench/` | Optional helper UX for Workbench authoring — not in wheel, not required on cluster |

---

## Current proven path

```
Current proven project path:
  PyTorchJob (kubeflow.org/v1)
    + wheel-based training image
    + OpenShift AI Connection-backed object storage (AWS_* env vars)
    + application-managed S3 checkpoint/resume (pragma_encoder.training.checkpoints)
```

The `kubeflow.org/v1` PyTorchJob CRD is confirmed present on the target RHOAI 3.4
cluster. Verify CRD availability on any new cluster before running Level 4/5 tests:

```bash
oc api-resources | grep -i pytorchjob
```

---

## Future / evaluation path

```
Future/evaluation path:
  RHOAI 3.4 TrainJob / Kubeflow Trainer v2 checkpointing
    — evaluate when target cluster has trainer.kubeflow.org/v1alpha1 CRDs
      and support posture is confirmed (see docs/tech-debt.md §TD-012)
```

- TrainJob is **not replacing PyTorchJob** in this deployment. The CRD was
  absent from the target cluster at time of validation.
- Level 5 S3 checkpoint/resume remains valid for the current PyTorchJob path.
- The full evaluation plan is in `docs/rhoai-3.4-trainjob-checkpointing.md`.

---

## KFP/OpenShift AI pipeline smoke

The KFP pipeline smoke (`test_03_pipeline_smoke_run.py`) was **cleanly skipped**
during the validation pass because `kfp` is a workbench-only dependency — it is
not installed in the local development venv. The architectural conclusion is not
blocked by this:

- The pipeline compile tests (static, Tier 2) passed.
- The wheel/image/PyTorchJob/S3/GPU path is fully validated.
- Running the KFP smoke requires `kfp` to be installed locally or the test to
  run from inside the Workbench pod. This is optional demonstration work.

---

## Known optional follow-ups

| Item | Priority | Notes |
|---|---|---|
| KFP/OpenShift AI pipeline runtime proof | Optional | Run `test_03_pipeline_smoke_run.py` from Workbench pod or with `kfp` installed locally |
| RHOAI 3.4 TrainJob checkpointing evaluation | Deferred | Requires cluster with `trainer.kubeflow.org/v1alpha1` — see TD-012 |
| Workbench-to-wheel contributor guide | Optional | Document the notebook → PR → wheel promotion workflow for new contributors |
| Customer demo script | Optional | End-to-end demo: fit tokenizer, train, resume, inspect checkpoint |
| Release/tagging workflow | Optional | Tag this milestone; consider versioning `pragma_encoder` beyond `0.1.0` |
| Runtime preflight script | Optional | `oc api-resources` + Connection check before Level 4/5 tests |

---

## Suggested milestone tag

```
validated-rhoai-pytorchjob-s3-gpu-runtime
```

To apply when ready:

```bash
git tag validated-rhoai-pytorchjob-s3-gpu-runtime fca41d7
git push origin validated-rhoai-pytorchjob-s3-gpu-runtime
```

---

*Reference: `docs/architecture.md` for the full layer model and model architecture.*
*Reference: `docs/openshift-ai-3.3-alignment.md` for the RHOAI 3.3/3.4 primitive map.*
*Reference: `docs/testing-strategy.md` for the four-tier test model.*
*Reference: `docs/tech-debt.md §TD-012` for the TrainJob evaluation plan.*
