# RHOAI 3.4 TrainJob / Kubeflow Trainer v2 — Checkpoint Evaluation

> **Status:** Evaluation document — not a decision record.
> The current production path is PyTorchJob + application-managed S3 checkpointing (Level 5).
> This document records research findings and defines the acceptance criteria for adopting
> the platform-native TrainJob checkpoint path.
>
> **Sources consulted (2026-05-24):**
> - Red Hat OpenShift AI 3.4 documentation (docs.redhat.com — JS-rendered, not directly crawlable)
> - Red Hat blog: "Resilient model training on Red Hat OpenShift AI with Kubeflow Trainer"
> - Kubeflow upstream migration guide: kubeflow.org/docs/components/trainer/operator-guides/migration/
> - Kubeflow trainer GitHub releases and CHANGELOG
> - Web search aggregation of RHOAI 3.x supported configurations
>
> **Uncertainty:** Several facts below are inferred from web search aggregation because
> the Red Hat docs pages render via JavaScript and content could not be directly extracted.
> All facts marked [VERIFY] must be confirmed against a live cluster or official Red Hat support.

---

## 1. Status of TrainJob / Kubeflow Trainer v2 in RHOAI 3.4

| Fact | Finding | Confidence | Needs verification? |
|---|---|---|---|
| TrainJob API | `trainer.kubeflow.org/v1alpha1` | High | No |
| TrainJob status in RHOAI 3.4 | **Tech Preview** (confirmed in existing repo docs and initial search) | Medium — conflicting signals in search results | [VERIFY] against RHOAI 3.4 release notes |
| RHOAI 3.4 ships ClusterTrainingRuntimes for Trainer v2 | Yes | High | No |
| PyTorchJob (`kubeflow.org/v1`) upstream status | **Deprecated** — upstream source code removed from kubeflow/trainer repo | High | No |
| PyTorchJob status in RHOAI 3.4 | Unclear — may be removed or still installed for backwards compat | Low | **[VERIFY on cluster]** |
| Kubeflow Trainer v2 resilient checkpointing introduced | RHOAI 3.2 | High | No |
| Checkpoint types supported | JIT (on SIGTERM/preemption) + periodic/interval | High | No |
| Primary checkpoint backend | PVC (confirmed) | High | No |
| S3-compatible checkpoint backend | Not conclusively confirmed from official docs | Low | **[VERIFY]** |
| Checkpoint SDK HF-trainer specific? | Currently oriented toward HuggingFace transformers/TRL trainers | High | No |
| Custom PyTorch loop supported? | Not confirmed — requires custom integration | Low | **[VERIFY]** |
| Credentials via RHOAI Connection | Yes (same `AWS_*` env var mechanism) | High | No |

### Critical finding: PyTorchJob deprecation

The upstream Kubeflow Training Operator v1 source code has been **removed** from
the `kubeflow/trainer` repository. Upstream Kubeflow docs now carry deprecation
warnings on all v1 documentation pages and redirect to v2.

From the Kubeflow website commit (Feb 2025):
> "This page is about Kubeflow Training Operator V1, for the latest information
> check the Kubeflow Trainer V2 documentation."

**Implication:** Our current production path (`kubeflow.org/v1` PyTorchJob) is on a
deprecation trajectory. Whether RHOAI 3.4 still ships the v1 CRDs for backwards
compatibility must be verified on the cluster before acting. If v1 is not installed,
the Level 5 cluster tests will fail.

**Do not assume either conclusion. Verify on the cluster.**

---

## 1b. Cluster Preflight — Verify Before Running Level 5 Tests

Before running Level 5 (`test_05_s3_checkpoint_resume.py`) or any cluster test
that submits a `PyTorchJob` or `TrainJob`, verify CRD availability on the target cluster:

```bash
# Verify PyTorchJob (kubeflow.org/v1) is still installed
oc api-resources | grep -i pytorchjob

# Verify TrainJob (trainer.kubeflow.org/v1alpha1) is available
oc api-resources | grep -i trainjob

# Verify TrainingRuntime / ClusterTrainingRuntime CRDs
oc api-resources | grep -i trainingruntime

# List available ClusterTrainingRuntimes (admin-configured blueprints)
oc get clustertrainingruntime

# Full CRD list filtered to Kubeflow/Trainer
oc get crd | grep -i trainer
oc get crd | grep -i pytorchjob
```

**Note:** CRD presence confirms the API is installed, not that it is GA or
production-supported. TrainJob CRDs may be present as Technology Preview.
PyTorchJob CRDs may be present for backwards compatibility even after upstream
source removal. Always check the RHOAI release notes for the target version.

---

## 2. How Kubeflow Trainer v2 Checkpointing Works

Based on the Red Hat blog (Resilient model training on RHOAI with Kubeflow Trainer):

### Checkpoint trigger modes

| Mode | How it fires | Use case |
|---|---|---|
| **Periodic** | On a configured time/step interval | Protects against unexpected failures (node crash, power loss) |
| **JIT (Just-In-Time)** | On SIGTERM / pod preemption / planned termination | Protects against planned events (preemption, maintenance, scale-down) |

Both modes can be used together.

### What the SDK does

- The Kubeflow Training SDK injects checkpoint configuration into training functions at runtime
- Detects existing checkpoints automatically to resume training
- Manages graceful shutdown during preemption and termination events
- Periodic checkpoint writes are triggered by the SDK on the configured interval

### What training code must provide

For the SDK-native path (HuggingFace trainers):
- Use a supported HuggingFace trainer class: `Trainer`, `SFTTrainer`, `Seq2SeqTrainer`, `DPOTrainer`, `PPOTrainer`, `RewardTrainer`
- The SDK hooks into HuggingFace's callback/checkpoint mechanism automatically
- **No manual `torch.save` / `torch.load` calls required** for the SDK path

For custom PyTorch training loops (like PRAGMA):
- The SDK does **not** natively hook into custom loops
- Custom integration is required — the checkpoint dir/S3 path must be read from
  the environment or TrainingRuntime configuration and wired into the custom loop
- This is a **gap** for PRAGMA: the application must implement save/load calls
  that align with the platform's checkpoint lifecycle

### Checkpoint backends

| Backend | Status | Notes |
|---|---|---|
| PVC (PersistentVolumeClaim) | Confirmed | Standard path; checkpoint dir mounted via PVC |
| S3-compatible | [VERIFY] | Not confirmed from official docs; may require custom training code |

### Configuration objects

| Object | Scope | Purpose |
|---|---|---|
| `ClusterTrainingRuntime` | Cluster-wide | Admin-defined blueprint: distributed topology, container spec, checkpoint runtime config |
| `TrainingRuntime` | Namespace-scoped | Per-project blueprint (overrides/extends ClusterTrainingRuntime) |
| `TrainJob` | Namespace-scoped | User-submitted job: references a runtime, specifies trainer image, args, node count |

The checkpoint backend and interval are configured in the `ClusterTrainingRuntime` /
`TrainingRuntime` spec, not in the `TrainJob` itself.

### Credentials for S3

If S3 is a supported backend (to be verified), credentials would be supplied via
the same RHOAI Connection mechanism:
- An OpenShift AI Connection (annotated Secret with `opendatahub.io/connection-type: s3`)
  injected via `envFrom.secretRef` — the same pattern used today with PyTorchJob.
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT`, `AWS_S3_BUCKET`
  would be injected into training pods as env vars.

---

## 3. Responsibility Comparison Table

| Concern | RHOAI 3.3 PyTorchJob path (current) | RHOAI 3.4 TrainJob path (evaluation) | pragma_encoder responsibility | Platform responsibility | Decision |
|---|---|---|---|---|---|
| Distributed worker orchestration | Kubeflow Training Operator v1 — Master + Worker pods | Kubeflow Trainer v2 — unified TrainJob + ClusterTrainingRuntime | None — platform concern | Platform | PyTorchJob deprecated upstream; TrainJob is forward path |
| torchrun / distributed env setup | RHOAI injects MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK | RHOAI injects same vars via ClusterTrainingRuntime launcher | None | Platform | Compatible — training code unchanged |
| Checkpoint save timing (periodic) | Application code: `upload_checkpoint_if_rank0()` on each epoch end | Platform SDK: automatic interval-based save (HF trainers) or custom loop | Application owns for custom loop; Platform owns for HF trainers | Platform (HF) / App (custom) | PRAGMA uses custom loop → app still responsible |
| JIT checkpoint on SIGTERM | Not implemented — application does not handle SIGTERM gracefully | Platform SDK: intercepts SIGTERM, triggers checkpoint before pod exits | Not implemented in pragma_encoder | Platform (HF) / Gap (custom) | Gap for PRAGMA — custom SIGTERM handler not written |
| Periodic checkpointing | Epoch-end upload via `upload_checkpoint_if_rank0` (application) | SDK-managed on time/step interval (HF) or custom wiring (custom loop) | Application controls timing | Platform (HF) / App (custom) | PRAGMA epoch-end upload remains valid; interval-based upgrade deferred |
| Latest checkpoint discovery | `select_latest_checkpoint_key()` — rank 0 queries S3 by `LastModified` | Platform detects automatically (PVC) or via configured path | Application owns for S3; platform owns for PVC | Platform (PVC) / App (S3) | If PVC: platform handles; if S3: app still handles |
| S3 upload / download | `upload_checkpoint_if_rank0` / `download_checkpoint_for_rank` in `checkpoints.py` | Unconfirmed — may require custom wiring even with TrainJob | Application (S3 path) | Unconfirmed | S3 upload/download remains application responsibility pending verification |
| PVC backend | Not used — no PVC for training data (ADR 003) | Platform-native — checkpoint dir mounted as PVC | None if PVC backend chosen | Platform | PVC backend is an architectural change from ADR 003 (no data PVCs); new ADR required |
| Object-storage credentials | `pragma-workbench-env` Secret via `envFrom.secretRef` | Same RHOAI Connection mechanism (`envFrom.secretRef`) | Reads `AWS_*` env vars | Platform injects | Compatible — no change to `pragma_encoder` |
| Model/optimizer/scheduler state | `torch.save({'model': ..., 'optimizer': ..., 'epoch': ...})` | Application still writes `torch.save` for custom loop | Application owns | Application | No change for custom loop |
| Resume after pod interruption | All ranks download from S3 via `resolve_resume_checkpoint()` (TD-006 fix) | Platform resumes from PVC automatically (HF); custom loop reads checkpoint path from env | App (S3) | Platform (HF/PVC) | S3 path: app unchanged; PVC path: platform handles |
| Resume after intentional second run | `--resume` flag → `resolve_resume_checkpoint()` | Same `--resume` flag pattern; checkpoint discovery differs by backend | Application owns | Backend-dependent | Evaluation needed |
| Rank-safe coordination | `dist.broadcast_object_list` + `dist.barrier()` in `resolve_resume_checkpoint` | Platform handles pod scheduling; rank-safe download is application concern | Application (S3 path) | Platform (PVC path) | Remains application responsibility for S3 |
| Training image selection | `PRAGMA_TRAINING_IMAGE` env var → PyTorchJob container spec | `TrainJob.spec.trainer.image` | Application specifies | Application specifies via TrainJob | Compatible |
| Resource selection / HardwareProfile | Resource requests in PyTorchJob spec; HardwareProfile via RHOAI GUI | TrainJob `resourcesPerNode`; HardwareProfile referenced via ClusterTrainingRuntime | None — platform concern | Platform | TrainJob integrates more cleanly with HardwareProfile |

---

## 4. Gap Analysis — PRAGMA Custom Training Loop

PRAGMA uses `pragma_encoder.training.train` — a **custom PyTorch training loop**,
not a HuggingFace Trainer subclass.

| Gap | Description | Effort to close |
|---|---|---|
| JIT checkpoint on SIGTERM | Training code does not install a SIGTERM handler. On pod preemption/termination, no checkpoint is written. | Medium — add `signal.signal(SIGTERM, ...)` in `train.py` |
| SDK checkpoint injection | Kubeflow Trainer SDK injects checkpoint config for HF trainers. PRAGMA's custom loop must read checkpoint dir/S3 from env or `TrainingRuntime` spec manually. | Medium — read checkpoint config from env at startup |
| PVC checkpoint backend | Current design explicitly avoids data PVCs (ADR 003). Using PVC checkpointing requires a new decision. | Architectural — needs new ADR |
| S3 checkpoint via TrainJob | Whether TrainJob + `ClusterTrainingRuntime` can configure S3 as the checkpoint backend is unconfirmed. | [VERIFY] against docs or cluster experiment |
| Interval-based checkpointing | Current path saves on epoch end. SDK-managed interval checkpointing requires custom integration. | Low-Medium — add step-count hook in `train.py` |

**Current state:** Application-managed S3 checkpointing (Level 5) is valid and
proven for the PyTorchJob path. It remains valid for TrainJob with custom loop
if S3 is the chosen backend and TrainJob replaces PyTorchJob.

The platform-managed checkpointing advantage (JIT on SIGTERM, automatic resume)
is **not accessible without either (a) switching to HF trainers or (b) writing
custom SIGTERM handlers and checkpoint-dir wiring in `train.py`.**

---

## 5. Level 5 Reclassification

**Current classification:** Level 5 is the application-managed S3 checkpoint/resume
smoke test for the RHOAI 3.3 PyTorchJob path (Tier 3, opt-in).

**Updated classification:**

- Level 5 remains the **current, proven** checkpoint/resume path for:
  - RHOAI 3.3 PyTorchJob (if still installed in cluster)
  - Any future TrainJob + application-managed S3 path
  - Local training (no cluster)

- Level 5 is **NOT** the forward platform path for RHOAI 3.4 if TrainJob reaches
  GA and the platform-managed PVC checkpoint path is adopted.

- Before expanding application-managed S3 checkpoint logic, evaluate whether
  TrainJob + platform-managed checkpointing satisfies the same requirements.

- If TrainJob satisfies requirements, `checkpoints.py` and the S3 resume logic
  become:
  - **Fallback path** for PyTorchJob/local training
  - **Unnecessary on-platform** for TrainJob with PVC checkpointing

---

## 6. PyTorchJob Deprecation — Urgency Assessment

| Signal | Source | Weight |
|---|---|---|
| Upstream Training Operator v1 source code removed from kubeflow/trainer | GitHub — Remove Training Operator V1 Source Code (#2389) | High |
| Kubeflow docs redirect all v1 pages to v2 with deprecation notice | kubeflow.org (Feb 2025) | High |
| RHOAI 3.4 ships ClusterTrainingRuntimes for Trainer v2 | Web search aggregation of RHOAI docs | Medium |
| PyTorchJob still listed as "GA (RHOAI 3.4)" in our repo | Existing repo docs (openshift-ai-primitives.md) | Medium — may be stale |
| Web search says "removed from RHOAI (effective 2025, confirmed in RHOAI 3.4+)" | AI-aggregated search summary | Low — not a direct quote |

**Assessment:** The upstream deprecation is definitive. Whether RHOAI 3.4 still
ships v1 CRDs for backwards compatibility is **unknown without cluster verification**.

**Required action:** Before running Level 5 cluster tests on a RHOAI 3.4 cluster,
verify that `kubeflow.org/v1` PyTorchJob is still available:
```bash
oc api-resources | grep kubeflow
oc api-resources | grep trainer
```

If `kubeflow.org/v1` is not present, Level 5 cluster tests cannot run until
migrated to TrainJob.

---

## 7. Evaluation Plan (see TD-012)

See `docs/tech-debt.md §TD-012` for the full acceptance criteria and step-by-step
evaluation plan.

Short summary:

1. Verify PyTorchJob (`kubeflow.org/v1`) availability on RHOAI 3.4 cluster.
2. Verify TrainJob (`trainer.kubeflow.org/v1alpha1`) availability and GA/TP status.
3. Submit a minimal TrainJob using the wheel-based training image.
4. Test PVC checkpoint backend end-to-end.
5. Investigate S3 checkpoint backend support.
6. Evaluate JIT checkpoint on SIGTERM with custom loop.
7. Compare with Level 5 PyTorchJob resume behaviour.
8. Write new ADR superseding ADR 005 based on findings.

---

## 8. Decision Matrix (Pending Evaluation)

| Scenario | Decision |
|---|---|
| PyTorchJob still available in RHOAI 3.4 | Keep Level 5 as-is; begin TrainJob evaluation in parallel |
| PyTorchJob removed from RHOAI 3.4 | Urgent: migrate Level 5 smoke to TrainJob; keep `checkpoints.py` for local path |
| TrainJob PVC checkpoint satisfies requirements | Add PVC checkpoint path to `train.py`; new ADR; Level 5b test |
| TrainJob S3 checkpoint confirmed | Evaluate replacing `checkpoints.py` S3 path with platform-managed version |
| TrainJob S3 checkpoint not supported | Keep `checkpoints.py` S3 path; TrainJob uses `envFrom` for credentials same as today |
| HF trainer adoption decision | Out of scope for PRAGMA v1 — custom loop architecture is paper-specified |

---

*Sources:*
- [Resilient model training on RHOAI with Kubeflow Trainer](https://www.redhat.com/en/blog/resilient-model-training-red-hat-openshift-ai-kubeflow-trainer)
- [RHOAI 3.4 documentation hub](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4)
- [Migrating to Kubeflow Trainer v2](https://www.kubeflow.org/docs/components/trainer/operator-guides/migration/)
- [Kubeflow Trainer GitHub releases](https://github.com/kubeflow/trainer/releases)
- [RHOAI supported configurations 3.x](https://access.redhat.com/articles/rhoai-supported-configs-3.x)
