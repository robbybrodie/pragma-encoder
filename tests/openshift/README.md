# PRAGMA OpenShift Integration Tests

## Purpose

This suite verifies the PRAGMA platform substrate on OpenShift and validates
cluster-facing behaviour that cannot be covered by local unit tests.

The local/unit test suite (`tests/`) has strong coverage of:
- model logic and architecture
- tokeniser and assembler
- workbench API and dry_run
- local training execution
- decorated pipeline authoring
- pipeline compilation

The gap is the transition from workbench authoring to **real OpenShift AI
pipeline execution and PyTorchJob training**. This suite fills that gap with
a safe, opt-in TDD loop for cluster-facing behaviour.

---

## Pipeline Runtime — OpenShift AI / KFP v2 (not Tekton)

This environment uses **OpenShift AI Data Science Pipelines** backed by
**KFP v2 via DataSciencePipelinesApplication (DSPA)**:

- `datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io` — DSPA CRD
- `pipelines.pipelines.kubeflow.org` — KFP v2 Pipeline CRD
- `pipelineversions.pipelines.kubeflow.org` — KFP v2 PipelineVersion CRD
- `ds-pipeline-*` pods provide the KFP v2 API server, workflow controller,
  and persistence agent

**Tekton PipelineRun/TaskRun (`pipelineruns.tekton.dev`, `taskruns.tekton.dev`)
are not required** unless the cluster explicitly uses that runtime.
The Level 1 substrate checks target DSPA/KFP v2, not Tekton.

## Architecture Boundary — batch/v1 Job vs. DSPA/KFP vs. PyTorchJob

Three distinct execution paths are tested at different maturity levels:

| Path | Level | Purpose | Status |
|------|-------|---------|--------|
| `batch/v1 Job` | 3b | **Diagnostic only.** Proves the training image is pullable and runs `--max-steps 1` to exit 0. One pod, no DDP, no S3, no KFP orchestration. | Implemented |
| DSPA/KFP v2 | 3 | **Product pipeline path.** Compiles a decorated pipeline, uploads to the DSPA API, creates a KFP Run, polls until Succeeded. This is the real execution path. | Implemented |
| PyTorchJob | 4 | **N-node distributed training.** Multi-node DDP via KFTO `kubeflow.org/v1 PyTorchJob`. Default smoke: nnodes=2 (minimal distributed case). Architecture is N-node capable. | Implemented |

The `batch/v1 Job` smoke (Level 3b) is **not** a substitute for the DSPA/KFP
pipeline smoke (Level 3). Level 3b passing means the image works. Level 3
passing means the full product pipeline path works.

---

## Architecture

```
Argo CD  ─── deploys ──►  platform substrate  (long-lived, Argo-managed)
                           └── namespace
                           └── ServiceAccount
                           └── RBAC
                           └── DSPA (DataSciencePipelinesApplication)
                           └── Secrets (S3, image pull)

tests/openshift/  ─── verify substrate ──►  read-only checks (Levels 0–1)
                  ─── create (future) ──►   short-lived labelled resources
                                            └── KFP v2 Run    (Level 3)
                                            └── batch/v1 Job  (Level 3b)
                                            └── PyTorchJob    (Level 4)
                  ─── cleanup ──────────►   only label-scoped resources
```

Durable data and training artifacts live in S3-compatible object storage.
Tests never write to S3 unless explicitly authorised.

`tests/openshift/fixtures/` — RHOAI 3.3-aligned fixture examples:
Connection templates, Hardware Profile examples, TrainJob reference (Tech Preview).
These are reference fixtures documenting expected platform config, not deployed artifacts.
See `docs/openshift-ai-3.3-alignment.md` for the full primitive map and responsibility split.

---

## Maturity Levels

| Level | Name | Status | File |
|-------|------|--------|------|
| 0 | oc access + namespace checks | Implemented | `test_00_oc_access.py` |
| 1 | Argo-managed substrate verification | Implemented | `test_01_cluster_prereqs.py` |
| 2 | Decorated pipeline compile | Implemented | `test_02_pipeline_compile.py` |
| 3 | OpenShift AI KFP v2 pipeline smoke | Implemented | `test_03_pipeline_smoke_run.py` |
| 3b | Training container Job smoke | Implemented | `test_03b_training_job_smoke.py` |
| 4 | PyTorchJob N-node smoke (default: nnodes=2) | Implemented | `test_04_pytorchjob_smoke.py` |
| 5 | S3-backed checkpoint/resume | Implemented | `test_05_s3_checkpoint_resume.py` |
| 6 | GPU training smoke (single-node opt-in) | Scaffold implemented | `test_06_gpu_training_smoke.py` |
| 7 | Production pipeline runtime — IBM TabFormer | Implemented | `test_07_tabformer_pipeline_runtime.py` |
| 8 | Single-node GPU loss curve smoke — IBM TabFormer | Implemented | `test_08_tabformer_loss_smoke.py` |

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `RUN_OPENSHIFT_TESTS=1` | Opt-in flag. All tests skip without this. |
| `PRAGMA_TEST_NAMESPACE=<namespace>` | Namespace where Argo CD has deployed the PRAGMA substrate. |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `PRAGMA_TEST_RUNTIME_NAMESPACE` | `PRAGMA_TEST_NAMESPACE` | Namespace for ephemeral test resources. Use when you want to isolate test-created resources from the Argo-managed namespace. |
| `PRAGMA_S3_SECRET_NAME` | (unset) | Name of the S3 credentials Secret. Test verifies existence only; data is never read. |
| `PRAGMA_TRAINING_SERVICE_ACCOUNT` | `pragma-encoder-training` | Name of the training ServiceAccount to verify. |
| `PRAGMA_IMAGE_PULL_SECRET_NAME` | (unset) | Name of the image pull Secret. Test verifies existence only; data is never read. |
| `RUN_OPENSHIFT_PIPELINE_SMOKE=1` | (unset) | Opt-in for Level 3 KFP v2 DSPA pipeline run smoke tests. |
| `RUN_OPENSHIFT_AI_PIPELINE_SMOKE=1` | (unset) | Opt-in for Level 7 production pipeline (TabFormer) runtime. Also requires `RUN_OPENSHIFT_TESTS=1` and `PRAGMA_TEST_NAMESPACE`. |
| `PRAGMA_SMOKE_MAX_STEPS` | `2` | Max training steps for Level 7 smoke (bounds wall-clock time). |
| `PRAGMA_SMOKE_LIMIT_ROWS` | `5` | Max TabFormer customers for Level 7 smoke (bounds data prep). |
| `PRAGMA_SMOKE_BATCH_SIZE` | `1` | Training batch size for Level 7 smoke (CPU-safe). |
| `PRAGMA_SMOKE_DEVICE` | `cpu` | Device for Level 7 smoke (`cpu`, `cuda`, `auto`). |
| `PRAGMA_SMOKE_MODEL_SIZE` | `S` | Model size for Level 7 smoke (`S`, `M`, `L`). |
| `PRAGMA_SMOKE_RUN_NAME` | `level7-tabformer-smoke` | Run name label applied to S3 artifacts. |
| `PRAGMA_SMOKE_DATASET_NAME` | `ibm-tabformer` | Dataset adapter key for Level 7 smoke. |
| `RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1` | (unset) | Opt-in for Level 3b batch/v1 Job training container smoke. |
| `RUN_PYTORCHJOB_TESTS=1` | (unset) | Opt-in for Level 4 PyTorchJob tests (static checks + CRD check). |
| `RUN_PYTORCHJOB_SMOKE=1` | (unset) | Opt-in for Level 4 runtime smoke (creates a short-lived PyTorchJob). Also requires `RUN_PYTORCHJOB_TESTS=1`. |
| `PRAGMA_PYTORCHJOB_NNODES` | `2` | Number of nodes for the Level 4 N-node smoke. Minimum 2 (nnodes=1 is non-distributed; use Level 3b). Values >2 require `PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1`. |
| `PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1` | (unset) | Permit N>2 node smoke. Guards against accidental cluster overload. Required when `PRAGMA_PYTORCHJOB_NNODES > 2`. |
| `RUN_OPENSHIFT_S3_RESUME_SMOKE=1` | (unset) | Opt-in for Level 5 S3-backed checkpoint/resume two-run smoke. Also requires `RUN_OPENSHIFT_TESTS=1`, `PRAGMA_TEST_NAMESPACE`, and `PRAGMA_TRAINING_IMAGE`. S3 credentials must be present in the `pragma-workbench-env` Secret. |
| `PRAGMA_S3_RESUME_PREFIX` | `pragma-encoder/test-checkpoints/<test_id>` | S3 key prefix for Level 5 test checkpoint storage. |
| `RUN_TEKTON_TESTS=1` | (unset) | Opt-in for Tekton PipelineRun/TaskRun CRD checks. Not required for OpenShift AI KFP v2. |
| `PRAGMA_TEST_TIMEOUT_SECONDS` | `300` | Timeout for cluster wait loops (minimum 30s). |
| `RUN_TABFORMER_LOSS_SMOKE=1` | (unset) | Opt-in for Level 8 static pre-flight checks (no cluster resources created). Also requires `RUN_OPENSHIFT_TESTS=1` and `PRAGMA_TEST_NAMESPACE`. |
| `RUN_TABFORMER_LOSS_SMOKE_RUN=1` | (unset) | Opt-in for the Level 8 runtime GPU PyTorchJob. Also requires `RUN_TABFORMER_LOSS_SMOKE=1`, `PRAGMA_TRAINING_IMAGE`, and `PRAGMA_TEST_NAMESPACE`. |
| `PRAGMA_TABFORMER_SAMPLE_FRACTION` | `0.10` | Fraction of IBM TabFormer customers to use (Level 8). Safe threshold ≤0.20; override with `PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1`. |
| `PRAGMA_TABFORMER_MAX_STEPS` | `100` | Max gradient steps for Level 8 training (safe threshold ≤500). |
| `PRAGMA_TABFORMER_BATCH_SIZE` | `4` | Batch size for Level 8 GPU training (L4-safe default). |
| `PRAGMA_LOSS_LOG_EVERY` | `5` | Loss logging frequency (steps) for Level 8. Must satisfy `max_steps // log_every >= min_loss_points`. |
| `PRAGMA_MIN_LOSS_POINTS` | `10` | Minimum number of finite loss records required to pass Level 8. Default: 100 // 5 = 20 ≥ 10 ✓. |
| `PRAGMA_ALLOW_LARGE_TABFORMER_SMOKE=1` | (unset) | Override Level 8 safety gates when `PRAGMA_TABFORMER_SAMPLE_FRACTION > 0.20` or `PRAGMA_TABFORMER_MAX_STEPS > 500`. |

---

## Safety Rules

1. Tests skip unless `RUN_OPENSHIFT_TESTS=1`.
2. Tests require `PRAGMA_TEST_NAMESPACE`.
3. Tests never create or delete namespaces.
4. Tests never delete or patch:
   - ServiceAccounts
   - Secrets
   - Argo CD Applications
   - Argo-managed ConfigMaps or templates
   - Namespace-level resources
5. Tests may verify that the above resources exist.
6. Tests create only short-lived resources with **both** labels:
   ```
   pragma.redhat.com/test-run=true
   pragma.redhat.com/test-id=<unique-test-id>
   ```
7. Cleanup deletes only resources carrying both labels.
8. Secret data is never printed, logged, or asserted on.
9. No broad destructive commands:
   - `oc delete all --all` — forbidden
   - `oc delete namespace` — forbidden
   - `oc delete secret` — forbidden
   - `oc delete serviceaccount` — forbidden
   - `oc delete sa` — forbidden
10. All cleanup commands are label-scoped.
11. `oc exec` is for diagnostics/log collection only — never the product path.

---

## How Cleanup Works

The `cleanup_labelled_resources` fixture runs after each test and deletes only:

```
pipelinerun, taskrun, pod, job, configmap, pytorchjob
```

…in `PRAGMA_TEST_RUNTIME_NAMESPACE`, scoped to the selector:

```
pragma.redhat.com/test-run=true,pragma.redhat.com/test-id=<test_id>
```

Cleanup failures are reported as warnings but do not hide the original test result.

PVCs are excluded from automatic cleanup by default. Secrets and ServiceAccounts
are never included.

---

## Running the Tests

### Default (skip all — no cluster needed)

```bash
pytest tests/openshift -q
```

Expected: all tests skipped with a clear message. No cluster access required.

### Read-only cluster checks (Levels 0–1)

```bash
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_00_oc_access.py tests/openshift/test_01_cluster_prereqs.py -q
```

Expected: DSPA/KFP v2 substrate checks pass. Tekton checks skipped unless
`RUN_TEKTON_TESTS=1`.

### Pipeline compile verification (Level 2, no cluster needed)

```bash
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_02_pipeline_compile.py -q
```

### Full read-only suite (Levels 0–2)

```bash
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_00_oc_access.py \
       tests/openshift/test_01_cluster_prereqs.py \
       tests/openshift/test_02_pipeline_compile.py -q
```

### With optional substrate checks

```bash
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_S3_SECRET_NAME=pragma-encoder-s3 \
PRAGMA_IMAGE_PULL_SECRET_NAME=pragma-encoder-pull \
pytest tests/openshift/test_01_cluster_prereqs.py -q
```

### Tekton CRD checks (only for clusters using Tekton as pipeline runtime)

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_TEKTON_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_01_cluster_prereqs.py -q
```

### OpenShift AI KFP v2 pipeline smoke (Level 3 — DSPA connectivity + xfail smoke)

Prereqs (5 local tests) run with just `RUN_OPENSHIFT_TESTS=1`. Connectivity and
smoke tests also require `RUN_OPENSHIFT_PIPELINE_SMOKE=1`. The final smoke test
xfails until the `pragma_smoke_training_pipeline` component is implemented.

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_OPENSHIFT_PIPELINE_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_03_pipeline_smoke_run.py -v
```

Expected: 5 prereqs PASS, 3 connectivity tests PASS, 1 smoke PASS (requires in-cluster access).

### Training container Job smoke (Level 3b — implemented)

Proves the training image runs correctly in-cluster with a `batch/v1 Job`.
This is diagnostic only — not the product pipeline path.

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<registry>/<repo>/pragma-encoder-training:latest \
pytest tests/openshift/test_03b_training_job_smoke.py -q
```

### PyTorchJob N-node smoke (Level 4 — static checks + runtime smoke)

Static checks (CRD, manifest, DNS naming — no cluster resources created):

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_PYTORCHJOB_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_04_pytorchjob_smoke.py -q
```

Expected: 9 passed (CRD check + 5 manifest checks + 3 naming checks), 1 skipped (runtime smoke).

Runtime smoke (creates a short-lived 2-node PyTorchJob):

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_PYTORCHJOB_TESTS=1 \
RUN_PYTORCHJOB_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<registry>/<repo>/pragma-encoder-training:latest \
pytest tests/openshift/test_04_pytorchjob_smoke.py -q
```

Expected: 10 passed (9 static + 1 runtime N-node smoke). Default nnodes=2.

Override N (N≥2, requires `PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1` for N>2):

```bash
PRAGMA_PYTORCHJOB_NNODES=3 PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1 \
RUN_OPENSHIFT_TESTS=1 RUN_PYTORCHJOB_TESTS=1 RUN_PYTORCHJOB_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<registry>/<repo>/pragma-encoder-training:latest \
pytest tests/openshift/test_04_pytorchjob_smoke.py::TestPyTorchJobSmoke -q
```

### S3-backed checkpoint/resume smoke (Level 5 — implemented)

Local prereqs only (no cluster, no S3 credentials needed):

```bash
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_05_s3_checkpoint_resume.py::TestS3ResumeLocalPrereqs -q
```

Expected: 3 passed.

Full two-run S3 smoke (requires S3 credentials in `pragma-workbench-env` Secret and a running cluster):

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_OPENSHIFT_S3_RESUME_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<registry>/<repo>/pragma-encoder-training:latest \
pytest tests/openshift/test_05_s3_checkpoint_resume.py -q
```

Expected: 4 passed (3 local prereqs + 1 two-run S3 checkpoint/resume runtime smoke). Runtime is typically around 70s on the reference test cluster.

### Level 8 — TabFormer GPU loss curve smoke (static checks + runtime)

Static pre-flight checks only (no cluster resources created):

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_TABFORMER_LOSS_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<registry>/<repo>/pragma-encoder-training:latest \
pytest tests/openshift/test_08_tabformer_loss_smoke.py::TestLossSmokeStaticPrereqs -v
```

Expected: 11 passed (bounds, coherence, naming, CRD, image env var, limit_rows — no GPU used).

Runtime smoke (creates a single-node GPU PyTorchJob, downloads IBM TabFormer CSV from S3):

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_TABFORMER_LOSS_SMOKE=1 \
RUN_TABFORMER_LOSS_SMOKE_RUN=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<registry>/<repo>/pragma-encoder-training:latest \
pytest tests/openshift/test_08_tabformer_loss_smoke.py -v
```

Expected: 11 static passed + 1 runtime passed. Runtime asserts: ≥10 finite loss records,
all finite (no NaN/inf), step values non-decreasing, ≥2 distinct steps.
Artifacts written to `test-artifacts/level8-tabformer-loss/<job_name>/`.

---

## Verifying Normal Test Suite Is Unaffected

The OpenShift suite must not break the default test run:

```bash
pytest tests/ -q
```

Expected: all OpenShift tests skip automatically (no `RUN_OPENSHIFT_TESTS`
in the environment). Only unit tests run.

---

## What Is Currently Implemented

### Level 0 — oc access
- `test_oc_binary_available` — oc is on PATH
- `test_oc_whoami` — session is authenticated
- `test_test_namespace_exists` — PRAGMA_TEST_NAMESPACE exists on cluster
- `test_can_list_pods_in_test_namespace` — basic RBAC confirmed
- `test_runtime_namespace_exists_if_different` — runtime namespace check (if set)

### Level 1 — Argo-managed substrate (DSPA/KFP v2)
- `test_dspa_crd_exists` — DataSciencePipelinesApplication CRD present
- `test_kfp_pipeline_crd_exists` — pipelines.pipelines.kubeflow.org CRD present
- `test_kfp_pipelineversion_crd_exists` — pipelineversions.pipelines.kubeflow.org CRD present
- `test_dspa_instance_exists_in_namespace` — at least one DSPA in PRAGMA_TEST_NAMESPACE
- `test_dspa_pods_running_in_namespace` — ds-pipeline-* pods Running
- `test_tekton_pipelinerun_crd_exists_if_enabled` — gated by `RUN_TEKTON_TESTS=1`
- `test_tekton_taskrun_crd_exists_if_enabled` — gated by `RUN_TEKTON_TESTS=1`
- `test_pytorchjob_crd_exists_if_enabled` — gated by `RUN_PYTORCHJOB_TESTS=1`
- `test_training_service_account_exists` — SA deployed by Argo CD
- `test_s3_secret_exists_if_configured` — existence check only, no data access
- `test_image_pull_secret_exists_if_configured` — existence check only

### Level 2 — Pipeline compile
- `test_decorated_pipeline_example_exists` — example file present
- `test_decorated_pipeline_compile_succeeds_if_kfp_installed` — compile to YAML
- `test_generated_pipeline_yaml_exists_after_compile` — YAML is non-empty
- `test_generated_pipeline_yaml_contains_expected_stages` — all 5 stages present
- `test_compile_does_not_require_oc_or_cluster` — monkeypatched safety check

### Level 3b — Training container Job smoke (Implemented)
- `TestTrainingJobSmokePrereqs` (4 local tests, no cluster needed):
  - `test_smoke_csv_has_required_columns` — embedded CSV has all TabFormer columns
  - `test_smoke_csv_has_sufficient_users` — ≥5 distinct users for 80/20 split
  - `test_smoke_shell_references_required_commands` — smoke shell calls fit_tokenizer + train_pragma
  - `test_job_manifest_has_required_fields` — manifest structure validated locally
- `TestTrainingJobSmoke` (1 opt-in cluster test, requires `RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1`):
  - `test_training_job_smoke` — submits a labelled `batch/v1 Job`, waits for completion,
    asserts "PRAGMA-S" and "Reached --max-steps" in pod logs.

**Note**: Level 3b proves the training image works. It does NOT prove KFP v2 pipeline
orchestration. Those are separate concerns at different maturity levels.

### Level 3 — OpenShift AI KFP v2 pipeline smoke (Implemented)
- `TestKFPPipelineSmokePrereqs` (5 local tests, no cluster needed):
  - `test_submit_module_importable_without_kfp` — lazy kfp import verified
  - `test_get_dspa_endpoint_uses_pragma_test_namespace` — endpoint URL construction
  - `test_dspa_config_token_not_in_repr` — SA token redaction
  - `test_wait_for_run_terminal_raises_timeout_error` — timeout semantics
  - `test_smoke_pipeline_module_contract` — smoke vs production module isolation
- `TestKFPDSPAConnectivity` (3 opt-in cluster tests, require `RUN_OPENSHIFT_PIPELINE_SMOKE=1`):
  - `test_dspa_endpoint_resolves_in_cluster` — endpoint construction in active namespace
  - `test_kfp_client_can_be_constructed` — kfp.Client init
  - `test_kfp_client_can_list_pipelines` — real DSPA API network call
- `TestKFPPipelineSmoke` (1 opt-in cluster smoke, requires `RUN_OPENSHIFT_PIPELINE_SMOKE=1`):
  - `test_kfp_pipeline_smoke_run` — compile → upload → create KFP Run → wait →
    assert SUCCEEDED + log markers (`PRAGMA-S`, `Reached --max-steps`,
    `PRAGMA smoke training completed`)

### Level 4 — PyTorchJob N-node smoke (gated by RUN_PYTORCHJOB_TESTS=1)

**10 tests total.** Tests 1–9 are static (no cluster resources). Test 10 requires `RUN_PYTORCHJOB_SMOKE=1`.

- `TestPyTorchJobCRD` (1 cluster read-only, requires `RUN_PYTORCHJOB_TESTS=1`):
  - `test_pytorchjob_crd_exists_when_enabled` — `pytorchjobs.kubeflow.org` CRD present (KFTO installed)
- `TestNNodeManifest` (5 static manifest checks, no cluster access):
  - `test_production_manifest_exists` — `openshift/training/pytorchjob-pragma-s-2node.yaml` present
  - `test_production_manifest_has_master_and_worker` — Master + Worker replicas defined
  - `test_production_manifest_has_no_canonical_pvc` — no PVC (uses emptyDir + S3 pattern)
  - `test_production_manifest_mentions_torchrun_or_distributed` — torchrun / WORLD_SIZE / MASTER_ADDR present
  - `test_production_manifest_warns_about_td006` — TD-006 --resume limitation documented
- `TestSmokeManifestNaming` (3 static DNS length checks, no cluster access):
  - `test_smoke_job_prefix_length_is_safe` — prefix ≤ 18 chars (leaves room for test_id + KFTO suffix)
  - `test_generated_master_pod_name_under_dns_limit` — master pod name ≤ 63 chars (RFC 1035)
  - `test_generated_worker_pod_name_under_dns_limit` — worker pod name ≤ 63 chars (RFC 1035)
- `TestPyTorchJobSmoke` (1 runtime test, requires `RUN_PYTORCHJOB_SMOKE=1`):
  - `test_pytorchjob_nnode_smoke` — applies a purpose-built N-node PyTorchJob (default nnodes=2),
    waits for all pods, asserts `Succeeded`, verifies `PRAGMA-S` / `Reached --max-steps` / DDP
    markers in logs, cleans up via label-scoped fixture.

**Architecture**: The Level 4 smoke validates the minimal distributed case (nnodes=2). The
architecture is N-node capable through `torchrun/KFTO`; larger N-node validation is
Level 6 (future scale testing).

### Level 5 — S3-backed checkpoint/resume (gated by RUN_OPENSHIFT_S3_RESUME_SMOKE=1)

**4 tests total.** 3 are local prereqs (no cluster). 1 is the runtime two-run smoke.

- `TestS3ResumeLocalPrereqs` (3 local prereqs, no cluster needed):
  - `test_s3_resume_job_prefix_safe` — `_S3_RESUME_JOB_PREFIX` leaves room for test_id + KFTO master suffix
  - `test_checkpoints_module_exists` — `src/training/checkpoints.py` exists
  - `test_train_pragma_has_resolve_resume_checkpoint` — `scripts/train_pragma.py` calls `resolve_resume_checkpoint`
- `TestS3CheckpointResumeSmoke` (1 runtime test, requires `RUN_OPENSHIFT_S3_RESUME_SMOKE=1`):
  - `test_s3_checkpoint_upload_and_all_rank_download` — two-run smoke:
    - Run 1: 2-node PyTorchJob trains 5 steps → uploads checkpoint to S3 (`Checkpoint uploaded` in logs)
    - Run 2: 2-node PyTorchJob resumes with `--resume` → all ranks independently download from S3
    - Asserts `Resuming from checkpoint` appears in run 2 logs (loaded by all ranks)
    - Asserts `RANK=1` appears in combined logs (worker pod executed independently)
    - Proves TD-006 is fixed: workers do not rely on rank 0's `emptyDir`

**Architecture**: All-rank S3 download implemented in `src/training/checkpoints.py`. Rank 0
selects the S3 key, broadcasts via `dist.broadcast_object_list`, all ranks download
independently to their own `emptyDir`, then `dist.barrier()`. 34 unit tests in
`tests/test_checkpoint_resume.py`. Cluster integration verified 2026-05-21 (4/4 PASSED, 70s).

### Level 7 — Production pipeline runtime — IBM TabFormer

**11 tests total.** 10 are local pre-flight checks (no cluster). 1 is the opt-in runtime test.

Gated by both `RUN_OPENSHIFT_TESTS=1` AND `RUN_OPENSHIFT_AI_PIPELINE_SMOKE=1`.

- `TestTabFormerPipelineLocalPrereqs` (10 local tests, no cluster needed):
  - `test_production_pipeline_importable` — `pragma_pretraining_pipeline` imports without error
  - `test_production_pipeline_has_max_steps` — pipeline signature exposes `max_steps`
  - `test_production_pipeline_has_limit_rows` — pipeline signature exposes `limit_rows`
  - `test_production_pipeline_has_batch_size` — pipeline signature exposes `batch_size`
  - `test_production_pipeline_has_device` — pipeline signature exposes `device`
  - `test_production_pipeline_has_run_name` — pipeline signature exposes `run_name`
  - `test_production_pipeline_compiles_to_yaml` — KFP Compiler produces non-empty YAML (kfp optional)
  - `test_ibm_tabformer_adapter_registered` — `get_adapter("ibm-tabformer")` resolves without KeyError
  - `test_components_use_wheel_based_training_invocation` — `pragma_encoder.training.train` in components source; no `/src/pragma_encoder` path
  - `test_export_checkpoint_writes_export_manifest` — `export_manifest.json` referenced in `export_checkpoint`
- `TestTabFormerPipelineRuntime` (1 opt-in cluster test, requires both gates):
  - `test_production_pipeline_reaches_succeeded` — compiles `pragma_pretraining_pipeline`, uploads
    to DSPA, submits Run with `max_steps=2, limit_rows=5, batch_size=1, device=cpu`, waits for
    `SUCCEEDED` state within timeout.

**Purpose**: Verifies the Workbench-to-pipeline-to-model-publication path for IBM TabFormer data.
Exercises all five §2.4 stages: prepare → upload → submit → train → export.

See `docs/tabformer-workbench-pipeline.md` for the end-to-end workflow description.

---

### Level 8 — Single-node GPU loss curve smoke — IBM TabFormer

**12 tests total.** 11 are static pre-flight checks (no cluster resources created). 1 is the opt-in runtime GPU test.

Gated by `RUN_OPENSHIFT_TESTS=1` AND `RUN_TABFORMER_LOSS_SMOKE=1` for static checks.
Runtime additionally requires `RUN_TABFORMER_LOSS_SMOKE_RUN=1`.

- `TestLossSmokeStaticPrereqs` (11 static tests, no cluster resources created):
  - `test_sample_fraction_within_bounds` — `PRAGMA_TABFORMER_SAMPLE_FRACTION` ≤ 0.20 (or override gate)
  - `test_max_steps_within_bounds` — `PRAGMA_TABFORMER_MAX_STEPS` ≤ 500 (or override gate)
  - `test_combined_bounds_consistent` — both fraction and max_steps pass safety gate together
  - `test_min_loss_points_is_positive` — `PRAGMA_MIN_LOSS_POINTS` ≥ 1
  - `test_default_min_loss_points_is_ten` — `_DEFAULT_MIN_LOSS_POINTS == 10` (hardcoded default)
  - `test_default_config_produces_enough_loss_points` — `100 // 5 = 20 ≥ 10` (default coherence)
  - `test_configured_values_produce_enough_loss_points` — `max_steps // log_every >= min_loss_points` (resolved values)
  - `test_job_name_fits_dns_label_limit` — master pod name ≤ 63 chars (RFC 1035 §2.3.4)
  - `test_pytorchjob_crd_present` — `pytorchjobs.kubeflow.org` CRD exists (cluster read)
  - `test_training_image_env_var_set` — `PRAGMA_TRAINING_IMAGE` is non-empty
  - `test_limit_rows_calculation_is_positive` — computed `limit_rows ≥ 10` customers
- `TestTabFormerLossSmokeRuntime` (1 opt-in GPU test, requires `RUN_TABFORMER_LOSS_SMOKE_RUN=1`):
  - `test_tabformer_gpu_loss_smoke` — submits a single-node Master-only GPU PyTorchJob that:
    1. Downloads IBM TabFormer CSV from S3 (`pragma-encoder/data/tabformer/card_transaction.v1.csv`)
    2. Fits vocab via `python -m pragma_encoder.data.fit_tokenizer`
    3. Runs `pragma-encoder-train` (PRAGMA-S, `--device cuda`) for `max_steps=100` steps
    4. Emits `metrics.jsonl` between `PRAGMA_LOSS_JSONL_BEGIN/END` markers in stdout
    5. Asserts: PyTorchJob `Succeeded`, ≥10 finite loss records, all finite, non-decreasing steps, ≥2 distinct steps
    6. Writes `test-artifacts/level8-tabformer-loss/<job_name>/loss.jsonl`, `metadata.json`, optional `loss.png`

**Purpose**: Proves the full training loop runs on GPU with real financial data and produces
a bounded, finite loss curve. No convergence or quality assertions.

**Coherence constraint** (checked statically): `max_steps // log_every >= min_loss_points`.
Default: `100 // 5 = 20 ≥ 10`. If you change defaults, the static tests will catch it.

See `tests/openshift/test_08_tabformer_loss_smoke.py` module docstring for full parameter reference.

---

## What Is Future / xfail

### Level 4 — N>2 node validation (scale testing)
Large N-node validation (N>2) is future scale testing. The current Level 4 smoke
proves the 2-node minimal distributed case. To test N>2:
set `PRAGMA_PYTORCHJOB_NNODES=<N>` and `PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1`.
This is not in CI; it is manual cluster validation only.

### Level 6 — GPU training smoke (scaffold implemented)

`test_06_gpu_training_smoke.py` is implemented but all tests skip unless
`RUN_OPENSHIFT_GPU_SMOKE=1`. The scaffold submits a `batch/v1 Job` to a GPU node,
runs PRAGMA-S training for 1 step, and asserts CUDA evidence in logs. Requires
a cluster node with `nvidia.com/gpu` capacity and `PRAGMA_TRAINING_IMAGE` set.

---

## Argo CD Interaction Model

Argo CD owns and reconciles the long-lived platform substrate in
`PRAGMA_TEST_NAMESPACE`. These tests treat that namespace as read-only
for substrate resources (ServiceAccount, Secrets, DSPA, ConfigMaps managed by Argo).

Ephemeral test resources (KFP v2 Run, PyTorchJob, Pod, Job) are created
in `PRAGMA_TEST_RUNTIME_NAMESPACE` (which may be the same namespace) with
test labels so they are clearly distinguishable from Argo-managed resources.

If Argo CD reconciles the namespace during a test run, it will not touch
test-labelled ephemeral resources (Argo ignores resources it does not manage).
