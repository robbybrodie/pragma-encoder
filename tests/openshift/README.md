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
| DSPA/KFP v2 | 3 | **Product pipeline path.** Compiles a decorated pipeline, uploads to the DSPA API, creates a KFP Run, polls until Succeeded. This is the real execution path. | In progress / xfail |
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

---

## Maturity Levels

| Level | Name | Status | File |
|-------|------|--------|------|
| 0 | oc access + namespace checks | Implemented | `test_00_oc_access.py` |
| 1 | Argo-managed substrate verification | Implemented | `test_01_cluster_prereqs.py` |
| 2 | Decorated pipeline compile | Implemented | `test_02_pipeline_compile.py` |
| 3 | OpenShift AI KFP v2 pipeline smoke | In progress / xfail | `test_03_pipeline_smoke_run.py` |
| 3b | Training container Job smoke | Implemented | `test_03b_training_job_smoke.py` |
| 4 | PyTorchJob N-node smoke (default: nnodes=2) | Implemented | `test_04_pytorchjob_smoke.py` |
| 5 | S3-backed checkpoint/resume | Unit tests complete; cluster integration scaffold present | `test_05_s3_checkpoint_resume.py` |
| 6 | GPU training smoke (single-node opt-in) | Scaffold implemented | `test_06_gpu_training_smoke.py` |
| 7 | Bank-data adapter validation | Future | — |

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
| `RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1` | (unset) | Opt-in for Level 3b batch/v1 Job training container smoke. |
| `RUN_PYTORCHJOB_TESTS=1` | (unset) | Opt-in for Level 4 PyTorchJob tests (static checks + CRD check). |
| `RUN_PYTORCHJOB_SMOKE=1` | (unset) | Opt-in for Level 4 runtime smoke (creates a short-lived PyTorchJob). Also requires `RUN_PYTORCHJOB_TESTS=1`. |
| `PRAGMA_PYTORCHJOB_NNODES` | `2` | Number of nodes for the Level 4 N-node smoke. Minimum 2 (nnodes=1 is non-distributed; use Level 3b). Values >2 require `PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1`. |
| `PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1` | (unset) | Permit N>2 node smoke. Guards against accidental cluster overload. Required when `PRAGMA_PYTORCHJOB_NNODES > 2`. |
| `RUN_TEKTON_TESTS=1` | (unset) | Opt-in for Tekton PipelineRun/TaskRun CRD checks. Not required for OpenShift AI KFP v2. |
| `PRAGMA_TEST_TIMEOUT_SECONDS` | `300` | Timeout for cluster wait loops (minimum 30s). |

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

Expected: 5 prereqs PASS, 3 connectivity tests PASS, 1 smoke xfail.

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

### Level 3 — OpenShift AI KFP v2 pipeline smoke (In progress)
- `TestKFPPipelineSmokePrereqs` (5 local tests, no cluster needed)
- `TestKFPDSPAConnectivity` (3 opt-in cluster tests, require `RUN_OPENSHIFT_PIPELINE_SMOKE=1`)
- `TestKFPPipelineSmoke` (1 opt-in cluster smoke, xfails until `pragma_smoke_training_pipeline`
  component is implemented)

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

---

## What Is Future / xfail

### Level 3 — Full KFP v2 pipeline smoke (smoke component not yet implemented)
`TestKFPPipelineSmoke.test_kfp_pipeline_smoke_run` — xfail. Compiles a minimal
`pragma_smoke_training_pipeline`, uploads to DSPA API, creates a Run, polls
until Succeeded, asserts "PRAGMA-S" and "Reached --max-steps" in logs.
Will xpass when a new `smoke_training` KFP component is implemented that runs
`fit_tokenizer + train_pragma --max-steps 1` directly in the component pod
(without S3 or PyTorchJob).

### Level 4 — N>2 node validation (scale testing)
Large N-node validation (N>2) is future scale testing. The current Level 4 smoke
proves the 2-node minimal distributed case. To test N>2:
set `PRAGMA_PYTORCHJOB_NNODES=<N>` and `PRAGMA_ALLOW_LARGE_NNODE_SMOKE=1`.
This is not in CI; it is manual cluster validation only.

### Level 5 — S3-backed checkpoint/resume (TD-006 resolved)

TD-006 is **resolved**. `src/training/checkpoints.py` implements the all-rank download
pattern: rank 0 selects the S3 key, broadcasts it via `dist.broadcast_object_list`,
all ranks download independently, then barrier. 34 unit tests pass
(`tests/test_checkpoint_resume.py`). `scripts/train_pragma.py` uses
`resolve_resume_checkpoint()` and `upload_checkpoint_if_rank0()`.

The **cluster integration test** (`test_05_s3_checkpoint_resume.py`) is the remaining
gap: it would submit a PyTorchJob, let it save a checkpoint to S3, interrupt it, restart
it with `--resume`, and assert both ranks load from the same checkpoint. This requires
a real S3 bucket configured in the cluster and is future work.

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
