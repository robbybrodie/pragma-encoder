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

The gap is the transition from workbench authoring to **real OpenShift / OpenShift
Pipelines / PyTorchJob execution**. This suite fills that gap with a safe, opt-in
TDD loop for cluster-facing behaviour.

---

## Why a Separate Suite

Argo CD reconciles the PRAGMA platform substrate. These tests do not deploy the
platform; they verify it and create short-lived labelled runtime objects against it.

The separation exists because:

- Unit tests must always run fast and without cluster access.
- Cluster tests require authentication, namespace access, and running operators.
- Cluster tests create real resources and must clean them up safely.
- Making cluster tests opt-in prevents accidental cluster mutation in CI.

---

## Architecture

```
Argo CD  ─── deploys ──►  platform substrate  (long-lived, Argo-managed)
                           └── namespace
                           └── ServiceAccount
                           └── RBAC
                           └── Secrets (S3, image pull)

tests/openshift/  ─── verify substrate ──►  read-only checks (Levels 0–1)
                  ─── create (future) ──►   short-lived labelled resources
                                            └── PipelineRun (Level 3)
                                            └── PyTorchJob  (Level 4)
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
| 3 | OpenShift Pipelines smoke run | Future / xfail | `test_03_pipeline_smoke_run.py` |
| 4 | PyTorchJob / two-node smoke | Future / xfail | `test_04_pytorchjob_smoke.py` |
| 5 | S3-backed checkpoint/resume | Future | — |
| 6 | Scaled training validation | Future | — |
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
| `RUN_OPENSHIFT_PIPELINE_SMOKE=1` | (unset) | Opt-in for Level 3 PipelineRun smoke tests. |
| `RUN_PYTORCHJOB_TESTS=1` | (unset) | Opt-in for Level 4 PyTorchJob tests. |
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

### Future: OpenShift Pipeline smoke (Level 3, xfail until implemented)

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_OPENSHIFT_PIPELINE_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_03_pipeline_smoke_run.py -q
```

### Future: PyTorchJob smoke (Level 4, xfail until implemented)

```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_PYTORCHJOB_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_04_pytorchjob_smoke.py -q
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

### Level 1 — Argo-managed substrate
- `test_pipelinerun_crd_exists` — Tekton CRDs installed
- `test_taskrun_crd_exists`
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

### Level 4 — Manifest structure (read-only, gated by RUN_PYTORCHJOB_TESTS=1)
- `test_two_node_manifest_exists`
- `test_two_node_manifest_has_no_pvc_canonical_storage`
- `test_two_node_manifest_mentions_world_size_or_torchrun`
- `test_two_node_manifest_has_master_and_worker`

---

## What Is Future / xfail

### Level 3 — OpenShift Pipelines runtime smoke
`test_pipeline_smoke_run_future` — xfail. Submit a PRAGMA-S PipelineRun with
`max_steps=1`, wait for completion, verify logs, cleanup labelled resources.
Will xpass when pipeline submission is implemented in `src/workbench/`.

### Level 4 — PyTorchJob execution smoke
`test_pytorchjob_two_node_smoke_future` — xfail. Apply two-node manifest with
test labels, wait for Master + Worker pods, verify DDP logs, cleanup.
Will xpass when the safe apply + wait + cleanup path is implemented.

---

## Argo CD Interaction Model

Argo CD owns and reconciles the long-lived platform substrate in
`PRAGMA_TEST_NAMESPACE`. These tests treat that namespace as read-only
for substrate resources (ServiceAccount, Secrets, ConfigMaps managed by Argo).

Ephemeral test resources (PipelineRun, PyTorchJob, Pod, Job) are created
in `PRAGMA_TEST_RUNTIME_NAMESPACE` (which may be the same namespace) with
test labels so they are clearly distinguishable from Argo-managed resources.

If Argo CD reconciles the namespace during a test run, it will not touch
test-labelled ephemeral resources (Argo ignores resources it does not manage).
