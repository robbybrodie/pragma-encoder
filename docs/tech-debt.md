# Technical Debt Register

## TD-001: Shared S3 bucket with nemo-tfm deployment

**Date:** 2026-05-17
**Severity:** Low (demo environment)
**Status:** Accepted for demo, must resolve before production

### Description
The pragma-encoder DSPA shares an S3 bucket with the
nemo-tfm deployment (transaction-foundation-model-openshiftai).
Pipeline artifacts from both deployments are stored in the
same bucket, differentiated only by KFP-generated run ID prefixes.

### Risk
- No hard isolation between the two deployments
- A bucket policy change affecting one affects both
- Cost attribution is unclear across deployments
- In production, data governance requires separate buckets
  per workload with separate IAM policies

### Resolution
Create a dedicated S3 bucket for pragma-encoder:
  Bucket: pragma-encoder-pipelines-<unique-suffix>
  IAM policy: scoped to pragma-encoder namespace only
  Region: same as nemo-tfm bucket

Re-seal the workbench runtime secret with new bucket values:
  kubeseal --scope namespace-wide \
    --namespace pragma-encoder \
    --format yaml \
    < openshift/secrets/workbench-secret.yaml \
    > openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml

### References
- Adjacent repo bucket: see nemo-tfm DSPA configuration
- OpenShift AI DSPA documentation: docs.redhat.com

---

## TD-002: NGC API key shared across deployments

**Date:** 2026-05-17
**Severity:** Low (demo environment)
**Status:** Accepted for demo, must resolve before production

### Description
The NGC API key used for pulling nvcr.io/nvidia/nemo:25.09.01
is the same key used in the nemo-tfm deployment.

### Risk
- Key rotation in one deployment requires updating both
- Rate limits on NGC API key are shared
- No audit trail separation between deployments

### Resolution
Generate a dedicated NGC API key for pragma-encoder:
  Log in to ngc.nvidia.com
  Generate a new API key scoped to pragma-encoder usage
  Re-seal registry-pull-secret.sealed.yaml with new key
  Rotate the nemo-tfm key independently

---

## TD-003: Profile tokenizer not implemented

**Date:** 2026-05-18
**Severity:** Medium (blocks real training on profile data)
**Status:** Partially resolved — xa zero-padded as shortcut

### Description
EmbeddingAssembler accepts xa_key_ids, xa_val_ids, xa_pos_ids
for the profile state path (static customer attributes).
These are produced correctly by the assembler via Equation 1.

However no ProfileTokenizerPipeline exists to produce these
inputs from real static customer attributes (plan, region,
balance quantile, life-long events).

FinancialTokenizerPipeline handles event tokens only.

### Impact
In the smoke test and early training: synthetic integer
tensors in valid ID ranges exercise the xa path correctly.
In production: callers must supply xa inputs manually
until ProfileTokenizerPipeline is implemented.

### Partial resolution (2026-05-18)
PragmaDataset (src/data/pragma_dataset.py) implements the
xa zero-padding shortcut:
  - na = 1 (single minimal profile token)
  - xa_key_ids: filled with vocab_spec.key_start
  - xa_val_ids: filled with vocab_spec.value_start
  - xa_pos_ids: filled with 0
  - ta: filled with 0.0

This allows real TabFormer training to proceed without
a ProfileTokenizerPipeline.

### Full resolution
Implement src/tokenizer/profile_pipeline.py:
  ProfileTokenizerPipeline that tokenises static customer
  attributes and life-long events, returning xa_key_ids,
  xa_val_ids, xa_pos_ids, ta consistent with VocabularySpec.

### References
  Paper: Section 2.1.2 (profile state definition)
  Paper: Section 2.3.2 (ProfileStateEncoder inputs)
  EmbeddingAssembler: src/model/assembler.py

---

## TD-004: RHOAI native pipeline-config injection bypassed

**Date:** 2026-05-18
**Severity:** Low (demo environment)
**Status:** Accepted for GitOps deployment; operator path not viable without dashboard

### Description
The RHOAI notebook-controller webhook injects `KFP_API_HOST` into workbench
pods by reading a namespace Secret named `ds-pipeline-config`. In a healthy
RHOAI cluster, this secret is created by the RHOAI dashboard when a user
connects a pipelines server to a workbench through the UI.

In a GitOps/bootstrap deployment (workbench declared via ArgoCD, not the
dashboard), no dashboard interaction occurs and the secret is never created.
Manually creating `ds-pipeline-config` does not work: the webhook rejects it
with "Skipping mounting secret not managed by workbenches" regardless of
labels or ownerReferences, because it was not created through the dashboard's
ownership chain.

### Workaround
`openshift/gitops/workbench/notebook.yaml` declares `KFP_API_HOST` and
`KF_PIPELINES_ENDPOINT` explicitly as env vars pointing to the
namespace-local DSPA service (port 8888, direct HTTPS, no OAuth proxy):

  https://ds-pipeline-pipelines-definition.pragma-encoder.svc.cluster.local:8888

This is equivalent to the operator-injected value and survives pod restarts.
The cluster CA bundle (`/etc/pki/tls/custom-certs/ca-bundle.crt`, injected
by the Notebook Controller) covers the DSPA TLS certificate.

### Resolution
If a dashboard-created workbench is ever needed, delete the GitOps-managed
Notebook CR, recreate the workbench through the RHOAI dashboard (connecting
the pipelines server), export the resulting Notebook CR, and replace the
manifest in openshift/gitops/workbench/notebook.yaml.

---

## TD-005: PAD token used as UNK corruption in MaskingStrategy

**Date:** 2026-05-18
**Severity:** Low (training correctness; PAD and UNK are both excluded from MLM loss)
**Status:** Known deviation — documented, not yet resolved

### Description
`src/masking/strategy.py` uses `TokenizerPipeline.PAD_ID = 0` as the
corruption token for the "UNK replacement" path in masking (§2.3.5).

The paper specifies that a fraction of selected positions are replaced
with the [UNK] token as input dropout (excluded from MLM loss).

`TokenizerPipeline` defines four specials: PAD (0), MASK (1), CLS (2), SEP (3).
There is no dedicated [UNK] token. PAD_ID is used in its place.

```python
# strategy.py
_UNK_TOKEN_ID: int = TokenizerPipeline.PAD_ID   # 0 — [UNK] replacement (no global UNK)
```

### Risk
- `EmbeddingAssembler` may interpret ID=0 as a PAD token (valid embedding slot)
  rather than as an [UNK] corruption — the embedding value differs from intent
- If a future VocabularySpec version reserves ID=0 for a different purpose,
  this silent aliasing becomes a correctness bug
- The PAD embedding is shared between genuine padding positions and UNK-corrupted
  positions — the model cannot distinguish them during training

### Resolution
Add a dedicated [UNK] token to `TokenizerPipeline`'s special token set:
  `UNK_ID = 4` (appended after SEP)

Update `VocabularySpec` and `EmbeddingAssembler` to allocate an embedding
for the new UNK ID. Update `_UNK_TOKEN_ID` in `strategy.py` to use
`TokenizerPipeline.UNK_ID`.

Alternatively, accept PAD-as-UNK as a permanent simplification and update
the comment in `strategy.py` to remove the "TODO" framing.

### References
- Paper: Section 2.3.5 (masking corruption)
- Code: `src/masking/strategy.py` — `_UNK_TOKEN_ID` constant
- Code: `src/tokenizer/pipeline.py` — special token IDs

---

## TD-006: Multi-node checkpoint resume assumes shared filesystem

**Date:** 2026-05-19
**Severity:** Medium (blocks fault-tolerant multi-node training; does not affect fresh runs)
**Status:** Resolved — 2026-05-21

### Description
`scripts/train_pragma.py` supports `--resume` to restart training from the
latest checkpoint. The original (broken) resume logic worked as follows:

1. Rank 0 searched locally, then downloaded from S3 if `--s3-checkpoint-prefix` was set.
2. Rank 0 broadcast a `found` flag (0 or 1) to all ranks via `dist.broadcast`.
3. Non-rank-0 workers then called `_find_latest_local_checkpoint(output_dir)` locally.

Step 3 assumed all ranks share the same filesystem (e.g. a shared NFS PVC).
In the two-node PyTorchJob manifests (`pytorchjob-pragma-s-2node.yaml`), each
pod has its own independent `emptyDir` for `/workspace`. The Worker pod's
`output_dir` was empty on startup, so `_find_latest_local_checkpoint` returned
`None` and the Worker began from a randomly initialised model state.

Result: rank 0 resumed from a checkpoint; rank 1 started from scratch.
DDP averaged gradients across ranks but did not re-synchronise starting
weights. Training continued with diverged model states.

### Resolution applied (2026-05-21)

Option A implemented: all ranks download the checkpoint from S3 independently.

**`src/training/checkpoints.py`** — new module implementing the all-rank download pattern:
- `parse_s3_config_from_env()` — reads native `AWS_*` env vars; returns `None` when absent
- `list_checkpoint_keys(client, bucket, prefix)` — lists `.pt` keys under an S3 prefix
- `select_latest_checkpoint_key(client, bucket, prefix)` — returns the latest key by `LastModified`
- `upload_checkpoint_if_rank0(ckpt_path, s3_prefix, is_rank0)` — rank-0-only upload (no-op on workers)
- `download_checkpoint_for_rank(key, output_dir)` — download by every rank to its own `emptyDir`
- `barrier_if_distributed(distributed)` — centralised `dist.barrier()` guard
- `resolve_resume_checkpoint(output_dir, s3_prefix, rank, distributed, device)` — main entry point:
  1. Rank 0 selects the latest key from S3.
  2. Rank 0 broadcasts the key string (not a tensor) via `dist.broadcast_object_list`.
  3. ALL ranks independently call `download_checkpoint_for_rank`.
  4. All ranks call `dist.barrier()` before returning.
  5. Falls back to local `.pt` scan in single-node mode when S3 is not configured.

**`scripts/train_pragma.py`** — updated to use `resolve_resume_checkpoint`:
- The broken `if args.resume:` block (17 lines with `dist.broadcast` of a boolean tensor +
  workers scanning their empty `emptyDir`) is replaced with a 9-line call.
- Dead helper functions removed: `_s3_client`, `_upload_checkpoint`,
  `_download_latest_checkpoint`, `_find_latest_local_checkpoint`.
- `upload_checkpoint_if_rank0` replaces the old `_upload_checkpoint` call.

**Unit tests**: `tests/test_checkpoint_resume.py` — 34 tests, all passing. Covers:
key selection, S3 config parsing, rank-0 upload semantics, all-rank download semantics,
broadcast + barrier coordination, local single-node fallback, no-shared-filesystem
invariant, `barrier_if_distributed` helper, and KFP import boundary.

**No kfp / kfp_kubernetes** imported in `checkpoints.py` (training-image-only module).
`boto3` imported lazily inside `_make_s3_client()` so the module loads without boto3.

### References
- Code: `src/training/checkpoints.py` (new — TD-006 fix implementation)
- Code: `scripts/train_pragma.py` (updated — uses `resolve_resume_checkpoint`)
- Tests: `tests/test_checkpoint_resume.py` (34 tests)
- Manifest: `openshift/training/pytorchjob-pragma-s-2node.yaml`
- ADR 003: docs/decisions/003-workbench-training-api.md (no PVC canonical storage)
- Paper: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4


---

## TD-007: pragma_smoke_pipeline component body uses source-tree paths

**Date:** 2026-05-21
**Severity:** Low (Level 3 cluster smoke only; no regression in current tests)
**Status:** Resolved — 2026-05-21

### Description

`pipeline/pragma_smoke_pipeline.py` contained a KFP component function body
(`pragma_smoke_training`) that located `fit_tokenizer.py` by searching for:

```python
_candidate / "src" / "pragma_encoder" / "data" / "fit_tokenizer.py"
```

And invoked it as:

```python
subprocess.run([sys.executable, str(project_root / "src" / "pragma_encoder" / "data" / "fit_tokenizer.py")])
```

Since the training image installs the `pragma_encoder` wheel (commit d15a249)
rather than copying the source tree, there is no `src/` directory in the image.
This path search failed at pod startup with a RuntimeError when the component
pod used the wheel-based training image.

### Resolution applied (2026-05-21)

**`pipeline/pragma_smoke_pipeline.py`** — component body updated:

1. `_search_roots` loop (looking for `src/pragma_encoder/data/fit_tokenizer.py`) removed.
2. fit_tokenizer invocation changed from direct file execution to module invocation:
   ```python
   subprocess.run([sys.executable, "-m", "pragma_encoder.data.fit_tokenizer"], cwd=str(work_dir), check=True)
   ```
3. `scripts/train_pragma.py` path now found via `_script_search_dirs` loop
   (searching for `scripts/train_pragma.py` at `/opt/app-root/src`, `/opt/app-root/src/pragma-encoder`,
   `/pragma-encoder`, and CWD), consistent with the fix applied to
   `test_05_s3_checkpoint_resume.py` (commit 787e55d).

**`tests/test_smoke_pipeline.py`** — new static tests added:

- `TestSmokePipelineSource::test_component_uses_module_invocation_for_fit_tokenizer`
  — asserts `-m` flag and `pragma_encoder.data.fit_tokenizer` in source
- `TestSmokePipelineSafety::test_component_has_no_src_tree_assumption`
  — asserts old source-tree path patterns are absent
- `TestSmokePipelineCompile::test_compiled_yaml_has_no_src_tree_paths`
  — asserts compiled KFP YAML does not embed source-tree paths (skips without kfp)

### References

- Code: `pipeline/pragma_smoke_pipeline.py` — `pragma_smoke_training` component body
- Tests: `tests/test_smoke_pipeline.py`
- Prior fix (same pattern): `tests/openshift/test_05_s3_checkpoint_resume.py` (commit 787e55d)
- Image wheel refactor: `openshift/training/Dockerfile.training` (commit d15a249)
- Test: `tests/openshift/test_03_pipeline_smoke_run.py` (xfail — Level 3 cluster smoke)

---

## TD-008: Training entrypoint depends on repo-copied scripts/train_pragma.py

**Date:** 2026-05-21
**Severity:** Low (operational; runtime correctness not affected)
**Status:** Resolved — 2026-05-21

### Description

`scripts/train_pragma.py` was the sole training entrypoint: the KFP smoke pipeline,
the PyTorchJob manifest, and the Level 5 S3 resume manifest all located and executed
it by searching filesystem paths at runtime. The file contained a `sys.path.insert`
to add the repo root to `sys.path`, making it incompatible with wheel-only images.

### Resolution applied (2026-05-21)

**`src/pragma_encoder/training/train.py`** — new canonical training module:
- Contains all training logic from the former `scripts/train_pragma.py`
- `sys.path.insert` removed (wheel-installed — no source tree needed)
- `parse_args(argv=None)` — accepts argv list for programmatic invocation
- `main(argv: list[str] | None = None) -> int` — returns 0 on success
- `if __name__ == "__main__": raise SystemExit(main())`

**`pyproject.toml`** — new `[project.scripts]` section:
```
pragma-encoder-train = "pragma_encoder.training.train:main"
```
The wheel now installs `pragma-encoder-train` into PATH.

**`scripts/train_pragma.py`** — reduced to a thin compatibility wrapper:
```python
from pragma_encoder.training.train import main
raise SystemExit(main())
```

**`openshift/training/Dockerfile.training`** — updated:
- Comment updated: scripts/ is now compatibility-only
- New RUN step verifies `pragma-encoder-train --help` at build time

**`pipeline/pragma_smoke_pipeline.py`** — updated:
- Removed `_script_search_dirs` loop and `train_script` path search
- Training now invoked as: `[sys.executable, "-m", "pragma_encoder.training.train", ...]`

**`openshift/training/pytorchjob-pragma-s.yaml`** — updated:
- Main container image changed from `pragma-encoder-workbench` to `pragma-encoder-training`
- Training command changed from `python scripts/train_pragma.py` to `python -m pragma_encoder.training.train`
- Init container: git clone step removed (code is now baked into training image)

**`tests/openshift/test_05_s3_checkpoint_resume.py`** — updated:
- `_render_s3_resume_manifest`: shell loop searching for `scripts/train_pragma.py` removed
- torchrun now invokes `-m pragma_encoder.training.train`

**Tests updated:**
- `tests/test_image_contract.py` — `TestKfpKubernetesBoundary`: added `_TRAIN_MODULE` path,
  new tests for `train.py` kfp boundary and `resolve_resume_checkpoint` presence;
  `test_train_script_references_resolve_resume_checkpoint` now checks wrapper imports `train.py`
- `tests/test_s3_manifest_render.py` — `test_train_pragma_has_resolve_resume_checkpoint`
  now checks `train.py` not `scripts/train_pragma.py`
- `tests/test_smoke_pipeline.py` — `test_component_source_references_train_pragma_max_steps`
  now checks for `pragma_encoder.training.train`; compiled YAML test updated
- `tests/test_packaging.py` — new `TestConsoleScript` class; new static/artifact checks
  for the console script entry point

### References
- Code: `src/pragma_encoder/training/train.py` (new — TD-008 fix implementation)
- Code: `scripts/train_pragma.py` (compatibility wrapper)
- Code: `pyproject.toml` — `[project.scripts]`
- Code: `openshift/training/Dockerfile.training`
- Code: `openshift/training/pytorchjob-pragma-s.yaml`
- Tests: `tests/test_packaging.py` — `TestConsoleScript`

---

## TD-009: pragma_encoder.workbench lives in the core wheel

**Date:** 2026-05-21
**Severity:** Low (already isolated; training image unaffected; workbench is optional extras only)
**Status:** Resolved — 2026-05-21

### Description

`pragma_encoder.workbench` was a **platform-aware** subpackage living inside
`src/pragma_encoder/workbench/` — within the core wheel distribution. This was
architecturally impure: a platform-aware subpackage in an otherwise platform-neutral wheel.

### Resolution applied (2026-05-21)

Workbench helpers moved from `src/pragma_encoder/workbench/` to `tools/workbench/`.

Key changes:
- `src/pragma_encoder/workbench/` deleted. `import pragma_encoder.workbench` now raises
  `ModuleNotFoundError`.
- New location: `tools/workbench/` — NOT part of the installed wheel.
  `tools/` is not under `src/` so `[tool.setuptools.packages.find]` never discovers it.
- Importable as `tools.workbench` when `PYTHONPATH=.` is set (the standard
  test invocation), or when the repo root is on `sys.path`.
- Internal cross-module imports within `tools/workbench/` use relative imports
  (`from ._run import`, `from ._intent import`).
- The `[workbench]` optional-dependencies entry in `pyproject.toml` removed (it referenced
  kfp/kfp-kubernetes; these remain optional for the pipeline/ directory which uses them).
- All tests and examples updated from `pragma_encoder.workbench.*` to
  `tools.workbench.*`.
- `tests/test_platform_neutral_wheel.py::TestWorkbenchRemovedFromWheel` mechanically verifies:
  - `import pragma_encoder.workbench` raises `ModuleNotFoundError`
  - `pragma_encoder` has no `.workbench` attribute
  - Built wheel zip contains no `pragma_encoder/workbench/` entries
  - `tools.workbench` is importable from the repo root
  - `kfp` is not in core `[project.dependencies]`

### References
- Code: `tools/workbench/` — new workbench location (not in wheel)
- Tests: `tests/test_platform_neutral_wheel.py::TestWorkbenchRemovedFromWheel`
- Docs: `docs/openshift-ai-3.3-alignment.md` — Platform Neutrality section
- ADR 003: `docs/decisions/003-workbench-training-api.md`
- ADR 004: `docs/decisions/004-workbench-decorated-pipelines.md`

---

## TD-010: SealedSecret carries legacy MODEL_REGISTRY_* key names

**Date:** 2026-05-22
**Severity:** Low (functional; emits DeprecationWarning; no production credential risk)
**Status:** Resolved — 2026-05-22 (SealedSecret re-sealed with native AWS_* keys)

### Description

The live SealedSecret (`openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml`)
was sealed with legacy custom key names:

```
MODEL_REGISTRY_ACCESS_KEY   → should be: AWS_ACCESS_KEY_ID
MODEL_REGISTRY_SECRET_KEY   → should be: AWS_SECRET_ACCESS_KEY
MODEL_REGISTRY_ENDPOINT     → should be: AWS_S3_ENDPOINT
MODEL_REGISTRY_BUCKET       → should be: AWS_S3_BUCKET
```

The native OpenShift AI S3 Connection schema (source of truth: `redhat-ods-applications/s3`
ConfigMap) uses the `AWS_*` names. The RHOAI dashboard injects these into pods as env vars
when a Connection is attached.

Additionally, the SealedSecret carried two non-S3 credentials that do not belong in an
S3 Connection:
- `AI_PLATFORM_API_URL` — RHOAI dashboard URL; belongs in a platform ConfigMap
- `NGC_API_KEY` — NVIDIA registry credential; belongs in a separate `registry-credentials` Secret

### Resolution applied (2026-05-22)

All steps completed:

1. `openshift/secrets/workbench-secret.yaml` rewritten with native `AWS_*` keys and
   `opendatahub.io/*` RHOAI annotations. `NGC_API_KEY` and `AI_PLATFORM_API_URL` removed.

2. Re-sealed with kubeseal — `openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml`
   now carries only `AWS_*` keys. Verified: `oc get secret pragma-workbench-env -o json | jq '.data | keys'`.

3. `MODEL_REGISTRY_*` fallback code removed from `checkpoints.py` and `ibm_tabformer.py`.

4. All manifests (`dspa.yaml`, `pytorchjob-pragma-s.yaml`, `pytorchjob-pragma-s-2node.yaml`,
   `pytorchjob-pragma-m.yaml`) updated to use native `AWS_*` field names.

5. NGC credentials remain in the separate `registry-credentials` Secret (correct).

### References
- Sealed Secret: `openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml`
- Template: `openshift/secrets/workbench-secret.template.yaml`
- Code: `src/pragma_encoder/training/checkpoints.py` — `parse_s3_config_from_env()`
- Code: `src/pragma_encoder/data/adapters/ibm_tabformer.py` — `_upload_to_s3()`
- Schema source of truth: `oc get cm s3 -n redhat-ods-applications -o yaml`
- Tests: `tests/test_checkpoint_resume.py::TestS3ConfigFromEnv::test_deprecated_model_registry_fallback_emits_warning`
- Tests: `tests/test_openshift_ai_fixtures.py::TestObjectStorageConnectionFixture::test_no_model_registry_aliases_in_connection_fixture`

---

## TD-011: Two-run local training smoke test deferred (requires real data files)

**Date:** 2026-05-22
**Severity:** Low (local correctness already covered by unit tests; smoke deferred only)
**Status:** Open — deferred pending data availability

### Description

A full two-run local training smoke test — where a real training run writes a
checkpoint to `--output-dir` and a second run resumes from it — requires the
IBM TabFormer dataset files:

```
data/tabformer/card_transaction.v1.csv
data/tabformer/vocab.pkl
```

These files cannot be committed to the repository (proprietary data). Without
them, a real end-to-end smoke test that exercises `pragma-encoder-train`,
`torch.save`, and `torch.load` across two process invocations cannot run in
the default `pytest tests/` suite.

### What is already covered

The following local checkpoint behaviours are verified without data files:

- `TestLocalCheckpointModeFirstClass` (9 tests): `build_checkpoint_store`,
  `LocalCheckpointStore.latest_key/fetch/put`, and `resolve_resume_checkpoint`
  all tested with real `AWS_*` env var clearing (no mocking of
  `parse_s3_config_from_env`). Proves local mode requires no S3 credentials.

- `TestLocalCheckpointRoundTrip` (2 tests): Full lifecycle with `torch.save`
  and `torch.load` — write a real `.pt` checkpoint, discover it via
  `store.latest_key()`, fetch, load, verify contents, confirm `put()` is a
  no-op. Multiple-epoch supersession also covered.

These tests cover the storage adapter contract end-to-end. The missing piece
is a black-box CLI smoke: two subprocess calls to `pragma-encoder-train` with
real data files.

### Resolution

When `data/tabformer/card_transaction.v1.csv` and `data/tabformer/vocab.pkl`
are available in the test environment (e.g. downloaded from S3 in CI), add:

```python
# tests/test_local_training_smoke.py
@pytest.mark.skipif(not DATA_AVAILABLE, reason="tabformer data not available")
def test_two_run_local_resume():
    """Run 1 trains 1 epoch. Run 2 resumes and trains 1 more epoch."""
    # subprocess.run(["pragma-encoder-train", ..., "--epochs", "1"])
    # assert checkpoint_epoch0001.pt exists
    # subprocess.run(["pragma-encoder-train", ..., "--resume", "--epochs", "2"])
    # assert checkpoint_epoch0002.pt exists
    # assert epoch in torch.load(ckpt) == 2
```

Gate with `RUN_LOCAL_TRAINING_SMOKE=1` to keep the default suite data-free.

### References
- Tests: `tests/test_checkpoint_resume.py::TestLocalCheckpointModeFirstClass`
- Tests: `tests/test_checkpoint_resume.py::TestLocalCheckpointRoundTrip`
- Docs: `docs/training-guide.md` — Local training section
- Code: `src/pragma_encoder/training/train.py` — `pragma-encoder-train` entrypoint

---

## TD-012: Evaluate RHOAI 3.4 TrainJob + Kubeflow Trainer v2 checkpointing

**Date:** 2026-05-24
**Severity:** Medium (affects forward platform path; current PyTorchJob path unaffected)
**Status:** Open — evaluation pending; PyTorchJob path remains production

### Description

Kubeflow Training Operator v1 source code has been **removed** from the upstream
`kubeflow/trainer` repository (Feb 2025, PR #2389). Kubeflow docs redirect all
v1 pages to v2 with deprecation notices. RHOAI 3.4 ships `ClusterTrainingRuntime`
objects for Kubeflow Trainer v2 (TrainJob), indicating the platform direction.

Whether RHOAI 3.4 still ships `kubeflow.org/v1` CRDs for backward compatibility
is **unverified on the live cluster**. If v1 CRDs are absent, Level 5 cluster
tests will fail immediately.

Kubeflow Trainer v2 introduces resilient checkpointing (since RHOAI 3.2):
- **JIT (Just-In-Time):** saves on SIGTERM/preemption before pod exits
- **Periodic:** saves at a configured time/step interval

The SDK-native checkpointing (automatic, no custom code) applies only to
HuggingFace trainers. PRAGMA uses a custom PyTorch loop — the following gaps
must be closed before TrainJob + platform-managed checkpointing is viable:

| Gap | Description | Effort |
|---|---|---|
| JIT checkpoint on SIGTERM | No SIGTERM handler in `train.py` | Medium |
| Checkpoint dir from env | SDK injects config for HF trainers; custom loops must read manually | Medium |
| PVC checkpoint backend | ADR 003 says no data PVCs; PVC backend requires new ADR | Architectural |
| S3 checkpoint backend via TrainJob | Not confirmed from official docs | [VERIFY] |
| Interval-based checkpointing | Epoch-end only today; step-count hook needed | Low-Medium |

The current Level 5 S3 checkpoint/resume path (`checkpoints.py`) remains valid
as the production path for:
- RHOAI 3.3/3.4 PyTorchJob (if still available on cluster)
- Any future TrainJob + application-managed S3 path
- Local training (no cluster)

Level 5 is **not** the forward platform path if TrainJob GA + PVC checkpointing
satisfies requirements and ADR 003 is superseded.

### Acceptance criteria

All of the following must be verified on a live RHOAI 3.4 cluster before
changing the production training path:

1. `oc api-resources | grep kubeflow` — `pytorchjobs` CRD is present (or absent, which triggers urgent migration)
2. `oc api-resources | grep trainer` — `trainjobs` CRD is present at `trainer.kubeflow.org/v1alpha1`
3. TrainJob status confirmed: GA or still Tech Preview in RHOAI 3.4
4. Submit a minimal TrainJob using `pragma-encoder-training` image — completes successfully
5. PVC checkpoint backend: checkpoint written to PVC, resumed in second run
6. S3 checkpoint backend: confirmed or ruled out from RHOAI 3.4 docs/cluster test
7. JIT checkpoint on SIGTERM: SIGTERM sent to training pod, checkpoint saved before exit
8. PRAGMA custom loop: define wiring pattern for reading checkpoint config from env
9. Results documented; decision matrix in `docs/rhoai-3.4-trainjob-checkpointing.md §8` updated
10. New ADR written (supersedes ADR 005) if TrainJob adoption is recommended

### Resolution

When all acceptance criteria are met, one of:

**Option A — TrainJob + PVC backend adopted:**
- New ADR superseding ADR 005
- New Level 4b/5b cluster smoke test for TrainJob + PVC checkpoint
- `checkpoints.py` S3 path retained as local/fallback path only
- Production PyTorchJob manifests replaced with TrainJob manifests

**Option B — TrainJob + S3 backend adopted (application-managed retained):**
- New ADR noting TrainJob replaces PyTorchJob; S3 checkpointing unchanged
- `checkpoints.py` remains production code
- TrainJob manifests created in `openshift/training/`

**Option C — PyTorchJob retained (if still GA in RHOAI 3.4+):**
- Close this TD as "WONTFIX / deferred"
- Monitor RHOAI release notes for CRD removal timeline
- Continue Level 5 as-is

### References
- Evaluation doc: `docs/rhoai-3.4-trainjob-checkpointing.md` (full responsibility table)
- Fixture: `tests/openshift/fixtures/trainjob-example.yaml` (updated for RHOAI 3.4 eval)
- ADR 005: `docs/decisions/005-training-orchestration.md`
- ADR 003: `docs/decisions/003-workbench-training-api.md` (no PVC canonical storage)
- Level 5 tests: `tests/openshift/test_05_s3_checkpoint_resume.py`
- Code: `src/pragma_encoder/training/checkpoints.py`
- Upstream: kubeflow/trainer PR #2389 (Training Operator v1 source removal)
