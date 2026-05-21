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
- `parse_s3_config_from_env()` — reads `MODEL_REGISTRY_*` env vars; returns `None` when absent
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
**Status:** Accepted for current release — boundary enforced by tests and docstring labelling

### Description

`pragma_encoder.workbench` is a **platform-aware** subpackage. It:

- Reads Kubernetes service account tokens from `/var/run/secrets/kubernetes.io/serviceaccount/token`
- Constructs DSPA/KFP endpoint URLs from Kubernetes namespace information
- Wraps `kfp.Client` for pipeline upload and run submission to OpenShift AI DSPA
- Guards `kfp-kubernetes` as an optional lazy import for Kubernetes-native pipeline features

It lives inside `src/pragma_encoder/workbench/` — within the core wheel distribution.
This is architecturally impure: a platform-aware subpackage in an otherwise platform-neutral wheel.

### Why it is currently acceptable

The isolation is already strong:

1. `import pragma_encoder` does NOT import workbench. The top-level `__init__.py` contains no
   import from `pragma_encoder.workbench`. No training-image code path triggers workbench.

2. `kfp` and `kfp-kubernetes` are `[workbench]` optional extras in `pyproject.toml`.
   They are not installed in the training image or in CI without `[dev,workbench]` extras.

3. `pragma_encoder.workbench.__init__` is explicitly annotated as a platform-aware optional
   subpackage, with a reference to TD-009.

4. `tests/test_platform_neutral_wheel.py` enforces the boundary mechanically:
   - No kfp import statements in non-workbench core modules
   - No Kubernetes pod paths in non-workbench modules
   - No OpenShift CRD resource names in core source
   - Top-level init has no workbench import statements

### Resolution

Move `pragma_encoder.workbench` to a separate distribution package:

Option A (preferred):
  Package: `pragma-encoder-openshift-ai` or `pragma-encoder-platform`
  Module:  `pragma_encoder_platform.workbench` or `pragma_encoder_openshift_ai`
  Install: `pip install 'pragma-encoder-openshift-ai'` (depends on `pragma-encoder[workbench]`)

Option B:
  Keep workbench in the wheel under a clearly named namespace:
  `pragma_encoder.platform.openshift_ai`
  (rename from `pragma_encoder.workbench`)

Prerequisites for resolution:
- ADR superseding ADR 003 / ADR 004 documenting the new package boundary
- Update workbench image requirements.txt import paths
- Update any external callers of `from pragma_encoder.workbench import ...`
- Ensure `pragma-encoder-train` console script still works (it does — training is not workbench)

### References
- Code: `src/pragma_encoder/workbench/` — the platform-aware subpackage
- Tests: `tests/test_platform_neutral_wheel.py` — boundary enforcement
- Docs: `docs/openshift-ai-3.3-alignment.md` — Platform Neutrality section
- ADR 003: `docs/decisions/003-workbench-training-api.md`
- ADR 004: `docs/decisions/004-workbench-decorated-pipelines.md`
