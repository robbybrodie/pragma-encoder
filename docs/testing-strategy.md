# PRAGMA Encoder — Testing Strategy

Tests in this repo have a single source of truth: the PRAGMA paper
(Ostroukhov et al., 2026, arXiv:2604.08649v1). Every number in a test must
appear in `docs/paper/key-numbers.md` with its section reference.

This document describes:
- The four-tier test model and when each tier runs
- Recommended pytest commands for each tier
- Classification of every test file (approx. 950 tests, 50 files)
- The stale-phrase guard policy
- Rules for adding new tests

---

## Four-tier test model

| Tier | Name | Runs | Prerequisites |
|------|------|------|---------------|
| 1 | Fast unit tests | Always (CI default) | Python + torch (skipped gracefully if absent) |
| 2 | Static platform contract | Always (CI default) | Python only — no cluster, no S3 |
| 3 | Opt-in runtime | `RUN_OPENSHIFT_TESTS=1` | Logged-in `oc`, live cluster, S3 credentials |
| 4 | Manual / demo | Never automated | Human in workbench |

The CI pipeline runs **Tier 1 + Tier 2** on every push. Tier 3 is triggered
manually or by specific environment flags. Tier 4 is never automated.

---

## Tier 1 — Fast unit tests

No cluster, no S3, no network. Torch is optional (`pytest.importorskip`).
These are the inner-loop tests; a developer should be able to run them in <2 min.

**Run with:**
```bash
pytest tests/ --ignore=tests/openshift/ -x -q
```

| Test file | Count | What it protects |
|---|---|---|
| `test_config.py` | 14 | `PRAGMAConfig` sizes match Table 1; no hardcoded constants |
| `test_vocabulary.py` | 29 | Tokenizer vocabulary structure from §2.2 |
| `test_tokenizer.py` | 19 | Key-value-time tokenisation strategy (§2.2) |
| `test_assembled_batch.py` | 19 | Assembled batch shape contract |
| `test_assembler.py` | 19 | Batch assembly correctness |
| `test_event_encoder.py` | 21 | `EventEncoder` calendar embeddings (§2.3.3) |
| `test_history_encoder.py` | 16 | `HistoryEncoder` bidirectional attention over `[USR:EVT]` (§2.3.4) |
| `test_profile_state_encoder.py` | 12 | `ProfileStateEncoder` (§2.3.2) |
| `test_rope.py` | 8 | RoPE implementation (§2.3.4) |
| `test_masking.py` | 17 | Three-strategy masking (§2.3.5) |
| `test_event_valid_masking.py` | 5 | Event-level mask validity |
| `test_mlm_head.py` | 14 | MLM head shape and loss (§2.3.5) |
| `test_model.py` | 14 | Full `PRAGMAModel` forward pass |
| `test_probe.py` | 13 | `EmbeddingProbe` linear probe (§3.1.1) |
| `test_lora.py` | 11 | `LoRAAdapter` fine-tuning (§3.1.2) |
| `test_dataset.py` | 5 | Dataset container |
| `test_dataset_manifest.py` | 27 | Dataset manifest serialisation |
| `test_dataset_adapters.py` | 17 | IBM TabFormer adapter (§2.2 data pipeline) |
| `test_learning_validation.py` | 26 | Loss decreases over training steps |
| `test_imports.py` | 22 | All `src/pragma_encoder` modules import without error |
| `test_smoke.py` | 1 | Minimal end-to-end smoke (no torch dependency) |

---

## Tier 2 — Static platform contract

No cluster, no S3. These tests verify deployment contracts, tool boundaries,
packaging invariants, and security guards. They are fast (static analysis of
files and Python imports), but they encode deliberate architectural decisions
that must not regress silently.

**Run with:**
```bash
pytest tests/ --ignore=tests/openshift/ -x -q
# (same command — Tier 1 and Tier 2 share the same default run)
```

To run Tier 2 in isolation:
```bash
pytest tests/ --ignore=tests/openshift/ -x -q \
  -k "contract or guard or image or platform or workbench or pipeline or packaging or s3_manifest"
```

| Test file | Count | What it protects |
|---|---|---|
| `test_platform_neutral_wheel.py` | 13 | `pragma_encoder` core has no kfp/K8s/platform imports (TD-009 boundary) |
| `test_image_contract.py` | 24 | Two-image model: workbench image vs training image; kfp boundary |
| `test_stale_contract_references.py` | 7 | `MODEL_REGISTRY_*` absent from code (TD-010); `train_pragma.py` stays thin |
| `test_gitops_secret_guard.py` | 10 | No plaintext creds in gitops/; only SealedSecrets; templates use `REPLACE_ME` |
| `test_openshift_ai_primitive_contract.py` | 72 | RHOAI 3.3/3.4 primitive YAML contract + TD-012 evaluation guards (no cluster) |
| `test_openshift_ai_fixtures.py` | 32 | Fixture YAML conforms to native AWS_* S3 Connection schema |
| `test_pipeline_components.py` | 70 | KFP v2 pipeline component static structure: signatures, stage names, no PVC |
| `test_smoke_pipeline.py` | 21 | Smoke pipeline static: no cluster needed, compiles cleanly |
| `test_s3_manifest_render.py` | 11 | S3 manifest YAML renderer (extracted from openshift/ to always run in CI) |
| `test_packaging.py` | 20 | pyproject.toml build config; wheel METADATA; editable + wheel install |
| `test_checkpoint_resume.py` | 63 | S3 checkpoint/resume logic (all-rank download pattern, TD-006 fix) |
| `test_training_artifact.py` | 48 | Training artifact contract: checkpoint + metadata.json (no S3/cluster) |
| `test_workbench_api.py` | 48 | `train_pragma()` workbench API (dry-run mode; no cluster) |
| `test_workbench_decorators.py` | 65 | Pipeline decorator API: compile, show_pipeline, intent capture |
| `test_workbench_submit.py` | 45 | DSPA submit path contract (mocked kfp.Client, no cluster) |
| `test_pragma_run.py` | 30 | `PragmaRun` / `PipelineStep` five-stage invariants (§2.4) |

---

## Tier 3 — Opt-in runtime tests (`tests/openshift/`)

Requires a live OpenShift AI cluster, logged-in `oc` session, the built
training image, and S3 credentials. Never run automatically in CI.

**Run with:**
```bash
# Minimal: cluster access + namespace verification only
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=image-registry.openshift-image-registry.svc:5000/pragma-encoder/pragma-encoder-training:latest \
pytest tests/openshift/ -x -v

# PyTorchJob smoke (adds single-node training job)
RUN_OPENSHIFT_TESTS=1 RUN_PYTORCHJOB_TESTS=1 RUN_PYTORCHJOB_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<image> \
pytest tests/openshift/ -x -v

# Full suite including GPU, S3 resume, and pipeline smoke
RUN_OPENSHIFT_TESTS=1 \
RUN_PYTORCHJOB_TESTS=1 RUN_PYTORCHJOB_SMOKE=1 \
RUN_OPENSHIFT_S3_RESUME_SMOKE=1 \
RUN_OPENSHIFT_GPU_SMOKE=1 \
RUN_OPENSHIFT_PIPELINE_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<image> \
pytest tests/openshift/ -v
```

| Test file | Level | What it tests |
|---|---|---|
| `test_00_oc_access.py` | L0 | `oc` binary present, authenticated, namespace accessible |
| `test_01_cluster_prereqs.py` | L1 | Cluster prerequisites: GPU node, RHOAI version, Sealed Secrets |
| `test_02_pipeline_compile.py` | L2 | Pipeline YAML compiles and is uploadable to DSPA |
| `test_02b_image_contract_runtime.py` | L2b | Training image entrypoint, installed packages (live image) |
| `test_03_dspa_runtime_discovery.py` | L3 | DSPA endpoint discovery in-cluster |
| `test_03_pipeline_smoke_run.py` | L3 | End-to-end smoke pipeline run on DSPA |
| `test_03b_training_job_smoke.py` | L3b | Single-node CPU training smoke (no GPU) |
| `test_03e_kfp_client_probe.py` | L3e | kfp.Client connects to live DSPA |
| `test_04_pytorchjob_smoke.py` | L4 | PyTorchJob single-node smoke (real GPU) |
| `test_05_s3_checkpoint_resume.py` | L5 | S3 checkpoint write + resume across two training runs |
| `test_06_gpu_training_smoke.py` | L6 | Multi-epoch GPU training smoke with checkpoint upload |

All tests in `tests/openshift/` skip automatically unless `RUN_OPENSHIFT_TESTS=1`
is set. See `tests/openshift/conftest.py` for the skip guard implementation.

---

## Tier 4 — Manual / demo

Not automated. Run these in the workbench or from a local terminal:

```python
# Show pipeline stages without compiling
run.show_pipeline()

# Compile to YAML
run.compile("pipeline/generated/pragma-s-ibm-tabformer.yaml")

# Dry-run train_pragma (no data, no cluster)
from tools.workbench import train_pragma
result = train_pragma(model_size="S", epochs=1, mode="dry_run")
result.show_pipeline()
```

---

## Stale-phrase guard policy

Some tests guard phrase-level properties of source files, YAML, or docs.
These are a double-edged sword: they catch real regressions, but they also
become maintenance drag if overused.

### Acceptable guard tests

A phrase/wording guard test is acceptable if and only if:

1. **It protects a real regression boundary** — a schema migration, security
   invariant, or architectural contract that would cause a silent breakage if
   violated.
2. **It catches a developer mistake not caught by any other mechanism** — linting,
   type checking, or import-time errors don't cover it.
3. **The guarded property is expected to be stable** — it is not prose that
   evolves naturally with documentation updates.

| Guard | File | Risk level | Verdict |
|---|---|---|---|
| `MODEL_REGISTRY_*` absent from code | `test_stale_contract_references.py` | High — TD-010 schema migration; wrong env var breaks S3 silently | **Keep** |
| `train_pragma.py` is thin (no business logic) | `test_stale_contract_references.py` | Medium — architectural boundary | **Keep** |
| kfp not in core `pyproject.toml` deps | `test_image_contract.py` | High — breaks wheel in training image | **Keep** |
| kfp not imported in training modules | `test_image_contract.py` | High — breaks training image at import time | **Keep** |
| `pragma-workbench-env` Secret name | `test_platform_neutral_wheel.py` | Medium — RHOAI Connection name contract | **Keep** |
| `pragma_encoder.workbench` not importable from wheel | `test_platform_neutral_wheel.py` | High — TD-009 boundary | **Keep** |
| No plaintext creds in gitops/ | `test_gitops_secret_guard.py` | High — security | **Keep** |
| `SealedSecret` kind in `*.sealed.yaml` | `test_gitops_secret_guard.py` | High — security | **Keep** |

### Not acceptable

- Sentence-level wording checks in markdown documentation files. Docs evolve;
  a test that asserts a specific sentence exists will fail on every legitimate
  docs update.
- Checks that verify a specific prose phrase in a commit message or PR description.
- Guards that duplicate what `ruff`, `mypy`, or Python import machinery already enforces.
- Tests that check presence of a specific comment string in source code.

### Current status

All existing phrase/wording guards fall into the "acceptable" category above.
The policy is: **do not add new phrase guards unless they meet the three criteria above**.

---

## Stale-wording guard consolidation (2026-05-24)

The repo reached architectural stability after TD-009 (workbench out of wheel),
TD-010 (AWS_* S3 Connection schema), and Level 3 pipeline milestone. At that
point several phrase-scanning tests in `test_pipeline_components.py` were
guarding old wording that no longer existed in executable code — they only
risked failing on legitimate historical comments.

### What was changed

**Removed (6 tests total — comment/docstring phrase guards):**

| Test | Reason |
|---|---|
| `TestWheelBasedLanguage.test_no_src_module_not_found_message` | Error phrase in old comment; no runtime risk |
| `TestWheelBasedLanguage.test_no_source_code_src_phrasing` | Comment wording only |
| `TestWheelBasedLanguage.test_no_src_baked_in_phrasing` | Comment wording only |
| `TestWheelBasedLanguage.test_no_scripts_train_pragma_as_runtime_entrypoint` | Specific comment string; entrypoint contract tested structurally |
| `TestPragmaPretrainingPipelineHonestContract.test_full_pipeline_docstring_no_skip_claim` | Docstring wording; no-`manifest_uri` contract tested by `test_full_pipeline_has_no_manifest_uri_param` |
| `TestBoundaryContracts.test_trainjob_not_implied_as_current_runtime` | Fragile pyproject.toml text scan; redundant with `test_trainjob_not_in_production_manifests` and `test_trainjob_marked_tech_preview` |

**Relaxed (2 tests — now check non-comment lines only):**

| Test | Before | After |
|---|---|---|
| `TestWheelBasedLanguage.test_no_src_workbench_path_reference` | Any line in source | Non-comment lines only via `_noncomment_lines()` |
| `TestWheelBasedLanguage.test_prepare_dataset_references_wheel_module_path` | Any line in source | Non-comment lines only via `_noncomment_lines()` |

The `_noncomment_lines()` helper strips blank lines and lines whose first
non-whitespace character is `#`, then joins the remainder. This means historical
mentions of old paths in comments are permitted; only executable code is checked.

### Before / after

| File | Before | After |
|---|---|---|
| `test_pipeline_components.py` | 75 | 70 |
| `test_openshift_ai_primitive_contract.py` | 67 | 66 |

### Principle applied

> **Test contracts, not vocabulary.**

A phrase-scanning test is justified only when the phrase itself is the contract
(e.g. `MODEL_REGISTRY_*` must not appear in any source file after TD-010 schema
migration). When the same contract is enforced by a structural test (parameter
absence, import check, YAML kind assertion), the phrase guard is redundant and
should be removed.

---

## Rules for adding new tests

### Tests that belong (add them)

| Criterion | Example |
|---|---|
| Paper equation or table value (TDD: tests from paper) | `assert config.d_model == 192  # Table 1, PRAGMA-S` |
| Fixed API contract (Protocol compliance) | `assert isinstance(result, PragmaRunProtocol)` |
| Regression guard for a Tech Debt fix | `test_all_ranks_download_checkpoint()` (TD-006) |
| Platform boundary or schema invariant | `MODEL_REGISTRY_*` absent after TD-010 |
| Security invariant | No plaintext creds in gitops/ |

### Tests that do not belong (do not add them)

| Criterion | Why not |
|---|---|
| Restates code already covered by existing tests | Redundant; adds maintenance without safety |
| Guards a prose phrasing expected to change | Fails on every docs update |
| Reimplements a check ruff/mypy already performs | Pick the right tool |
| Requires a cluster to answer a question answerable statically | Use Tier 2 instead |
| "Integration test" that mocks every dependency | Mock-heavy tests test the mocks, not the code |

### Where new tests go

- Paper architecture or training logic → Tier 1 (`tests/test_*.py`)
- Deployment contract, YAML fixture, or platform boundary → Tier 2 (`tests/test_*.py`)
- Live cluster verification → Tier 3 (`tests/openshift/test_*.py`)

---

## Quick reference — pytest commands

```bash
# Default CI run (Tier 1 + Tier 2, fast feedback)
pytest tests/ --ignore=tests/openshift/ -x -q

# With verbose output
pytest tests/ --ignore=tests/openshift/ -v

# Core unit tests only (skip slow packaging and Torch-heavy tests)
pytest tests/ --ignore=tests/openshift/ -x -q \
  -k "not packaging and not workbench and not checkpoint"

# All wording/guard tests only
pytest tests/ -k "stale or guard or contract or image or platform" -v

# Specific test file
pytest tests/test_checkpoint_resume.py -v

# Tier 3 — opt-in runtime (requires live cluster)
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<image> \
pytest tests/openshift/ -v

# Lint
ruff check src/ tests/ pipeline/
```

---

## Test count summary (as of 2026-05-24)

| Category | Files | Tests |
|---|---|---|
| Core unit tests (model, tokenizer, training math) | 21 | ~345 |
| Static platform contract (boundary, YAML, package) | 16 | ~540 |
| Security guards | 1 | 10 |
| Opt-in runtime (openshift/) | 11 | ~76 |
| **Total** | **~50** | **~970** |

Reference: `docs/development-process.md`, `CLAUDE.md §Development Process`
