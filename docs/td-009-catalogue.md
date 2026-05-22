# TD-009 Workbench Package Split — Catalogue

**Branch:** `fix/td-009-workbench-package-split`
**Date:** 2026-05-22

This document catalogues every file involved in extracting
`tools/openshift_ai/workbench/` into a proper second Python wheel
(`pragma-workbench`) at `src/pragma_workbench/`.

It is committed first, before any file is moved, as a navigable record
of intent.

---

## 1. Source files to move

All six modules under `tools/openshift_ai/workbench/` are moved verbatim
into `src/pragma_workbench/`. Only `__init__.py` needs an import-style
change (absolute → relative); the other five are unchanged.

| Source (current) | Destination | Change required |
|---|---|---|
| `tools/openshift_ai/workbench/__init__.py` | `src/pragma_workbench/__init__.py` | Absolute `from tools.openshift_ai.workbench._X import` → relative `from ._X import` |
| `tools/openshift_ai/workbench/_api.py` | `src/pragma_workbench/_api.py` | None — already uses relative imports for siblings, absolute for `pragma_encoder.*` |
| `tools/openshift_ai/workbench/_decorators.py` | `src/pragma_workbench/_decorators.py` | None — already uses relative imports |
| `tools/openshift_ai/workbench/_intent.py` | `src/pragma_workbench/_intent.py` | None — pure Python, no external imports |
| `tools/openshift_ai/workbench/_run.py` | `src/pragma_workbench/_run.py` | None — already uses absolute import of `pragma_encoder.data.dataset_manifest` |
| `tools/openshift_ai/workbench/_submit.py` | `src/pragma_workbench/_submit.py` | None — lazy kfp/kfp_kubernetes imports, no pragma_encoder import |

---

## 2. New files to create

| File | Purpose |
|---|---|
| `src/pragma_workbench/__init__.py` | Copy of workbench `__init__.py` with relative imports |
| `src/pragma_workbench/pyproject.toml` | Second wheel declaration (`pragma-workbench`, deps: `pragma-encoder`, `kfp`, `kfp-kubernetes`) |
| `src/pragma_workbench/py.typed` | PEP 561 marker (empty) |

---

## 3. Files to replace (deprecation shim)

After the move, the original `tools/openshift_ai/workbench/__init__.py` is
replaced with a thin shim that re-exports everything from `pragma_workbench`
and emits `DeprecationWarning` on import. The other five modules
(`_api.py`, `_decorators.py`, `_intent.py`, `_run.py`, `_submit.py`) remain
in place as compatibility forwarding stubs.

| File | Treatment |
|---|---|
| `tools/openshift_ai/workbench/__init__.py` | Replace with deprecation shim re-exporting from `pragma_workbench` |
| `tools/openshift_ai/workbench/_api.py` | Replace with forwarding stub: `from pragma_workbench._api import *` |
| `tools/openshift_ai/workbench/_decorators.py` | Replace with forwarding stub |
| `tools/openshift_ai/workbench/_intent.py` | Replace with forwarding stub |
| `tools/openshift_ai/workbench/_run.py` | Replace with forwarding stub |
| `tools/openshift_ai/workbench/_submit.py` | Replace with forwarding stub |

Documented as **TD-012** — shim removal in a future release.

---

## 4. Core wheel changes

| File | Change |
|---|---|
| `pyproject.toml` | Add `exclude = ["pragma_workbench*"]` to `[tool.setuptools.packages.find]` so `src/pragma_workbench/` is not bundled into the `pragma-encoder` wheel |

---

## 5. Import sites to update

Every occurrence of `tools.openshift_ai.workbench` in Python import
statements is replaced with `pragma_workbench`. Mock-patch strings and
`importlib.import_module()` calls are updated to match.

### 5a. Tests

| File | Lines (approx) | What changes |
|---|---|---|
| `tests/test_workbench_api.py` | 31–32 | `from tools.openshift_ai.workbench._api import` → `from pragma_workbench._api import`; mock-patch strings |
| `tests/test_workbench_decorators.py` | 12, 42–49 | `from tools.openshift_ai.workbench import` → `from pragma_workbench import`; path reads for source inspection |
| `tests/test_workbench_submit.py` | 45 | `from tools.openshift_ai.workbench._submit import`; `importlib.import_module("tools.openshift_ai.workbench._submit")` |
| `tests/test_pragma_run.py` | 26 | `from tools.openshift_ai.workbench._run import` |
| `tests/test_image_contract.py` | 330–381 | `from tools.openshift_ai.workbench._submit import`; mock-patch strings |
| `tests/test_pipeline_components.py` | 268, 1276 | `from tools.openshift_ai.workbench._run import PIPELINE_STEP_NAMES`; `importlib.import_module("tools.openshift_ai.workbench._decorators")` |
| `tests/test_platform_neutral_wheel.py` | multiple | `import tools.openshift_ai.workbench as wb`; assertions on importability — updated to assert `pragma_workbench` is importable; deprecation shim still re-exports correctly |
| `tests/test_openshift_ai_primitive_contract.py` | 880–901 | Location assertions updated to reference `pragma_workbench` |
| `tests/openshift/test_02_pipeline_compile.py` | 87, 111, 143, 198, 214 | `from tools.openshift_ai.workbench import dataset, pragma_pipeline, train` |
| `tests/openshift/test_03_pipeline_smoke_run.py` | 65, 132 | `from tools.openshift_ai.workbench._submit import` |

### 5b. Examples

| File | Change |
|---|---|
| `examples/workbench/01_train_ibm_tabformer.py` | `from tools.openshift_ai.workbench import` → `from pragma_workbench import` |
| `examples/workbench/02_understand_pipeline.py` | same |
| `examples/workbench/03_two_node_training_demo.py` | same |
| `examples/workbench/04_local_training_smoke.py` | same |
| `examples/workbench/05_decorated_pipeline.py` | same |

### 5c. Core source comments (no import change, comment update only)

| File | Change |
|---|---|
| `src/pragma_encoder/__init__.py` | Update comment referencing `tools/openshift_ai/workbench/` → `pragma_workbench` |

### 5d. Pipeline (comment/docstring only, no import change needed)

| File | Change |
|---|---|
| `pipeline/components_pragma.py` | Comments referencing `tools/openshift_ai/workbench/` updated if present |

---

## 6. Infrastructure files to update

| File | Change |
|---|---|
| `openshift/notebook-image/Dockerfile` | Add `pip install ./src/pragma_workbench/` (or from built wheel) so `pragma_workbench` is installed in the workbench image |
| `.github/workflows/ci.yml` | Add `pragma_workbench` wheel build; install in unit-test lane; optionally add separate `pragma-workbench` packaging lane |

---

## 7. Test invariant updates

| File | What changes |
|---|---|
| `tests/test_packaging.py` | Add `TestPragmaWorkbenchWheel` class: assert `pragma_workbench` is importable, assert `pragma_encoder.workbench` still raises `ModuleNotFoundError`, assert `tools.openshift_ai.workbench` emits `DeprecationWarning` |
| `tests/test_platform_neutral_wheel.py` | Update `TestWorkbenchRemovedFromWheel` to assert `pragma_workbench` (not `tools.openshift_ai.workbench`) is the primary import path; assert deprecation shim emits warning |

---

## 8. Documentation updates

| File | Change |
|---|---|
| `docs/tech-debt.md` | TD-009: update resolved description to reference `pragma_workbench` wheel; add TD-012 (deprecation shim removal) |
| `docs/openshift-ai-3.3-alignment.md` | §2 Workbench: update import example and package location; §Storage Adapter Boundary: remove stale `MODEL_REGISTRY_*` references (already superseded by TD-010) |
| `docs/decisions/003-workbench-training-api.md` | Add addendum: workbench helpers now live in `pragma_workbench` wheel |
| `docs/decisions/004-workbench-decorated-pipelines.md` | Add addendum: decorator DSL now in `pragma_workbench` |
| `docs/architecture.md` | Update "Workbench tooling" section from `tools/openshift_ai/workbench/` to `pragma_workbench` |
| `README.md` | If referenced, update workbench import path |

---

## 9. New ADR

| File | Content |
|---|---|
| `docs/decisions/007-pragma-workbench-wheel.md` | Documents the decision to extract workbench helpers into a separate installable wheel; rationale (tooling discoverability, proper dep declaration for kfp, wheel boundary enforcement); status: Accepted |

---

## 10. Constraint

**No files are moved from `tools/openshift_ai/workbench/` until this catalogue
commit is merged into the branch.**

This document is the single source of truth for the scope of this change.
If additional files are discovered during implementation, this catalogue must
be updated first.
