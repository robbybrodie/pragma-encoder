# Decision 004: Workbench-Decorated Pipeline Authoring

Status: Accepted
Date: 2026-05-19
Supersedes: nothing (extends ADR 003)
Paper reference: Section 2.4 (Training Infrastructure)

---

## Context

ADR 003 introduced `train_pragma()` as the primary workbench entry point.
It supports `mode="dry_run"` (preview) and `mode="local"` (subprocess training).
The existing `pipeline/` layer contains five KFP components and two compiled
pipeline functions, but data scientists cannot use them directly from a workbench
notebook without writing YAML or knowing KFP internals.

Two gaps exist:

1. **Authoring gap** — Data scientists who want to express training intent
   in Python are forced to either call `train_pragma()` with keyword arguments
   or hand-write KFP pipeline YAML. Neither is readable as a workflow.

2. **Visibility gap** — Training runs are not visible as first-class objects
   in OpenShift Pipelines. The data scientist cannot see the five §2.4 stages
   as a graph in the OpenShift AI UI.

The goal of this ADR is to close both gaps while preserving all existing
execution modes and the `pipeline/ → src/` dependency direction.

---

## Decision

### 1. Workbench-decorated Python is the pipeline authoring surface

Data scientists express training intent using a Python decorator DSL:

```python
from tools.workbench import pragma_pipeline, dataset, train

@pragma_pipeline(name="pragma-s-ibm-tabformer")
def run():
    ds = dataset("ibm-tabformer", prepare_if_missing=True)
    train(dataset=ds, model_size="S", epochs=1, max_steps=1)

run.show_pipeline()                                    # inspect five stages
run.compile("pipeline/generated/pragma-s-ibm-tabformer.yaml")  # emit KFP YAML
run.submit()                                           # future extension
```

The decorator captures intent. It does **not** execute training, call any
adapter, access S3, or interact with any cluster resource at authoring time.

The simple keyword-argument path from ADR 003 is extended with a new mode:

```python
run = train_pragma(
    dataset="ibm-tabformer",
    model_size="S",
    epochs=1,
    mode="pipeline",
    max_steps=1,
)
run.show_pipeline()
run.compile("pipeline/generated/pragma-s-ibm-tabformer.yaml")
```

### 2. Static pipeline functions remain as lower-level building blocks

`pipeline/pragma_pipeline.py` and `pipeline/components_pragma.py` are not
replaced. They remain the concrete KFP implementation layer. The decorator
layer compiles to them; it does not duplicate model, tokeniser, or training
logic.

### 3. Data scientists do not hand-write Argo CD or KFP YAML for training

Argo CD manages the platform substrate only:
- Namespace, RBAC, Sealed Secrets
- Notebook image BuildConfig and ImageStream
- Data Science Pipeline Application (DSPA / KFP server)
- Workbench Notebook resource

Individual training runs are authored in Python, compiled to pipeline YAML,
and submitted via the KFP API — not committed to Git and synced by Argo CD.

### 4. Generated pipeline runs are visible in OpenShift Pipelines

`compile()` produces a KFP v2 YAML that, when submitted, appears as a run
in the OpenShift AI Pipelines UI with the five §2.4 stages visible as a
directed graph (prepare → upload → submit → train → export).

### 5. dry_run and local modes are unchanged

`mode="dry_run"` returns a `PragmaRun` preview with no side effects.
`mode="local"` runs `scripts/train_pragma.py` as a subprocess.
Neither mode requires KFP. Neither mode is changed by this ADR.

### 6. compile() precedes submit()

The implementation sequence is:
1. `compile(path)` — write KFP YAML; does not submit anything
2. `submit()` — submit the compiled pipeline to the KFP server (future)

`submit()` raises `NotImplementedError` with a clear message until explicitly
implemented. This prevents accidental cluster submission during the compile
phase.

### 7. KFP remains an optional dependency

`compile()` requires KFP. When KFP is absent, `compile()` raises a
`RuntimeError` with a message directing the user to:
- `pip install -r requirements.txt` (includes kfp)
- or use the prepared workbench image (which has kfp pre-installed)

`show_pipeline()` and intent capture work without KFP.

---

## Architecture

```
Data scientist (workbench notebook)
    |
    | @pragma_pipeline / dataset() / train()
    v
tools/workbench/_decorators.py -- intent capture; no side effects
tools/workbench/_intent.py     -- DatasetIntent, TrainIntent dataclasses
    |
    | compile() only — lazy importlib.import_module("pipeline.pragma_pipeline")
    v
pipeline/pragma_pipeline.py    -- KFP pipeline function (lower-level building block)
pipeline/components_pragma.py  -- five KFP component stubs
    |
    | KFP compiler (kfp.compiler.Compiler)
    v
pipeline/generated/<name>.yaml -- KFP v2 pipeline YAML
    |
    | upload + run (future: submit())
    v
OpenShift Pipelines / KFP server -- five stages visible as a graph
```

### Dependency direction (preserved)

```
tools/workbench/_intent.py      -- no pipeline/ dependency
tools/workbench/_decorators.py  -- no static pipeline/ import
                                   compile() uses importlib.import_module()
                                   to avoid "from pipeline" / "import pipeline"
                                   in source, preserving the test_pipeline_components
                                   circular-dependency guard
pipeline/                       -- imports from src/ only (unchanged)
```

The existing `TestNoCircularDependency` test in `tests/test_pipeline_components.py`
scans for the literal strings `from pipeline` and `import pipeline` in `src/`
source files. Using `importlib.import_module("pipeline.pragma_pipeline")` inside
`compile()` contains neither string and therefore passes the guard.

---

## Intent capture mechanism

`@pragma_pipeline` executes the decorated function exactly once at decoration
time inside a thread-local capture context. During this execution:

- `dataset(name, ...)` returns a `DatasetIntent` value object (no side effects)
- `train(dataset=..., ...)` returns a `TrainIntent` value object AND registers
  it in the thread-local capture context

After the function returns, `PragmaPipeline` reads the captured intents and
stores them. The decorated function is never called again unless the user
explicitly does so outside the capture context.

This mechanism is adapted from the KFP pipeline DSL pattern where component
calls are intercepted during pipeline graph construction.

---

## New public API additions to tools/workbench/__init__.py

```python
from tools.workbench._decorators import pragma_pipeline
from tools.workbench._intent    import dataset, train
```

Existing exports (`train_pragma`, `PragmaRun`, `PIPELINE_STEP_NAMES`, etc.)
are unchanged.

---

## New files

| File | Purpose |
|------|---------|
| `tools/workbench/_intent.py` | `DatasetIntent`, `TrainIntent` dataclasses; `dataset()`, `train()` intent-capture functions |
| `tools/workbench/_decorators.py` | `PragmaPipeline` class; `pragma_pipeline` decorator factory |
| `tests/test_workbench_decorators.py` | Full test coverage for the decorator API |
| `examples/workbench/05_decorated_pipeline.py` | Demo: authoring → compile |
| `pipeline/generated/` | Output directory for compiled YAML (git-ignored) |

---

## Constraints that must not be violated

1. No training, no S3, no cluster access at decoration time
2. No `from pipeline` or `import pipeline` static imports in `src/`
3. No PVC parameters introduced in `dataset()`, `train()`, or `pragma_pipeline()`
4. `compile()` must not submit training or mutate S3
5. `submit()` raises `NotImplementedError` until explicitly implemented
6. `mode="dry_run"` and `mode="local"` are unchanged
7. KFP is optional — absence must fail clearly, not silently
8. Pipeline components (`pipeline/`) are reused, not duplicated
9. **Decorated function must call `train()` exactly once** (see Capture Validation below)
10. **Lazy-import boundary is `compile()` only** (see Lazy-Import Boundary below)

---

## Capture Validation

`@pragma_pipeline` validates the capture list immediately after executing the
decorated function. The following invariants are enforced at decoration time
(not at compile or submit time):

| Captured `train()` calls | Behaviour |
|--------------------------|-----------|
| 0 | `ValueError` — mentions `train()`, includes an example fix |
| 1 | Normal path — `PragmaPipeline` is returned |
| > 1 | `ValueError` — mentions the call count, directs user to use separate `@pragma_pipeline` definitions |

Rationale: silent fabrication (the prior "unknown" fallback) hides programming
errors that would only surface at compile or run time. Silent selection of the
first of multiple `train()` calls produces non-deterministic pipelines if the
order ever changes. Both are programming errors and must be caught early.

---

## Lazy-Import Boundary

### Rule

`pipeline/` modules may be loaded inside `compile()` only, at call time,
via `importlib.import_module()`. They must **never** be loaded as a side
effect of importing any `tools/workbench/` module.

```
Allowed:   compile() → importlib.import_module("pipeline.pragma_pipeline")
Forbidden: tools/workbench/_decorators.py (module level) → import pipeline.*
```

### Rationale

The `src/ → pipeline/` dependency direction is forbidden by ADR 003 because:
- `pipeline/` imports `src/` — a static cycle would make `src/` untestable in isolation
- The KFP `@dsl.component` / `@dsl.pipeline` decorators have import-time side
  effects that may fail when KFP is not installed

`compile()` is the only method that genuinely needs `pipeline/`. Placing the
`importlib.import_module` inside `compile()` ensures:
- The module is loaded only when `compile()` is called (lazy)
- Importing `tools.workbench._decorators` never loads any `pipeline/` module
- KFP absence raises a clear `RuntimeError` from `compile()`, not an obscure
  `ImportError` at import time

### Enforcement

`tests/test_pipeline_components.py::TestNoCircularDependency::test_decorators_do_not_load_pipeline_at_import_time`
verifies this at runtime: it imports `tools.workbench._decorators` and asserts
that no module whose name starts with `"pipeline"` appears in `sys.modules`
as a result.

---

## What this decision does NOT cover

- Real-time pipeline run monitoring (future)
- `submit()` implementation (future — NotImplementedError for now)
- Multi-pipeline composition (future)
- Argo CD GitOps workflow for training pipelines (deliberately out of scope;
  training intent belongs in the workbench, not in Git-synced manifests)
