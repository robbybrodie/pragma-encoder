# OpenShift GitOps and ArgoCD — PRAGMA Encoder

This document describes what ArgoCD and OpenShift GitOps own in the
`pragma-encoder` project, with particular attention to the pipeline authoring
lifecycle and where each artifact belongs.

**Key principle:** ArgoCD is a promotion and control surface for declared
OpenShift AI/OpenShift resources — not a separate deployment path. The GUI,
Workbench SDK, ArgoCD, `oc`, CI/CD, and tests all interact with the same
underlying OpenShift AI primitives. ArgoCD promotes stable, long-lived platform
substrate into the cluster. The Workbench is the primary authoring surface for
pipeline code. OpenShift AI (RHOAI) is the runtime and GUI surface for pipeline
definitions, versions, and runs. These roles must not be conflated.

ArgoCD may own: namespace, RBAC, SealedSecrets, Connection templates,
ExternalSecrets, HardwareProfiles, DSPA, workbench manifests, BuildConfigs,
ImageStreams, and promoted compiled pipeline artifacts.

ArgoCD must not own: ad hoc PyTorchJob training runs, pipeline runs, or
exploratory notebook state.

See also:
- `docs/deployment.md` — bootstrap steps, sync wave order, ArgoCD prerequisites
- `docs/openshift-ai-3.3-alignment.md` — RHOAI 3.3 primitive map and boundaries

---

## ArgoCD Ownership Table

ArgoCD manages **long-lived, stable cluster resources** that form the platform
substrate. It does not manage exploratory or dynamic objects.

| Resource | ArgoCD owned? | Reason | Location in repo |
|---|---|---|---|
| Namespace (`pragma-encoder`) | Yes | Stable platform object | `openshift/gitops/namespace.yaml` |
| RBAC, ServiceAccounts | Yes | Stable platform object | `openshift/gitops/rbac/` |
| Sealed Secrets (S3 creds, registry pull) | Yes | Stable platform object | `openshift/gitops/secrets/` |
| Workbench image BuildConfig + ImageStream | Yes | Stable platform object | `openshift/gitops/notebook-image/` |
| DSPA (KFP pipeline server) | Yes | Stable platform runtime | `openshift/gitops/pipeline/dspa.yaml` |
| Workbench Notebook pod | Yes | Stable platform object | `openshift/gitops/workbench/notebook.yaml` |
| Hardware Profile definitions | No (managed by RHOAI admin) | Platform admin object | `tests/openshift/fixtures/` (examples only) |
| **Pipeline Python source** | No (Git-owned, not a cluster resource) | Human-authored decorator code | `pipeline/*.py` |
| **Compiled pipeline IR YAML** | Optionally, when promoted | Stable promoted artifact | `pipeline/generated/` or `openshift/gitops/pipelines/` |
| **Pipeline definition/version registration** | Optional | Team choice — see below | `openshift/gitops/pipelines/` (if used) |
| **Pipeline run** | **No** | Dynamic execution object | OpenShift AI runtime (GUI/API) |
| **Exploratory notebook work** | **No** | Exploratory user workspace | Workbench storage or `examples/` |
| PyTorchJob (training job) | **No** | One-shot dynamic job | Launched by pipeline or directly |

---

## Workbench-Authored Pipelines and GitOps Promotion

### The authoring lifecycle

OpenShift AI pipelines in this project are **not hand-written declarative YAML**.
They are authored as Python using KFP SDK v2 decorators and components, then
compiled to IR YAML for execution on the DSPA runtime.

The lifecycle has these stages:

```
1. Author (Workbench)
   ─────────────────
   Data scientist opens an OpenShift AI Workbench (Jupyter environment).
   Pipeline logic is written as decorated Python functions using KFP SDK v2:

     @dsl.component(base_image=TRAINING_IMAGE)
     def prepare_dataset(dataset_name: str, model_size: str) -> str:
         ...

     @dsl.pipeline(name="pragma-pretraining")
     def pragma_pretraining_pipeline(...):
         ...

   Or via the @pragma_pipeline decorator DSL (ADR 004):

     @pragma_pipeline(name="pragma-s-ibm-tabformer")
     def run():
         ds = dataset("ibm-tabformer", prepare_if_missing=True)
         train(dataset=ds, model_size="S", epochs=10)

   Pipeline source lives in:
     pipeline/components_pragma.py        — five §2.4 stage components
     pipeline/pragma_pipeline.py          — full and manifest-bypass pipelines
     examples/workbench/05_decorated_pipeline.py — decorator DSL example

2. Explore / iterate (Workbench or KFP SDK)
   ─────────────────────────────────────────
   Exploratory pipeline runs are launched from:
     - The RHOAI dashboard (GUI → "Create run")
     - The KFP SDK client in the Workbench: kfp.Client().create_run_from_pipeline_func(...)
     - The registration notebook: pipeline/00_register_pipeline.ipynb

   These exploratory runs are NOT owned by ArgoCD.
   They are transient execution objects in the OpenShift AI runtime.

3. Compile (Workbench or CI)
   ──────────────────────────
   Once the pipeline is ready for promotion, compile it to KFP v2 IR YAML:

     from kfp import compiler
     compiler.Compiler().compile(
         pipeline_func=pragma_pretraining_pipeline,
         package_path="pipeline/generated/pragma-pretraining-pipeline.yaml",
     )

   Or via the @pragma_pipeline decorator DSL:

     run.compile("pipeline/generated/pragma-s-ibm-tabformer.yaml")

   Compiled YAML is deposited in pipeline/generated/ (gitignored by default)
   or in openshift/gitops/pipelines/ if being promoted to GitOps.

   CI validates pipeline compilability (without cluster access):
     pytest tests/test_pipeline_components.py  (skips if kfp not installed)
     pytest tests/test_smoke_pipeline.py       (skips if kfp not installed)

4. Promote (Git PR)
   ─────────────────
   If the team wants GitOps-controlled pipeline registration:
     - Commit the compiled IR YAML to openshift/gitops/pipelines/
     - Add a pipeline registration resource (future, team decision)
     - ArgoCD syncs the committed artifact to the cluster

   This is OPTIONAL. Many teams register pipelines directly from the
   Workbench or CI without GitOps-managed registration.

5. Register and run (OpenShift AI GUI or KFP API)
   ─────────────────────────────────────────────────
   OpenShift AI displays pipeline definitions, versions, and runs in the GUI.
   Runs are launched from the GUI or KFP API — not from ArgoCD.
   The Workbench remains the interactive surface for iterating on pipeline logic.
```

### What ArgoCD owns in the pipeline layer

ArgoCD currently manages **the DSPA** — the KFP pipeline server — via:

```
openshift/gitops/pipeline/dspa.yaml
```

This is the wave-3 resource in the sync order (see `docs/deployment.md`).

The DSPA is the stable platform substrate that makes the KFP v2 API available
in the namespace. It is correctly managed by ArgoCD because it is a long-lived
infrastructure object, not an exploratory pipeline artifact.

ArgoCD does **not** currently manage:
- Pipeline Python source (that lives in `pipeline/` as application code)
- Compiled pipeline IR YAML (belongs in `pipeline/generated/` until promoted)
- Pipeline definitions or versions registered with the DSPA
- Pipeline runs (always dynamic, always owned by RHOAI runtime)

### Promoting compiled artifacts (future path)

When the team decides to GitOps-manage pipeline registration, the following
directory can be used:

```
openshift/gitops/pipelines/
  pragma-pretraining-pipeline.yaml     — compiled IR YAML for the full pipeline
  pragma-from-manifest-pipeline.yaml   — compiled IR YAML for manifest-bypass pipeline
  pragma-smoke-pipeline.yaml           — compiled IR YAML for smoke/test pipeline
```

ArgoCD would sync these artifacts and a registration resource (e.g. a Job or
pipeline registration script) would upload them to the DSPA API.

**This path is not yet implemented.** The current registration mechanism is
the Workbench notebook at `pipeline/00_register_pipeline.ipynb`.

---

## Pipeline Source vs Compiled Artifact vs Run

This is the critical distinction to keep clear:

| Artifact | What it is | Who authors it | Where it lives | ArgoCD managed? |
|---|---|---|---|---|
| Pipeline Python source | KFP decorator-authored Python | Data scientist (Workbench) | `pipeline/*.py` | No — Git-only |
| Compiled IR YAML | KFP SDK output from `compiler.Compiler().compile()` | CI or Workbench | `pipeline/generated/` → `openshift/gitops/pipelines/` when promoted | Optional, when promoted |
| Pipeline definition/version | Registered with DSPA via API | Registration notebook or CI | DSPA (cluster state) | Optional, via registration resource |
| Pipeline run | Execution instance with parameters | RHOAI GUI or KFP API call | RHOAI runtime (cluster) | No — always dynamic |

**Key rule:** A pipeline run is never a GitOps-managed resource. Runs are
ephemeral execution objects created by users or automation at runtime. Committing
or syncing pipeline runs via ArgoCD would be incorrect.

---

## What Remains in the Workbench

The Workbench is not only for notebook exploration. It is the **primary
authoring surface** for:

- Iterating on pipeline component code
- Testing pipeline compilation locally (without submitting to DSPA)
- Inspecting pipeline structure with `show_pipeline()`
- Submitting exploratory pipeline runs against the DSPA
- Registering pipeline versions manually
- Validating the pipeline before promoting to Git for GitOps management

`pipeline/00_register_pipeline.ipynb` documents the Workbench-side registration
workflow. This notebook is committed to the repo but is not executed by ArgoCD.

---

## Compile Tests — No Cluster Required

Pipeline compilation is validated in CI without cluster access.

The compile test in `tests/test_pipeline_components.py`:
```
TestKfpOptionalExecution::test_pipeline_function_compilable_when_kfp_available
TestKfpOptionalExecution::test_manifest_pipeline_compilable_when_kfp_available
```

And in `tests/test_smoke_pipeline.py`:
```
TestSmokePipelineCompile::test_smoke_pipeline_compiles_to_yaml
```

All compile tests use `pytest.importorskip("kfp")` or `find_spec("kfp")` to
skip cleanly when kfp is not installed. This keeps default CI cluster-free.

`kfp` is not a `pragma_encoder` wheel dependency. It is installed into the
workbench image via `openshift/notebook-image/requirements.txt`. To run compile
tests locally, install it directly:
```bash
pip install kfp
```

---

## repo Structure for Pipeline Artifacts

```
pipeline/
  components_pragma.py       KFP v2 components (decorator-authored, five §2.4 stages)
  pragma_pipeline.py         Full and manifest-bypass pipeline definitions
  pragma_smoke_pipeline.py   Minimal smoke pipeline for Level 3 DSPA tests
  00_register_pipeline.ipynb Workbench notebook for pipeline registration
  generated/                 Landing zone for compiled IR YAML
    .gitkeep                 (placeholder — compiled YAML goes here pre-promotion)
    *.yaml                   compiled KFP v2 IR — generated by compiler.Compiler()
                             gitignored by default; commit only when promoting

openshift/gitops/
  pipeline/
    dspa.yaml                DSPA (KFP pipeline server) — wave 3, ArgoCD managed
  pipelines/                 (future — if GitOps-managed pipeline registration adopted)
    *.yaml                   promoted compiled IR YAML + registration resources

tests/
  test_pipeline_components.py  static + compile tests for production pipelines
  test_smoke_pipeline.py       static + compile tests for smoke pipeline

tests/openshift/
  test_02_pipeline_compile.py  static pipeline compile check (no cluster)
  test_03_pipeline_smoke_run.py Level 3 DSPA connectivity + smoke run (opt-in)
  test_03_dspa_runtime_discovery.py Level 3 DSPA endpoint probe (opt-in)

examples/workbench/
  05_decorated_pipeline.py   @pragma_pipeline decorator DSL demo
  07_dspa_client_probe.py    DSPA connection probe from Workbench
```

---

## ArgoCD Wave Ownership (Reference)

Reproduced from `docs/deployment.md` with pipeline layer clarification:

| Wave | Resources | ArgoCD owned? | Notes |
|---|---|---|---|
| -1 | `namespace.yaml` | Yes | Platform substrate |
| 0 | RBAC, Sealed Secrets | Yes | Platform substrate |
| 1 | `imagestream.yaml` | Yes | Platform substrate |
| 2 | `buildconfig.yaml` | Yes | Triggers workbench image build |
| 3 | `dspa.yaml` | Yes | KFP pipeline **server** — not pipeline definitions |
| 4 | `notebook.yaml` | Yes | Workbench pod |
| — | Pipeline Python source | No | Git-owned application code |
| — | Compiled pipeline IR YAML | Optional | In `pipeline/generated/` until promoted |
| — | Pipeline definitions/versions | Optional | Registered with DSPA by Workbench/CI |
| — | Pipeline runs | **No** | Dynamic; owned by RHOAI runtime |
| — | PyTorchJob (training runs) | **No** | One-shot; launched by pipeline or directly |

---

## Summary

| Question | Answer |
|---|---|
| Who authors pipeline code? | Data scientist in the Workbench using KFP decorators |
| Is pipeline YAML hand-written? | No — it is compiled from Python by KFP SDK |
| Where does pipeline Python live? | `pipeline/*.py` (Git, not a cluster resource) |
| Where does compiled IR YAML go? | `pipeline/generated/` (pre-promotion) or `openshift/gitops/pipelines/` (promoted) |
| Does ArgoCD own pipeline runs? | No — runs are dynamic RHOAI runtime objects |
| Does ArgoCD own the pipeline server? | Yes — `dspa.yaml` (the DSPA/KFP server) is ArgoCD-managed |
| Where are pipeline definitions/versions displayed? | OpenShift AI GUI — the canonical interface for users |
| Can we validate pipeline compilation in CI? | Yes — compile tests skip cleanly without kfp or cluster |
