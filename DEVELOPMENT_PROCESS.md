# Development Process

This document defines the mandatory development process
for all code in this repository.

The PRAGMA architecture is complex and subtle.
AI-generated ML code fails silently — the model trains,
loss decreases, numbers look plausible, but the architecture
is wrong. The only protection against this is rigorous
test-driven development derived from the paper specification.

---

## The Five Steps — Mandatory for Every Component

### Step 1 — Read the paper section

Before writing any code:
- Identify the exact paper section for the component
- Read it carefully
- Extract the mathematical specifications:
  - Input and output shapes
  - Architectural parameters (layers, heads, dimensions)
  - Mathematical properties (attention patterns, masking rates)
  - Expected behaviours (gradient flow, aggregation tokens)
- Reference the section number in every test and docstring

Paper: "PRAGMA: Revolut Foundation Model"
arXiv:2604.08649v1
Authors: Ostroukhov et al., Revolut Research + NVIDIA

### Step 2 — Write the tests first

Create tests/test_<component>.py BEFORE any implementation.

Every test file must contain these four test types:

#### 2a. Shape test
Verifies that input shapes produce correct output shapes.
Fast. Catches structural bugs immediately.

Example:
  def test_profile_state_encoder_output_shape():
      """Section 2.3.2: Profile State Encoder outputs [USR] token.

      Input:  (batch, n_profile_tokens, d_model)
      Output: (batch, 1, d_model) — only [USR] token returned
      """
      config = PRAGMAConfig.pragma_s()
      encoder = ProfileStateEncoder(config)

      batch, n_tokens, d_model = 2, 10, config.d_model
      x = torch.randn(batch, n_tokens, d_model)
      t = torch.zeros(batch, n_tokens)  # temporal coords

      out = encoder(x, t)

      assert out.shape == (batch, 1, d_model), \
          f"Expected ({batch}, 1, {d_model}), got {out.shape}"

#### 2b. Mathematical property test
Verifies the mathematical behaviour specified in the paper.
Tests what the component does, not just its shape.

Example:
  def test_rope_temporal_decay():
      """Section 2.2: RoPE encodes temporal coordinates.

      The dot product between two position encodings should
      depend only on their relative temporal distance.
      Closer events should have higher dot product similarity.
      """
      rope = RoPEEncoding(d_model=64)

      t_near  = torch.tensor([0.0, 1.0])   # 1 second apart
      t_far   = torch.tensor([0.0, 100.0]) # 100 seconds apart

      q = torch.randn(1, 2, 64)

      q_near = rope(q, t_near)
      q_far  = rope(q, t_far)

      sim_near = torch.dot(q_near[0,0], q_near[0,1]).item()
      sim_far  = torch.dot(q_far[0,0],  q_far[0,1]).item()

      assert sim_near > sim_far, \
          "RoPE should produce higher similarity for closer events"

#### 2c. Gradient flow test
Verifies that all parameters receive gradients during backprop.
Catches frozen parameters, detached tensors, wrong loss computation.

Example:
  def test_profile_state_encoder_gradients():
      """All parameters must receive gradients.

      A frozen or detached parameter means that part of the
      model is not being trained. This fails silently without
      this test.
      """
      config = PRAGMAConfig.pragma_s()
      encoder = ProfileStateEncoder(config)

      x = torch.randn(2, 10, config.d_model)
      t = torch.zeros(2, 10)

      out = encoder(x, t)
      loss = out.sum()
      loss.backward()

      for name, param in encoder.named_parameters():
          assert param.grad is not None, \
              f"Parameter '{name}' has no gradient — " \
              f"it is not being trained"
          assert not torch.all(param.grad == 0), \
              f"Parameter '{name}' has all-zero gradients"

#### 2d. Paper specification test
Verifies the component matches the exact values in the paper.
Tests parameter counts, vocabulary sizes, layer counts, etc.

Example:
  def test_pragma_s_matches_table_1():
      """Section 2.3, Table 1: PRAGMA-S architectural parameters.

      From Table 1:
        Params:  10M
        d_model: 192
        d_ffn:   768
        Profile encoder layers: 1
        Event encoder layers:   5
        History encoder layers: 2
        Attention heads:        3
      """
      config = PRAGMAConfig.pragma_s()

      assert config.d_model == 192,             f"d_model should be 192, got {config.d_model}"
      assert config.d_ffn == 768,               f"d_ffn should be 768, got {config.d_ffn}"
      assert config.profile_encoder_layers == 1, f"profile layers should be 1"
      assert config.event_encoder_layers == 5,   f"event layers should be 5"
      assert config.history_encoder_layers == 2, f"history layers should be 2"
      assert config.n_heads == 3,               f"n_heads should be 3"

      model  = PRAGMA(config)
      n_params = sum(p.numel() for p in model.parameters())

      assert 9_000_000 < n_params < 11_000_000, \
          f"PRAGMA-S should have ~10M params, got {n_params:,}"

### Step 3 — Show the tests for review

After writing the tests and BEFORE writing any implementation:

Present a summary table:

| Test | Type | Paper Section | Tests What |
|------|------|---------------|------------|
| test_X_shape | Shape | 2.3.X | Input/output dimensions |
| test_X_property | Math | 2.3.X | [specific property] |
| test_X_gradients | Gradient | 2.3.X | All params trained |
| test_X_spec | Paper spec | Table 1 | [specific values] |

Then STOP. Wait for explicit approval before proceeding.

Do not write implementation code until the human says:
"Tests approved. Implement."

If the human asks to change a test — that is acceptable.
If the implementation causes you to want to change a test —
that is NOT acceptable. Fix the implementation instead.

### Step 4 — Implement to pass the tests

Only after test approval:

Write the implementation.

Run the tests immediately:
  pytest tests/test_<component>.py -v

If tests pass: commit both tests and implementation together.
If tests fail: fix the implementation. Never fix the test.

The test represents the paper specification.
The implementation must match the paper.
If there is a conflict, the paper wins.

### Step 5 — Run the full test suite

After every implementation:

pytest tests/ -v --tb=short

All previously passing tests must still pass.
No regressions.

If a new implementation breaks an existing test:
fix the new implementation, not the existing test.

---

## The Comparison Test — The Final Arbiter

Beyond unit tests, there is one end-to-end test that
determines whether the implementation is fundamentally correct:

PRAGMA embeddings must outperform NVIDIA blueprint embeddings
on the TabFormer fraud detection task after reasonable training.

Expected result after PRAGMA-S training on TabFormer:
  PRAGMA-S embedding AUC > NVIDIA blueprint embedding AUC

If this does not hold after 1,000+ training steps:
  Something is architecturally wrong.
  Unit tests are necessary but not sufficient.
  Investigate the architecture against the paper.

This comparison test must be run:
  After first complete implementation
  After any architectural change
  Before any demo or presentation

Reference:
  Adjacent repo embeddings:
    ../transaction-foundation-model-openshiftai/
  TabFormer dataset: shared between both repos
  Evaluation code: scripts/evaluate_pragma.py

---

## What Never To Do

These are absolute rules. No exceptions.

1. Never fix a test to match the implementation.
   The test represents the paper specification.
   The paper is the ground truth.

2. Never skip the tests-first step.
   Even for "simple" components.
   Even when under time pressure.
   The bugs that hurt most are in the "simple" components.

3. Never commit implementation without tests.
   Every src/ module must have a corresponding tests/ file.
   No untested code reaches main branch.

4. Never ignore a failing test.
   A failing test is information.
   It means the implementation does not match the paper.
   Fix the implementation.

5. Never generate tests from the implementation.
   Tests must come from the paper specification.
   If you read the code first and then write tests,
   you will test what the code does, not what it should do.
   Read the paper section. Write the test. Then read the code.

---

## Component Implementation Order

Implement in this order. Each component depends on the previous.

1. PRAGMAConfig (src/model/config.py)
   Paper: Table 1
   Tests: parameter values, all three model sizes
   No dependencies.

2. Tokeniser (src/tokenizer/)
   Paper: Section 2.2
   Tests: numerical buckets, categorical tokens, BPE text,
          temporal encoding, vocabulary sizes
   Depends on: config

3. RoPE encoding (src/encoders/rope.py)
   Paper: Section 2.3.2, Su et al. (2024)
   Tests: rotation properties, temporal decay, relative positions
   Depends on: config

4. Profile State Encoder (src/encoders/profile_state_encoder.py)
   Paper: Section 2.3.2
   Tests: shape, [USR] token, RoPE integration, gradients
   Depends on: config, RoPE

5. Event Encoder (src/encoders/event_encoder.py)
   Paper: Section 2.3.3
   Tests: shape, [EVT] token, calendar features, gradients
   Depends on: config

6. History Encoder (src/encoders/history_encoder.py)
   Paper: Section 2.3.4
   Tests: shape, fusion of profile + events, RoPE, gradients
   Depends on: Profile State Encoder, Event Encoder, RoPE

7. Masking strategy (src/masking/strategy.py)
   Paper: Section 2.3.5
   Tests: 15% token masking, 10% event masking,
          10% semantic-type masking, UNK replacement
   Depends on: tokeniser

8. MLM head (src/model/mlm_head.py)
   Paper: Section 2.3.5
   Tests: 3d input fusion, logit shape, label smoothing
   Depends on: all encoders

9. Full PRAGMA model (src/model/pragma.py)
   Paper: Section 2.3
   Tests: parameter count vs Table 1, end-to-end shape,
          gradient flow through all components
   Depends on: all of the above

10. LoRA fine-tuning (src/adaptation/lora.py)
    Paper: Section 3.1.2
    Tests: 2-4% parameter update fraction, QKV targets,
           frozen backbone, rank=8 alpha=8
    Depends on: full model

11. Embedding probe (src/adaptation/probe.py)
    Paper: Section 3.1.1
    Tests: linear probe shape, frozen embeddings,
           L-BFGS convergence
    Depends on: full model

---

## Test File Template

Every tests/test_<component>.py must start with:

  """Tests for <component>.

  Derived from PRAGMA paper Section X.X:
  "PRAGMA: Revolut Foundation Model"
  Ostroukhov et al. (2026), arXiv:2604.08649v1

  Test types:
    Shape tests:      verify input/output dimensions
    Math tests:       verify mathematical properties from paper
    Gradient tests:   verify all parameters receive gradients
    Spec tests:       verify values match paper exactly
  """

  import pytest
  import torch
  import numpy as np
  from src.model.config import PRAGMAConfig

Then the four test classes:
  class TestShapes:
  class TestMathProperties:
  class TestGradientFlow:
  class TestPaperSpecifications:

Each class contains the relevant tests for that component.

---

## Commit Message Format

Every commit that adds tests and implementation must use:

  git commit -m "Implement <component> with tests (Section X.X)

  Tests derived from paper Section X.X:
  - test_<component>_shape: [what it tests]
  - test_<component>_property: [what property]
  - test_<component>_gradients: [what it verifies]
  - test_<component>_spec: [what paper values]

  All tests pass. No regressions in existing test suite.

  Reference: Ostroukhov et al. (2026), Section X.X"

---

## When Claude Code Is Implementing

The following instructions apply when Claude Code is
generating implementation code for this repository:

BEFORE writing any implementation code for a component:

1. State which paper section you are implementing.
   Quote the relevant passage from the paper.

2. Write the test file first.
   Present the four test types in a summary table.
   STOP and wait for explicit human approval.
   Do not proceed to implementation without "Tests approved."

3. After approval, implement to pass the tests.
   Run pytest immediately after implementation.
   Report all test results.

4. If any test fails:
   Analyse why the implementation does not match the paper.
   Fix the implementation.
   Do not modify the test unless the human explicitly
   approves a test change with a reason.

5. After all tests pass:
   Run the full test suite: pytest tests/ -v
   Report any regressions.
   Only commit if all tests pass.

This process cannot be shortened.
There are no exceptions.
The PRAGMA architecture is subtle.
The tests protect against silent implementation errors
that would only be discovered when the model fails
to outperform the NVIDIA blueprint at evaluation time —
weeks of wasted training compute.

---

## Interface Contracts — Before Any Implementation

Before implementing ANY component, define its interface
as a Python Protocol in the relevant src/ __init__.py.

The Protocol defines:
  - Exact method signatures
  - Tensor shapes as comments on every argument
  - Return type and shape

Example for ProfileStateEncoder:

  from typing import Protocol
  import torch

  class ProfileStateEncoderProtocol(Protocol):
      def forward(
          self,
          x: torch.Tensor,  # (batch, n_profile_tokens, d_model)
          t: torch.Tensor,  # (batch, n_profile_tokens) log-seconds
      ) -> torch.Tensor:    # (batch, 1, d_model) — [USR] token only
          ...

Rules:
  - Define the Protocol BEFORE writing the implementation
  - Review the Protocol BEFORE writing the implementation
  - The Protocol is fixed unless there is a paper-derived
    reason to change it
  - The test file imports the Protocol and validates against it
  - Changing an interface after implementation causes refactors
    Define it right once. Do not change it.

---

## Naming Conventions — Fixed, Not Negotiable

These names match the PRAGMA paper notation exactly.
When the paper uses za, we use za.
Never invent alternative names.

### Tensor variables (match paper equations exactly)
  x       input token embeddings
  t       temporal coordinates
  z       encoder output embeddings
  za      profile state encoder output — the [USR] token
  ze      event encoder output — the [EVT] tokens
  zh      history encoder final output
  zbe     event encoder token-level output (pre-[EVT] pooling)
  mask    attention mask
  k       key embedding
  v       value embedding

### Dimension variables
  batch           batch size
  n_events        number of events in the history sequence
  n_tokens        number of tokens per event
  na              number of profile state tokens
  ne              number of events
  d_model         model hidden dimension
  d_ffn           feed-forward layer dimension
  n_heads         number of attention heads

### Class names (match paper section names exactly)
  PRAGMAConfig              configuration dataclass
  PRAGMA                    full model (Section 2.3)
  ProfileStateEncoder       paper Section 2.3.2
  EventEncoder              paper Section 2.3.3
  HistoryEncoder            paper Section 2.3.4
  PRAGMATokenizer           paper Section 2.2
  MaskingStrategy           paper Section 2.3.5
  MLMHead                   masked modelling head
  LoRAAdapter               paper Section 3.1.2
  EmbeddingProbe            paper Section 3.1.1

### File names (match class names exactly)
  src/model/config.py                   → PRAGMAConfig
  src/model/pragma.py                   → PRAGMA
  src/model/mlm_head.py                 → MLMHead
  src/encoders/profile_state_encoder.py → ProfileStateEncoder
  src/encoders/event_encoder.py         → EventEncoder
  src/encoders/history_encoder.py       → HistoryEncoder
  src/encoders/rope.py                  → RoPEEncoding
  src/tokenizer/pipeline.py             → TokenizerPipeline
  src/tokenizer/financial_pipeline.py   → FinancialTokenizerPipeline
  src/masking/strategy.py               → MaskingStrategy
  src/adaptation/lora.py                → LoRAAdapter
  src/adaptation/probe.py               → EmbeddingProbe

### Enforcement
  If Claude Code generates a different name: reject it.
  Ask for the correct name from this list.
  Do not accept "it means the same thing."
  Inconsistent naming is the first step toward refactors.

---

## Configuration — Single Source of Truth

ALL architectural parameters live in PRAGMAConfig.
Nothing is hardcoded anywhere in src/.

### The rule
  Bad:
    class ProfileStateEncoder(nn.Module):
        def __init__(self):
            self.d_model = 192  ← hardcoded, causes refactors

  Good:
    class ProfileStateEncoder(nn.Module):
        def __init__(self, config: PRAGMAConfig):
            self.d_model = config.d_model  ← always correct

### Every class takes config as its first argument
  ProfileStateEncoder(config: PRAGMAConfig)
  EventEncoder(config: PRAGMAConfig)
  HistoryEncoder(config: PRAGMAConfig)
  PRAGMA(config: PRAGMAConfig)
  MaskingStrategy(config: PRAGMAConfig)
  MLMHead(config: PRAGMAConfig)

### Three variants defined upfront from Table 1
  PRAGMAConfig.pragma_s()  → 10M params
    d_model=192, d_ffn=768
    profile_layers=1, event_layers=5, history_layers=2
    n_heads=3

  PRAGMAConfig.pragma_m()  → 100M params
    d_model=512, d_ffn=2048
    profile_layers=3, event_layers=16, history_layers=6
    n_heads=8

  PRAGMAConfig.pragma_l()  → 1B params
    d_model=1024, d_ffn=4096
    profile_layers=9, event_layers=45, history_layers=18
    n_heads=16

### Scaling test
  Scaling from PRAGMA-S to PRAGMA-M must require changing
  exactly ONE line in training code:
    config = PRAGMAConfig.pragma_s()
    → config = PRAGMAConfig.pragma_m()
  Nothing else changes.
  If anything else needs to change: the code has hardcoded values.
  Find them. Remove them.

### Every hardcoded number is a bug
  If Claude Code generates a hardcoded architectural number:
  reject it. Ask for the config-driven equivalent.

---

## Dependency Rules — Enforced by Directory Structure

Dependencies flow in ONE direction only.
No exceptions. No circular imports.

### The dependency graph
  src/model/config.py      → no dependencies
  src/tokenizer/           → depends on config only
  src/encoders/rope.py     → depends on config only
  src/encoders/            → depends on config, rope
  src/masking/             → depends on config, tokenizer
  src/model/pragma.py      → depends on config, encoders, masking
  src/model/mlm_head.py    → depends on config, encoders
  src/adaptation/          → depends on config, model
  src/training/            → depends on config, model, masking
  src/evaluation/          → depends on config, model, adaptation
  pipeline/                → depends on src/ only
  scripts/                 → depends on src/ only
  notebooks/               → depends on src/ only

### Absolutely forbidden
  src/encoders/ importing from src/model/
  src/tokenizer/ importing from src/encoders/
  src/model/config.py importing from anywhere in src/
  Any circular import of any kind

### Enforcement
  tests/test_imports.py verifies this graph automatically.
  If Claude Code generates a circular import: reject it.
  Restructure to respect the dependency graph above.
  The graph is fixed. The code bends to fit it.

---

## Type Hints — Complete, Always

Every function signature must have complete type hints.
Every tensor argument must have a shape comment.
No exceptions.

### The standard
  Bad:
    def forward(self, x, t):

  Good:
    def forward(
        self,
        x: torch.Tensor,  # (batch, n_profile_tokens, d_model)
        t: torch.Tensor,  # (batch, n_profile_tokens) log-seconds
    ) -> torch.Tensor:    # (batch, 1, d_model) — [USR] token

### mypy runs in CI
  mypy src/ --strict --ignore-missing-imports

  If mypy fails: fix the types. Not the mypy config.
  The only allowed mypy config change is adding a new
  ignore_missing_imports for a third-party library
  that does not ship type stubs.

### Tensor shape comments
  Every tensor argument must have a shape comment.
  Format: # (dim1, dim2, ...) — description
  Example: # (batch, n_events, d_model) — event embeddings

---

## Pre-Implementation Checklist

Claude Code must confirm each item before generating
implementation code for any component.

Present this checklist to the human. Wait for confirmation.
Do not proceed with implementation until all items are checked.

  [ ] 1. Paper section identified and read
         State the section number and quote the key passage.

  [ ] 2. Interface Protocol defined
         Show the Protocol in src/<module>/__init__.py
         with all method signatures and shape comments.

  [ ] 3. All variable names match the naming conventions
         List the variables this component will use.
         Confirm each is in the naming conventions table.

  [ ] 4. All parameters come from PRAGMAConfig
         Confirm no hardcoded architectural values.
         Show how config is threaded through.

  [ ] 5. No circular imports introduced
         Show the import chain for this component.
         Confirm it respects the dependency graph.

  [ ] 6. All function signatures have complete type hints
         with shape comments on every tensor argument.

  [ ] 7. Test file written and shown for review
         Four test types: shape, math, gradient, spec.
         STOP here. Wait for explicit "Tests approved."

  [ ] 8. Dependency graph respected
         This component's position in the graph stated.

All eight items must be checked before implementation.
No exceptions. No shortcuts.

---

## Lessons from Prior Projects

These rules exist because they were learned the hard way.
They are not theoretical. They prevented real refactors.

1. Build the config first, everything else second.
   Components built without a shared config inevitably
   have hardcoded values that conflict with each other.
   The config is the contract between all components.

2. Name things after the paper, not after what feels natural.
   Natural names drift. Paper names are anchored.
   When you read Section 2.3.2 and the variable is za
   not profile_output or encoder_result — you can follow
   the math directly. That matters at 2am.

3. The interface between components is more important
   than the implementation of either component.
   Get the interface wrong and both components need rewriting.
   Get it right and either can be rewritten independently.

4. An AI will generate plausible code faster than you can
   review it. That is the trap. The review is the work.
   The generation is just typing.

5. When something feels wrong — it probably is.
   Stop. Write a test that captures what feels wrong.
   If the test fails: you were right.
   If the test passes: you learned something.
   Either way you did not waste days building on it.

6. Hardcoded numbers are time bombs.
   They work fine until the second component that needs
   the same number uses a slightly different value.
   Then you spend a day finding the inconsistency.
   Every number comes from config. Always.

7. Circular imports announce themselves at import time.
   That is actually a gift. They tell you immediately
   that the dependency graph is wrong.
   Fix the graph. Do not work around the import error.

---

## OpenShift Integration Test Gate

The repo has two separate test layers.

### Layer 1 — Unit / local tests (default)

```
pytest tests/ -q
```

Always runs. No cluster required. Covers model logic, tokeniser, assembler,
workbench API, dry_run, local training, decorated pipeline authoring, and
pipeline compilation.

### Layer 2 — OpenShift integration tests (opt-in)

```
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=<namespace> \
pytest tests/openshift/ -q
```

Skipped by default. Requires `oc` access and a running OpenShift cluster.
Lives in `tests/openshift/`. See `tests/openshift/README.md` for full docs.

---

### When OpenShift tests are required

OpenShift integration tests are required before merging any change that affects:

- `openshift/` manifests (PyTorchJob YAML, namespace config, RBAC)
- OpenShift Pipeline runtime execution (PipelineRun, TaskRun submission)
- PyTorchJob manifests or distributed training configuration
- Service account, RBAC, image pull, or S3 secret assumptions
- Cluster-side dataset staging or S3-backed artefact paths
- Two-node or multi-node distributed execution on OpenShift
- `src/workbench/` submit or compile behaviour when it touches cluster runtime

OpenShift tests are **not** required for:

- Model-only changes (encoders, masking, MLM head, config)
- Tokeniser-only changes
- Local-only workbench behaviour (dry_run, local mode)
- Documentation-only changes

...unless those changes alter cluster-facing behaviour.

---

### Safety rules for the OpenShift test suite

1. Skip unless `RUN_OPENSHIFT_TESTS=1`.
2. Require `PRAGMA_TEST_NAMESPACE`.
3. Argo CD deploys the long-lived platform substrate. Tests verify it but
   do not deploy or mutate Argo-managed resources.
4. Tests create only short-lived resources labelled with:
   ```
   pragma.redhat.com/test-run=true
   pragma.redhat.com/test-id=<unique-test-id>
   ```
5. Cleanup must be label-scoped. Never use `--all` or kind-wide deletes.
6. Never delete or patch ServiceAccounts, Secrets, namespaces, or
   Argo CD Applications.
7. Never print secret data, tokens, or kubeconfig values.

---

### Cluster-facing development flow

When implementing or changing anything that touches OpenShift:

1. Write / update unit tests (Layer 1).
2. Write / update OpenShift integration tests (Layer 2).
3. Show tests for review before implementation (same rule as model components).
4. Implement.
5. Run unit tests: `pytest tests/ -q`
6. Run workbench examples: `PYTHONPATH=. .venv/bin/python examples/workbench/05_decorated_pipeline.py`
7. Run OpenShift tests if `oc` access is available:
   ```
   RUN_OPENSHIFT_TESTS=1 \
   PRAGMA_TEST_NAMESPACE=<namespace> \
   pytest tests/openshift/ -q
   ```
8. If cluster access is not available, report the exact command and the
   reason it was not run (e.g. "no VPN / no cluster credentials").

---

### Required report for cluster-facing changes

When submitting a PR for cluster-facing work, include:

```
Unit tests:          <pass count> passed, <skip count> skipped
OpenShift tests:     <result> OR "not run — <reason>"
Namespace used:      <namespace>
Resources created:   <list or "none">
Cleanup result:      <success / warnings / not applicable>
xfail / future:      <list of xfail tests and their expected trigger>
Argo mutations:      none (confirmed)
```

---

### Quick reference commands

Read-only cluster checks (safe, no resources created):
```bash
RUN_OPENSHIFT_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_00_oc_access.py \
       tests/openshift/test_01_cluster_prereqs.py \
       tests/openshift/test_02_pipeline_compile.py -q
```

Future pipeline smoke (creates labelled PipelineRun, opt-in):
```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_OPENSHIFT_PIPELINE_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_03_pipeline_smoke_run.py -q
```

Training container smoke — Level 3b (creates ConfigMap + batch/v1 Job, opt-in):
This proves the PRAGMA training image runs correctly in-cluster before DSPA/KFP
is attempted. Not a PyTorchJob — a single-pod batch/v1 Job with no DDP.
```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_OPENSHIFT_TRAINING_JOB_SMOKE=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
PRAGMA_TRAINING_IMAGE=<registry>/<repo>/pragma-encoder:latest \
pytest tests/openshift/test_03b_training_job_smoke.py -q
```

Future PyTorchJob smoke (creates labelled PyTorchJob, opt-in):
```bash
RUN_OPENSHIFT_TESTS=1 \
RUN_PYTORCHJOB_TESTS=1 \
PRAGMA_TEST_NAMESPACE=pragma-encoder \
pytest tests/openshift/test_04_pytorchjob_smoke.py -q
```
