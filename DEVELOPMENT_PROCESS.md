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
