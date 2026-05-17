# Architecture Decision Records — PRAGMA Encoder

These decisions are derived from the PRAGMA paper:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Once accepted, these decisions are fixed.
Future Claude Code sessions must not contradict them.
To change a decision: create a new ADR superseding this one.

---

## Decision 001: Encoder-Only Architecture

Status: Accepted
Paper reference: Section 2.3

### Context
The adjacent NVIDIA blueprint uses a decoder-only
(GPT/Llama-style) architecture with causal language
modelling. A decision was needed on whether to follow
the same approach or implement the PRAGMA architecture
as described in the paper.

### Decision
Use encoder-only (BERT-style) bidirectional Transformer.
Do not use decoder-only or causal architecture.

### Consequences
Enables:
  - Each token attends to all other tokens in both directions
  - Better representation learning for embedding extraction
  - Masked modelling objective (requires bidirectional attention)
  - Direct implementation of the paper as published

Constrains:
  - Cannot generate text (no autoregressive decoding)
  - Cannot use causal attention mask anywhere in the model
  - All three encoders must use bidirectional attention

What future sessions must not contradict:
  Do not add causal masking to any attention layer.
  Do not switch to a decoder architecture.
  Do not use GPT-style generation.

---

## Decision 002: Three Separate Encoder Branches

Status: Accepted
Paper reference: Section 2.3

### Context
The model must process two distinct input types:
profile state (static customer attributes) and event
history (sequential transactions). A decision was needed
on whether to use one unified encoder or separate branches.

### Decision
Use three separate encoder branches:
  1. Profile State Encoder (Section 2.3.2)
  2. Event Encoder (Section 2.3.3)
  3. History Encoder (Section 2.3.4)

Each is a separate bidirectional Transformer with its
own weights, depth, and positional encoding strategy.

### Consequences
Enables:
  - Profile state processed independently of events
  - Event-level encoding before cross-event attention
  - Clean ablation (can remove profile branch entirely
    as in paper Section 3.4.2)
  - Profile state contributes a single [USR] token to
    the History Encoder — clean interface

Constrains:
  - Three separate forward passes required
  - Cannot merge into one unified encoder
  - The [USR] token from Profile State Encoder and
    [EVT] tokens from Event Encoder are the only
    outputs passed to the History Encoder

What future sessions must not contradict:
  Do not merge the three encoders into one.
  Do not pass raw profile tokens to the History Encoder.
  Do not pass raw event tokens to the History Encoder.
  Only the aggregated [USR] and [EVT] tokens are passed.

---

## Decision 003: RoPE Positional Encoding

Status: Accepted
Paper reference: Section 2.3.2, Su et al. (2024)

### Context
The model must encode temporal coordinates (log-seconds
to the most recent event) as positional information.
Standard sinusoidal or learned positional embeddings
were considered.

### Decision
Use Rotary Position Embeddings (RoPE) for temporal
coordinate encoding in the Profile State Encoder and
History Encoder.

Use calendar feature embeddings (periodic functions)
for the Event Encoder's within-event temporal signals.

### Consequences
Enables:
  - Relative temporal positions encoded naturally
  - Extrapolation to unseen temporal gaps
  - Disentangled from value-level positional embeddings
    (separate from the within-field position embeddings)

Constrains:
  - Must implement RoPE correctly as rotation matrices
  - Cannot use absolute positional embeddings
  - Cannot use learned positional embeddings
  - The temporal coordinate t must be provided to every
    RoPE-using encoder as a separate argument

What future sessions must not contradict:
  Do not replace RoPE with sinusoidal positional encoding.
  Do not replace RoPE with learned positional embeddings.
  Do not mix the temporal RoPE with within-field PosEmb.
  (Within-field positions use standard sinusoidal PosEmb
   per Section 2.3.1 — this is separate from temporal RoPE)

---

## Decision 004: Masked Modelling Objective

Status: Accepted
Paper reference: Section 2.3.5

### Context
The adjacent NVIDIA blueprint uses causal language
modelling (predict next token). A decision was needed
on the pre-training objective for PRAGMA.

### Decision
Use masked modelling (MLM) with three masking strategies:
  1. Token-level masking: 15% probability
  2. Event-level masking: 10% probability
  3. Semantic-type (key) level masking: 10% probability

A small fraction of selected positions replaced with
[UNK] instead of [MASK] for input dropout effect.

The MLM head receives a 3d-dimensional input:
  - Event Encoder token-level output at masked position
  - History Encoder [EVT] token for cross-event context
  - History Encoder [USR] token for user-level context

### Consequences
Enables:
  - Bidirectional attention (requires non-causal model)
  - Each masked token uses full context (past and future)
  - Three masking strategies teach different capabilities:
    token-level: individual field patterns
    event-level: full event reconstruction
    key-level: value prediction given field type

Constrains:
  - Training is more complex than causal LM
  - Three masking rates must match paper exactly
  - MLM head must concatenate three d-dimensional vectors
    not just use the encoder output directly

What future sessions must not contradict:
  Do not switch to causal LM objective.
  Do not change the masking rates without new ADR.
  Do not simplify the MLM head to use one input vector.
  All three (token-level, event-level, key-level) masking
  strategies must be implemented.

---

## Decision 005: NeMo for Training Orchestration

Status: Accepted
Paper reference: N/A (implementation decision)

### Context
The PRAGMA paper describes custom training infrastructure
(Section 2.4). For this open source implementation a
decision was needed on the training framework.

### Decision
Use NVIDIA NeMo AutoModel for training orchestration.

The model itself is pure PyTorch.
NeMo wraps the training loop, handles distributed
training via torchrun, manages checkpointing,
and integrates with the OpenShift AI deployment pattern
established in the adjacent NVIDIA blueprint repo.

### Consequences
Enables:
  - Consistent training pattern across both repos
  - Distributed training via KFTO PyTorchJob on OpenShift
  - NeMo's bf16 mixed precision and Muon/AdamW optimisers
  - Checkpoint management and resume

Constrains:
  - Model must be compatible with NeMo AutoModel interface
  - Training config must use NeMo YAML format
  - Cannot use PyTorch Lightning or other training frameworks
    without a new ADR

What future sessions must not contradict:
  Do not introduce PyTorch Lightning.
  Do not introduce Hugging Face Trainer for pre-training.
  NeMo is the training framework. PyTorch is the model.

---

## Decision 006: PEFT/LoRA for Fine-Tuning

Status: Accepted
Paper reference: Section 3.1.2

### Context
The paper describes LoRA fine-tuning updating only
2-4% of model weights. A decision was needed on
the LoRA implementation library.

### Decision
Use Hugging Face PEFT library for LoRA fine-tuning.

Default configuration from the paper:
  rank = 8
  alpha = 8
  target modules: QKV projections and MLP layers
  trainable parameters: 2-4% of total model parameters

### Consequences
Enables:
  - Minimal parameter overhead per downstream task
  - Frozen backbone shared across tasks
  - Fast specialisation without catastrophic forgetting
  - Standard, well-tested LoRA implementation

Constrains:
  - Must use PEFT library, not custom LoRA implementation
  - rank and alpha defaults are 8 as per paper
  - Target modules must include QKV and MLP
  - Cannot use full fine-tuning without new ADR

What future sessions must not contradict:
  Do not implement custom LoRA. Use PEFT.
  Do not change rank/alpha defaults without paper justification.

---

## Decision 007: Key-Value-Time Tokenisation

Status: Accepted
Paper reference: Section 2.2

### Context
The NVIDIA blueprint uses tabular GPT-style tokenisation
(token strings joined by separators). The PRAGMA paper
describes a different disentangled embedding space.

### Decision
Use PRAGMA's key-value-time tokenisation scheme:
  - Semantic type (key): single token per field
  - Value: type-specific encoding
    numerical → percentile bucket token
    categorical → single token
    textual → BPE subword tokens
  - Temporal: log-seconds + calendar features

This is fundamentally different from the NVIDIA blueprint.
Do not import or reuse the adjacent repo's tokeniser.

### Consequences
Enables:
  - Model distinguishes field meaning from field value
  - Numerical values preserve magnitude and ordering
  - Text fields use semantic subword representation
  - Temporal structure explicit and learnable

Constrains:
  - Two separate vocabularies: key vocab (~60) and value vocab (~28k)
  - Positional encodings index within-field not across-fields
  - Cannot use the NVIDIA blueprint's tokeniser
  - BPE tokeniser must be trained on the transaction corpus

What future sessions must not contradict:
  Do not reuse the NVIDIA blueprint tokeniser.
  Do not serialise records as text strings.
  Keys and values must be separate embedding lookups.
  Within-field positions index values within one field only.

---

## Decision 008: Profile State Separate from Event History

Status: Accepted
Paper reference: Section 2.1.2

### Context
A decision was needed on whether to treat static customer
attributes (plan, region, balance quantile) as regular
events in the sequence or as a separate input stream.

### Decision
Profile state is a separate input to the Profile State
Encoder. It is never mixed with the event sequence.

Life-long events (first_topup, account_age) are encoded
as profile state with individual timestamps, not as
regular events in the history.

### Consequences
Enables:
  - Clean ablation: remove profile branch to test event-only
    (paper Section 3.4.2 shows +31.8% PR-AUC from profile state
    on credit scoring)
  - Profile state can include features unavailable in events
  - Static and dynamic signals processed by specialised encoders

Constrains:
  - Two separate inputs to the model (not one sequence)
  - Profile state tokenised identically to events
    but processed by a different encoder
  - Cannot prepend profile tokens to the event sequence

What future sessions must not contradict:
  Do not merge profile state into the event sequence.
  Do not pass profile tokens to the Event Encoder.
  Profile State Encoder and Event Encoder are independent.
