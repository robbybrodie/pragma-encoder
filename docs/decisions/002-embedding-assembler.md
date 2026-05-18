# Decision 002: EmbeddingAssembler Architecture

Status: Accepted — pending implementation in Issue 11
Date: 2026-05-18

---

### Context

The TokenizerPipeline produces symbolic token IDs.
PRAGMA.forward() consumes embedded float tensors.
A bridge component is needed to connect them.

Without an explicit bridge, the conversion would be scattered:
embedding table lookups, ID-range arithmetic, mask application,
and target-ID localisation would each need to be reimplemented
by every caller. That creates four sources of the same bug.

---

### Decision

Implement EmbeddingAssembler as a first-class nn.Module
with four supporting types:

**1. VocabularySpec (frozen dataclass)**

Exported by TokenizerPipeline.vocabulary_spec().
Contains canonical ID ranges — not tokenizer internals.

Fields:
```
special_tokens: dict[str, int]
key_start: int, key_size: int
value_start: int, value_size: int
total_embedding_vocab_size: int
field_key_ids: dict[str, int]          (optional, for debug)
field_value_ranges: dict[str, tuple[int,int]]  (optional)
```

VocabularySpec is the only public surface of TokenizerPipeline
that EmbeddingAssembler is allowed to import. It contains ID
ranges, not tokenizer objects or fit state.

---

**2. VocabularyMap**

Constructed from VocabularySpec only.
Owns all global/local ID conversion:

```
global_to_local_value_id(ids) → ids - value_start
is_key_id(ids)                → bool
is_special_id(ids)            → bool
is_global_value_id(ids)       → bool
```

VocabularyMap is the single source of truth for ID arithmetic.
No other class performs ID range checks or offset subtraction.

---

**3. AssembledBatch (dataclass)**

```
xa:       (batch, na, d_model)       — profile embeddings
xe:       (batch, ne, ni, d_model)   — event token embeddings
xt:       (batch, ne, 3)             — calendar features
te:       (batch, 1+ne)              — history temporal coords
mlm_mask: (batch, ne, ni) bool       — True at masked positions
targets:  (batch, ne, ni) long       — value_vocab_ids or -100
```

targets uses ignore_index = -100 at unmasked positions, matching
PyTorch cross_entropy convention. Masked positions hold
value-vocab-local IDs in [0, value_vocab_size).

---

**4. EmbeddingAssembler (nn.Module)**

```
E:      nn.Embedding(vocab_spec.total_embedding_vocab_size, d_model)
PosEmb: sinusoidal within-field (Equation 1)
vocab:  VocabularyMap
```

Implements Equation 1: x = PosEmb(E(k) + E(v))

Responsibility boundary:

```
MaskingStrategy:     CHOOSES which positions to mask
EmbeddingAssembler:  APPLIES corruption in embedding space
                     CONVERTS global IDs to local target IDs
```

EmbeddingAssembler does not decide which positions to mask.
MaskingStrategy does not convert IDs or apply embeddings.

---

### Key invariant (enforced by test)

```python
assert targets[targets != IGNORE_INDEX].max() < config.value_vocab_size
```

This invariant must be tested in `tests/test_assembler.py`.
Any implementation that cannot satisfy it is incorrect.

---

### Vocabulary ID naming (explicit, no ambiguity)

| Name | Range | Used for |
|---|---|---|
| `global_token_id` | `[0, total_embedding_vocab_size)` | Embedding table lookup `E[id]` |
| `value_vocab_id` | `[0, value_vocab_size)` | MLM target passed to MLMHead / cross_entropy |
| `field_value_id` | per-field local range | Internal to tokenizer only, NOT used by MLMHead |

Never pass a `global_token_id` as an MLM target. Never pass a
`value_vocab_id` to the embedding table. Confusion between these
causes silent out-of-range errors that are hard to detect at test
time because the loss still computes without raising.

---

### Dependency graph

```
TokenizerPipeline → VocabularySpec → VocabularyMap → EmbeddingAssembler
                                                             ↓
                                                       AssembledBatch
                                                             ↓
                                                    PRAGMA.forward()
```

EmbeddingAssembler does NOT import TokenizerPipeline.
PRAGMA does NOT import EmbeddingAssembler.
The data flow is one-directional and acyclic.

---

### What this enables

- PRAGMA.forward() remains tensor-only — no token ID knowledge
  inside PRAGMA or any encoder
- Clean seam for testing each component independently:
  VocabularyMap tested without assembler
  AssembledBatch shape tested without PRAGMA
  PRAGMA forward tested with random float tensors
- The key invariant (targets < value_vocab_size) is assertable
  in isolation before wiring to training

---

### Consequences

Enables:
  - Single point of responsibility for ID-to-embedding conversion
  - Single point of responsibility for target localisation
  - Testable invariant on output targets
  - EmbeddingAssembler can be swapped for a different bridge
    (e.g. dense embedding from pretrained language model) by
    replacing only this component

Constrains:
  - TokenizerPipeline must implement vocabulary_spec()
  - PRAGMA.forward() must never receive global_token_ids as targets
  - All ID arithmetic must go through VocabularyMap
  - EmbeddingAssembler must be a nn.Module (embedding table is
    a learnable parameter trained end-to-end with the encoders)

What future sessions must not contradict:
  Do not pass global_token_ids as MLM targets.
  Do not perform ID range arithmetic outside VocabularyMap.
  Do not import TokenizerPipeline from EmbeddingAssembler.
  Do not embed IDs inside PRAGMA or any encoder.
  VocabularySpec is the only public surface of TokenizerPipeline
  that crosses the tokenizer/model boundary.

---

### Paper

Equation 1 (Section 2.3.1):

```
x_{ij} = PosEmb_j( E(k_i) + E(v_{ij}) )
```

where:
  `k_i`    — key token ID for field i (semantic type)
  `v_{ij}` — value token ID for position j within field i
  `E`      — shared embedding table (both key and value IDs)
  `PosEmb_j` — sinusoidal positional embedding for position j
               within the field (not across fields)

The EmbeddingAssembler implements this equation exactly.
Key and value token IDs are looked up in the same embedding
table E and summed before positional encoding is applied.

Section 2.2 defines the vocabulary structure:
  ~60 key tokens (one per field type)
  ~28k value tokens (numerical buckets, categoricals, BPE subwords)

Section 2.3.1 defines:
  d_model:          192 / 512 / 1024 (S / M / L)
  key_vocab_size:   ~60
  value_vocab_size: ~28k
  total_embedding_vocab_size: key_vocab_size + value_vocab_size
                              + n_special_tokens

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1,
Equation 1, Sections 2.2 and 2.3.1
