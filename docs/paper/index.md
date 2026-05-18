# PRAGMA Paper Knowledge Base — Index

Structured extraction of:

**"PRAGMA: Revolut Foundation Model"**
Ostroukhov et al. (2026), arXiv:2604.08649v1

Full PDF: `docs/pragma-paper.pdf`

---

## Quick Reference

| What you need | Go to |
|---------------|-------|
| Every specific number from the paper | `key-numbers.md` |
| Every equation with implementation mapping | `equations.md` |
| Model size parameters (Table 1) | `table-1-model-sizes.md` |
| Main results vs baseline (Table 2) | `table-2-main-results.md` |
| Tokenisation details | `02-02-tokenisation.md` |
| Architecture details | `02-03-architecture.md` |
| LoRA configuration | `03-01-protocol.md` |
| Why profile state matters | `03-04-ablations.md` |

---

## File Index

### Reference Files (read before writing tests)

| File | Content | Paper Reference |
|------|---------|-----------------|
| `key-numbers.md` | Every number with source section and implementation target | throughout |
| `equations.md` | All equations with LaTeX, variable definitions, implementation | §2.2, §2.3 |
| `table-1-model-sizes.md` | Table 1 extracted exactly + derived invariants | Table 1, p.6 |
| `table-2-main-results.md` | Table 2 extracted exactly | Table 2, p.9 |

### Section Files

| File | Content | Paper Reference |
|------|---------|-----------------|
| `00-abstract.md` | Abstract, key claims, design constraints | Abstract, p.1 |
| `01-introduction.md` | Problem framing, contributions, encoder-only rationale | §1, p.1–3 |
| `02-01-dataset.md` | Dataset scale, event history, profile state, life-long events | §2.1, p.3–4 |
| `02-02-tokenisation.md` | Key-value-time tokenisation scheme | §2.2, p.4–5 |
| `02-03-architecture.md` | Three encoders + training objective (CRITICAL) | §2.3, p.5–7 |
| `02-04-infrastructure.md` | Batching, packing, truncation, compute | §2.4, p.7–8 |
| `03-evaluation.md` | Evaluation overview and reporting convention | §3, p.8 |
| `03-01-protocol.md` | Embedding probe + LoRA configuration | §3.1, p.8 |
| `03-02-tasks.md` | Six downstream tasks described | §3.2, p.9 |
| `03-03-results.md` | Main results, scale effects | §3.3, p.9–12 |
| `03-04-ablations.md` | LoRA effect, profile state effect, AML limitation | §3.4, p.12–15 |
| `04-related-work.md` | Transformer, MLM, tabular, recommender, finance models | §4, p.15–17 |

---

## Paper Structure with Page Numbers

| Section | Title | Pages |
|---------|-------|-------|
| Abstract | — | p.1 |
| 1 | Introduction | p.1–3 |
| 2 | Pre-training | p.3–8 |
| 2.1 | Dataset | p.3–4 |
| 2.1.1 | Event History | p.3 |
| 2.1.2 | Profile State | p.4 |
| 2.1.3 | Pre-training Time Range | p.4 |
| 2.2 | Tokenisation | p.4–5 |
| 2.3 | Model Architecture | p.5–7 |
| 2.3.1 | Token Embedding | p.6 |
| 2.3.2 | Profile State Encoder | p.6 |
| 2.3.3 | Event Encoder | p.6–7 |
| 2.3.4 | History Encoder | p.7 |
| 2.3.5 | Training | p.7 |
| 2.4 | Training Infrastructure | p.7–8 |
| 3 | Evaluation | p.8–15 |
| 3.1 | Evaluation Protocol | p.8 |
| 3.1.1 | Embedding Probing | p.8 |
| 3.1.2 | Downstream Adaptation with LoRA | p.8 |
| 3.1.3 | Preparing Downstream Datasets | p.8 |
| 3.2 | Downstream Tasks | p.9 |
| 3.3 | Main Results | p.9–12 |
| 3.3.1 | Effect of Model Scale | p.10–12 |
| 3.3.2 | Effect of Pre-training | p.12 |
| 3.4 | Additional Experiments and Ablations | p.12–15 |
| 3.4.1 | Effect of Low-Rank Adaptation | p.12 |
| 3.4.2 | Effect of Profile State | p.12–13 |
| 3.4.3 | Communication Engagement (Uplift) | p.12–13 |
| 3.4.4 | Effect of a Pre-trained Text Encoder | p.13–14 |
| 3.4.5 | Limitations in Highly Relational Tasks: AML | p.14–15 |
| 4 | Related Work | p.15–17 |
| 5 | Conclusion | p.17 |

---

## How to Use This Knowledge Base

### Before implementing any component:

1. Find the relevant section file for the component
2. Read the paper section
3. Check `key-numbers.md` for every number you plan to use in tests
4. Check `equations.md` for every equation you plan to implement
5. If a number is not in `key-numbers.md` — it is not from the paper. Do not use it.

### Component → Knowledge Base mapping:

| Component | Primary reference | Secondary reference |
|-----------|-------------------|---------------------|
| `PRAGMAConfig` | `table-1-model-sizes.md` | `02-03-architecture.md` |
| `PRAGMATokenizer` | `02-02-tokenisation.md` | `equations.md` (Eq 1, Eq 2) |
| `RoPEEncoding` | `02-03-architecture.md` §2.3.2 | `equations.md` (Eq 9) |
| `ProfileStateEncoder` | `02-03-architecture.md` §2.3.2 | `equations.md` (Eq 4) |
| `EventEncoder` | `02-03-architecture.md` §2.3.3 | `equations.md` (Eq 3, 5) |
| `HistoryEncoder` | `02-03-architecture.md` §2.3.4 | `equations.md` (Eq 6, 7) |
| `MaskingStrategy` | `02-03-architecture.md` §2.3.5 | `key-numbers.md` |
| `MLMHead` | `02-03-architecture.md` §2.3.5 | `equations.md` (Eq 8) |
| `PRAGMA` (full model) | `02-03-architecture.md` | `table-1-model-sizes.md` |
| `LoRAAdapter` | `03-01-protocol.md` §3.1.2 | `key-numbers.md` |
| `EmbeddingProbe` | `03-01-protocol.md` §3.1.1 | `key-numbers.md` |
