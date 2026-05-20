# Architecture Decision Records — PRAGMA Encoder

Decisions derived from the PRAGMA paper:
> Ostroukhov, M. et al. (2026). PRAGMA: Revolut Foundation Model. arXiv:2604.08649v1

Rules:
- One decision per file, sequentially numbered.
- Accepted decisions are binding. Contradicting them requires a new superseding ADR.
- To change a decision: create a new ADR with `Supersedes: NNN` in the header.

---

## Index

| ADR | Title | Status | Summary |
|-----|-------|--------|---------|
| [001](001-encoder-architecture.md) | PRAGMA Encoder Architecture | Accepted | Encoder-only bidirectional Transformer with three separate branches, RoPE, MLM objective, and key-value-time tokenisation. All architecture constraints from §2.1–2.3. |
| [002](002-embedding-assembler.md) | EmbeddingAssembler Architecture | Accepted — pending impl in Issue 11 | Bridge component between TokenizerPipeline and PRAGMA.forward(). Defines VocabularySpec, VocabularyMap, AssembledBatch, and EmbeddingAssembler. ID arithmetic lives exclusively in VocabularyMap. |
| [003](003-workbench-training-api.md) | Workbench-Led Training API | Accepted | `train_pragma()` as the workbench entry point. Modes: `dry_run` (preview), `local` (subprocess), `pipeline` (compiled intent), `auto` (env-detect), `cluster` (not yet implemented). `DatasetManifest` as canonical prepared-data contract. |
| [004](004-workbench-decorated-pipelines.md) | Workbench-Decorated Pipeline Authoring | Accepted | `@pragma_pipeline` decorator DSL for expressing training intent. `compile()` emits KFP YAML. `submit()` is a future extension. Exactly one `train()` call required per decorated function. Lazy import boundary: `compile()` only. |
| [005](005-training-orchestration.md) | Training Orchestration | Accepted | Custom PyTorch training loop (`scripts/train_pragma.py`). torchrun/DDP for distributed training. KFTO PyTorchJob on OpenShift. KFP SDK v2 for workflow orchestration. S3 for all canonical storage. NeMo AutoModel **not currently used** (future option only). |
| [006](006-fine-tuning-lora.md) | LoRA / PEFT Fine-Tuning | Accepted | Hugging Face PEFT library for LoRA fine-tuning (§3.1.2). Implemented in `src/adaptation/lora.py`. rank=8, alpha=8. Targets QKV + FFN layers across all three encoders. 2.37% trainable parameter fraction for PRAGMA-S. |

---

## Supersession chain

None of the current ADRs have been superseded. ADR 005 replaces the stale embedded
"Decision 005: NeMo for Training Orchestration" that previously lived inside ADR 001.
ADR 006 replaces the stale embedded "Decision 006: PEFT/LoRA for Fine-Tuning" from ADR 001.

---

## Non-negotiable constraints (summary)

From ADR 001:
- Encoder-only. Bidirectional attention. No causal masks.
- Three separate encoder branches. No merged encoder.
- RoPE in Profile State Encoder and History Encoder.
- MLM objective. Three masking strategies. Three-vector MLM head.
- Key-value-time tokenisation. No text serialisation.

From ADR 002:
- ID arithmetic only in VocabularyMap.
- Never pass global_token_ids as MLM targets.

From ADR 003:
- DatasetManifest is the canonical prepared-data contract. No scattered S3 paths.
- No PVC for canonical storage. S3 only.

From ADR 004:
- `@pragma_pipeline` decorated function must call `train()` exactly once.
- No static `src/ → pipeline/` import. `compile()` lazy-loads `pipeline/` only.

From ADR 005:
- No NeMo, PyTorch Lightning, or HF Trainer for pre-training without new ADR.
- PyTorchJob `kubeflow.org/v1` only (no TrainJob Tech Preview).

From ADR 006:
- Use PEFT library. No custom LoRA implementation.
- `rank=8`, `alpha=8` defaults from paper.
