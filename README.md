# PRAGMA Encoder — Open Source Implementation

An independent implementation of the PRAGMA foundation model architecture for financial transaction data. This is not Revolut's trained model — it is an open implementation of the published architecture. It solves the problem of learning dense, contextualised representations of customer transaction histories for downstream financial ML tasks.

## Attribution

This is an independent implementation of the architecture described in:

> Ostroukhov, M. et al. (2026). PRAGMA: Revolut Foundation Model.
> arXiv:2604.08649v1

The original paper is by Revolut Research and NVIDIA.
This implementation is not affiliated with Revolut or NVIDIA.
The architecture is implemented from the paper description.
No weights, training data, or proprietary code from Revolut are used or included.

---

## What this implements

### Architecture (Section 2.3)

The three-encoder architecture from the paper:

- **Profile State Encoder** (Section 2.3.2) — bidirectional Transformer processing static customer attributes (plan, region, account tenure, balance quantile) with RoPE positional encoding. Outputs the `[USR]` representation.

- **Event Encoder** (Section 2.3.3) — bidirectional Transformer processing a single financial event (transaction) as a flat `(key, value, time)` token sequence with calendar embeddings. Outputs the `[EVT]` representation.

- **History Encoder** (Section 2.3.4) — bidirectional Transformer that processes the concatenated sequence `z = [USR : EVT₁ : EVT₂ : ...]` with RoPE temporal encoding. The `[USR]` token at position 0 conditions all event representations through bidirectional self-attention. No cross-attention sublayer. No `[HIST]` token.

### Tokenisation (Section 2.2)

The key-value-time tokenisation scheme:

| Field type | Strategy | Implementation |
|---|---|---|
| Numerical | Percentile buckets | `src/tokenizer/numerical.py` |
| Categorical | Single token | `src/tokenizer/categorical.py` |
| Text | BPE subwords | `src/tokenizer/textual.py` |
| Timestamp | Log-seconds + calendar | `src/tokenizer/temporal.py` |

### Training Objective (Section 2.3.5)

Three-strategy masked event modelling (MEM):

1. Token masking — standard BERT-style token replacement
2. Field masking — mask all tokens of a field type across the sequence
3. Event masking — mask entire event representations

MLM loss with label smoothing (ε = 0.1).

### Downstream Adaptation (Section 3.1)

- **Linear embedding probe** (Section 3.1.1) — frozen encoder + logistic regression
- **LoRA fine-tuning** (Section 3.1.2) — parameter-efficient via Hugging Face PEFT

---

## What this does NOT implement

The full-scale training infrastructure from the paper (207B tokens, LMDB storage, dynamic batching at Revolut scale) is not implemented in this release. A simplified training loop suitable for research and smaller datasets is provided.

The Profile State Encoder is implemented but the ProfileTokenizerPipeline for real static customer attributes (plan, region, balance quantile) is not — during training the profile path receives synthetic integer tensors in valid ID ranges rather than tokenised customer profiles.

The pretrained weights from Revolut are not available and are not included. Train your own weights on your own transaction data.

---

## Relationship to the NVIDIA blueprint

This repo sits adjacent to:

```
../transaction-foundation-model-openshiftai/
```

That repo deploys the NVIDIA Transaction Foundation Model blueprint on OpenShift AI using a decoder-only, causal language modelling approach (tabular-GPT style).

**This repo implements the PRAGMA architecture** which is architecturally distinct:

| | PRAGMA (this repo) | NVIDIA blueprint (adjacent) |
|---|---|---|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional | Causal |
| Tokenisation | Key-value-time | Tabular text |
| Encoders | 3 separate | 1 monolithic |
| Objective | Masked modelling | Causal LM |

Both repos share the same OpenShift AI deployment patterns (GitOps via ArgoCD, KFP pipelines, KFTO PyTorchJob, Sealed Secrets). See `docs/architecture.md` for the full comparison.

---

## Deployment

Deploys on Red Hat OpenShift AI via the same GitOps pattern as the adjacent NVIDIA blueprint repo.

```
openshift/
  gitops/          # Kubernetes manifests (ArgoCD syncs these)
  argocd/          # ArgoCD Application definition
  secrets/         # Credential templates (populated values gitignored)
  notebook-image/  # Dockerfile for OpenShift AI workbench
  serving/         # KServe InferenceService for embedding serving
```

Quick start:
```bash
./openshift/scripts/bootstrap-project.sh
```

See `docs/training-guide.md` for pretraining instructions and `docs/evaluation.md` for downstream evaluation.

OpenShift AI 3.3 primitive alignment and platform responsibility split: `docs/openshift-ai-3.3-alignment.md`

---

## Repository Structure

```
src/
  tokenizer/    Key-value-time tokenisation (Section 2.2)
  encoders/     Three-encoder architecture (Sections 2.3.2–2.3.4)
  masking/      Three-strategy MLM masking (Section 2.3.5)
  model/        Full PRAGMA model assembly
  adaptation/   LoRA fine-tuning and linear probes (Section 3.1)
  training/     MLM objective, sequence packing, dynamic batching
  evaluation/   Downstream task evaluation and metrics (Section 3)
tests/          Test suite (pytest)
configs/        Model hyperparameter configs (PRAGMA-S/M/L)
docs/           Architecture, training, and evaluation guides
notebooks/      Interactive Jupyter notebooks
pipeline/       KFP SDK v2 pipeline components
openshift/      OpenShift AI deployment manifests
scripts/        Training and evaluation entrypoints
```

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Explore architecture
python -c "
from src.model import PRAGMA, PRAGMAConfig
config = PRAGMAConfig.pragma_s()
model = PRAGMA(config)
n = sum(p.numel() for p in model.parameters())
print(f'PRAGMA-S: {n:,} parameters')
"
```

---

## Paper Reference

```bibtex
@article{ostroukhov2026pragma,
  title={PRAGMA: Revolut Foundation Model},
  author={Ostroukhov, Mikhail and others},
  journal={arXiv preprint arXiv:2604.08649},
  year={2026}
}
```
