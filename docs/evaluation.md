# Evaluation Guide

How to evaluate a pretrained PRAGMA model on downstream tasks.

> Reference: Ostroukhov et al. (2026), Section 3

---

## Evaluation Protocols

Two protocols from the paper (Section 3.1):

### 3.1.1 — Linear Embedding Probe

Freeze the PRAGMA encoder and train a single linear layer on top.

**When to use:** Rapid evaluation of embedding quality. Diagnostic tool.
**Speed:** Minutes on CPU.

```python
from pragma_encoder.adaptation import LinearProbe
from pragma_encoder.evaluation import DownstreamEvaluator

evaluator = DownstreamEvaluator(task_name="fraud_detection", metric="auc")

# Extract embeddings from frozen PRAGMA
# Train logistic regression on embeddings
results = evaluator.evaluate_linear_probe(
    embeddings_train, labels_train,
    embeddings_test, labels_test,
)
print(results)  # {'auc': 0.85, 'pr_auc': 0.72, 'f1': 0.68, 'ks': 0.61}
```

See `notebooks/03_pragma_embedding_probe.ipynb` for an interactive example.

### 3.1.2 — LoRA Fine-tuning

Apply LoRA adapters to the PRAGMA encoder and fine-tune end-to-end.

**When to use:** Maximum performance on a specific task.
**Speed:** Hours on GPU.

```python
from pragma_encoder.adaptation import apply_lora_to_pragma, PRAGMALoRAConfig

lora_config = PRAGMALoRAConfig(r=16, lora_alpha=32, n_classes=2)
lora_model = apply_lora_to_pragma(pragma_model, lora_config)
# Fine-tune lora_model on labelled data
```

See `notebooks/04_pragma_lora_finetuning.ipynb` for a walkthrough.

---

## Downstream Tasks

Tasks from the paper (Section 3):

| Task | Labels | Primary Metric | Section |
|---|---|---|---|
| Fraud detection | Fraudulent transaction (0/1) | AUC | 3.2 |
| Churn prediction | Customer churned in 90d (0/1) | AUC | 3.3 |
| Credit risk | Loan default (0/1) | PR-AUC | 3.4 |
| Card decline | Transaction declined (0/1) | AUC | 3.5 |
| Merchant category | MCC code (multiclass) | F1-macro | 3.6 |

---

## Running Evaluation

```bash
python scripts/evaluate_pragma.py \
    --checkpoint outputs/pragma-s/checkpoint-final.pt \
    --task fraud_detection \
    --data-dir /data/labelled \
    --output-dir outputs/evaluation
```

---

## Metrics Reference

| Metric | Range | Interpretation |
|---|---|---|
| AUC (AUROC) | 0.5–1.0 | 0.5 = random, 1.0 = perfect |
| PR-AUC | 0.0–1.0 | Higher = better, useful for imbalanced labels |
| F1 | 0.0–1.0 | Harmonic mean of precision and recall |
| KS statistic | 0.0–1.0 | Max separation between score distributions |
