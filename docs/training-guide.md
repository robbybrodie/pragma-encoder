# Training Guide

How to pretrain the PRAGMA encoder on your transaction data.

> Reference: Ostroukhov et al. (2026), Section 2.4 (Training Setup)

---

## Prerequisites

1. Python environment with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Transaction data in parquet format with at minimum:
   - `customer_id` — unique customer identifier
   - `timestamp` — transaction datetime
   - `amount_local` — transaction amount in local currency
   - `currency` — ISO 4217 currency code
   - `merchant_category` — merchant category code (MCC)
   - `transaction_type` — debit/credit/transfer/etc.

---

## Step 1: Data Preparation

Tokenise your transaction data using the FinancialTokenizerPipeline:

```python
from src.tokenizer import FinancialTokenizerPipeline

pipeline = FinancialTokenizerPipeline()
# Fit tokenisers on training data
# Save fitted tokeniser state
```

See `notebooks/01_pragma_tokenization.ipynb` for an interactive walkthrough.

---

## Step 2: Choose a Model Size

| Config | Parameters | Recommended Use |
|---|---|---|
| `pragma_s.yaml` | ~10M | Research, development, quick experiments |
| `pragma_m.yaml` | ~100M | Aspirational, medium-scale datasets |
| `pragma_l.yaml` | ~1B | Aspirational, Revolut-scale datasets |

Start with PRAGMA-S for all initial experiments. It fits on a single GPU node and trains in reasonable time on datasets of millions of transactions — a 90-day window across a retail customer base is well within scope; hundreds of billions of tokens is not.

---

## Step 3: Run Pretraining

### Local training (PRAGMA-S, research):
```bash
python scripts/train_pragma.py \
    --config configs/pragma_s.yaml \
    --data-dir /data/transactions \
    --output-dir outputs/pragma-s \
    --epochs 10 \
    --batch-size 32
```

### OpenShift AI (via KFP pipeline):
Register the pipeline:
```
notebooks/pipeline/00_register_pipeline.ipynb
```
Then trigger a run from the OpenShift AI pipeline UI or via the KFP SDK.

### Distributed training (KFTO PyTorchJob):
Apply the PyTorchJob manifest:
```bash
oc apply -f openshift/gitops/training/pytorchjob-pragma-s.yaml
```

---

## Step 4: Monitor Training

Training metrics to watch:
- **MLM loss** — should decrease from ~log(vocab_size) to < 3.0
- **Perplexity** — derived from MLM loss, lower is better
- **Masking strategy distribution** — verify all three strategies fire

---

## Masking Strategy

PRAGMA uses three masking strategies (Section 2.3.5):

```python
from src.masking import MaskingStrategy

masker = MaskingStrategy(config)
masked_ids, target_ids, mask = masker.forward(token_ids, key_ids)
```

`MaskingStrategy` applies all three strategies (token, field, event masking) according
to the probabilities in `config`. The strategy mixture is controlled by `PRAGMAConfig`.

---

## Checkpointing

Checkpoints are saved to `--output-dir`. Never commit checkpoint files
to git (covered by `.gitignore`). Model weights go in `checkpoints/`.
