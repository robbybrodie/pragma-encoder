# Table 2 — Main Results

Extracted from the paper.

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Table 2 (page 9)

---

## Table 2 (exact)

PRAGMA significantly outperforms internal task-specific models while sharing most
of the parameters across tasks.

Relative performance computed as: `(PRAGMA / baseline − 1)`

Model used: PRAGMA-L with LoRA fine-tuning.

| Task | Metric | Baseline (ref.) | PRAGMA |
|------|--------|-----------------|--------|
| Credit scoring | PR-AUC | — | +130.2% |
| Credit scoring | ROC-AUC | — | +12.4% |
| Communication engagement | PR-AUC | — | +79.4% |
| Communication engagement | ROC-AUC | — | +20.4% |
| External fraud | Precision | — | +16.7% |
| External fraud | Recall | — | +64.7% |
| Product recommendation | mAP | — | +40.5% |
| Recurrent transactions | F1 | — | +5.8% |
| Lifetime value | PR-AUC | — | +1.8% |
| Lifetime value | ROC-AUC | — | +2.6% |

*Absolute metrics are commercially sensitive and not reported.*

---

## Implementation Notes

These numbers are reference only — they are not used in unit tests.

They are the benchmark that the end-to-end comparison test in
DEVELOPMENT_PROCESS.md targets. After training PRAGMA-S on TabFormer:

> PRAGMA-S embedding AUC > NVIDIA blueprint embedding AUC

This is the minimum bar. The full PRAGMA-L results in Table 2
represent the aspirational target.

The comparison test is run:
- After first complete implementation
- After any architectural change
- Before any demo or presentation

Reference: `scripts/evaluate_pragma.py`
