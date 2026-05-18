# Section 3.3 — Main Results

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 3.3

---

## Summary

PRAGMA (PRAGMA-L with LoRA) consistently outperforms task-specific baselines
across nearly all evaluated domains, while sharing most parameters across tasks.

See `docs/paper/table-2-main-results.md` for the full table.

---

## Key Findings

**Strongest improvements in precision-recall metrics:**
- Credit Scoring PR-AUC: **+130.2%**
- Communication Engagement PR-AUC: **+79.4%**

These large PR-AUC gains suggest PRAGMA is particularly effective at identifying
low-frequency, high-value signals where traditional models struggle.

**More modest but consistent improvements:**
- Lifetime Value: +1.8% PR-AUC, +2.6% ROC-AUC
- Recurrent Transactions: +5.8% F1

**Scale matters for harder tasks:**

Section 3.3.1 (Effect of Model Scale):
- Larger models yield substantially larger gains on harder tasks (Credit Scoring)
- Smaller models already competitive for easier tasks (LTV, Recurrent Transactions)
- Practical trade-off: PRAGMA-S competitive on many tasks with much lower compute cost

---

## 3.3.2 Effect of Pre-training

LoRA fine-tuning of a pre-trained backbone consistently matches or outperforms
full training from scratch.

---

## Implementation Notes

These results are the benchmark this implementation must approach.

The minimum bar (DEVELOPMENT_PROCESS.md comparison test):
> PRAGMA-S embedding AUC > NVIDIA blueprint embedding AUC on TabFormer fraud detection

If this does not hold after 1,000+ training steps, something is architecturally wrong.
