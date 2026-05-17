# Section 3 — Evaluation (Overview)

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 3

---

## Reporting Convention

For commercial sensitivity reasons, **absolute metrics are not reported**.
All results expressed as relative changes:

$$\text{relative improvement} = \left(\frac{x}{\text{baseline}} - 1\right) \times 100\%$$

---

## Evaluation Structure

| Section | Content |
|---------|---------|
| 3.1 | Evaluation Protocol (embedding probe + LoRA) |
| 3.2 | Downstream Tasks (6 tasks described) |
| 3.3 | Main Results (Table 2) |
| 3.4 | Additional Experiments and Ablations |

---

## Implementation Notes

The end-to-end comparison test from DEVELOPMENT_PROCESS.md uses the evaluation
framework described in this section.

See the sub-sections for details:
- `docs/paper/03-01-protocol.md` — evaluation methodology
- `docs/paper/03-02-tasks.md` — the 6 downstream tasks
- `docs/paper/03-03-results.md` — main results (Table 2)
- `docs/paper/03-04-ablations.md` — ablation studies
