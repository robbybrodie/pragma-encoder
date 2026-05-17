# Table 1 — PRAGMA Model Family

Extracted exactly from the paper.

Paper: "PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1, Table 1 (page 6)

---

## Table 1 (exact)

PRAGMA scales across three variants (10 M, 100 M, 1 B parameters) by jointly
increasing model width (d_model, d_ffn), depth of the profile-state, event, and
history encoders, and the number of attention heads.

| Model | Params | d_model | d_ffn | Profile layers | Event layers | History layers | Heads |
|-------|--------|---------|-------|----------------|--------------|----------------|-------|
| PRAGMA-S | 10 M | 192 | 768 | 1 | 5 | 2 | 3 |
| PRAGMA-M | 100 M | 512 | 2048 | 3 | 16 | 6 | 8 |
| PRAGMA-L | 1 B | 1024 | 4096 | 9 | 45 | 18 | 16 |

---

## Derived Invariants (confirmed for all variants)

These invariants hold across all three variants and are tested in `tests/test_config.py`:

| Invariant | Value | Verification |
|-----------|-------|-------------|
| d_ffn = 4 × d_model | 768=4×192, 2048=4×512, 4096=4×1024 | `test_ffn_is_four_times_dmodel` |
| d_model % n_heads == 0 | 192%3=0, 512%8=0, 1024%16=0 | `test_heads_divide_dmodel_evenly` |
| head_dim = d_model / n_heads | 192/3=64, 512/8=64, 1024/16=64 | `test_head_dimension_is_64` |

---

## All Variants Share

From Section 2.3 (applies to all model sizes):

- **Activation:** GELU (Hendrycks et al., 2016)
- **Normalisation:** pre-norm LayerNorm (Xiong et al., 2020)
- **Dropout:** 0.1 (Srivastava et al., 2014)
- **Architecture:** encoder-only bidirectional Transformer
- **Three encoders:** ProfileStateEncoder, EventEncoder, HistoryEncoder

---

## Implementation

```python
# src/model/config.py

@classmethod
def pragma_s(cls) -> "PRAGMAConfig":
    return cls(
        model_name="pragma-s",
        d_model=192,
        d_ffn=768,
        n_heads=3,
        profile_encoder_layers=1,
        event_encoder_layers=5,
        history_encoder_layers=2,
        # ... other params
    )

@classmethod
def pragma_m(cls) -> "PRAGMAConfig":
    return cls(
        model_name="pragma-m",
        d_model=512,
        d_ffn=2048,
        n_heads=8,
        profile_encoder_layers=3,
        event_encoder_layers=16,
        history_encoder_layers=6,
        # ...
    )

@classmethod
def pragma_l(cls) -> "PRAGMAConfig":
    return cls(
        model_name="pragma-l",
        d_model=1024,
        d_ffn=4096,
        n_heads=16,
        profile_encoder_layers=9,
        event_encoder_layers=45,
        history_encoder_layers=18,
        # ...
    )
```

**Test file:** `tests/test_config.py::TestPaperSpecifications`

Values used in tests:
- `d_model=192` → `used in: tests/test_config.py::TestPaperSpecifications::test_pragma_s_table_1`
- `d_ffn=768` → `used in: tests/test_config.py::TestPaperSpecifications::test_pragma_s_table_1`
- `n_heads=3` → `used in: tests/test_config.py::TestPaperSpecifications::test_pragma_s_table_1`
- `profile_encoder_layers=1` → `used in: tests/test_config.py::TestPaperSpecifications::test_pragma_s_table_1`
- `event_encoder_layers=5` → `used in: tests/test_config.py::TestPaperSpecifications::test_pragma_s_table_1`
- `history_encoder_layers=2` → `used in: tests/test_config.py::TestPaperSpecifications::test_pragma_s_table_1`

---

## Implementation Notes

Table 1 is the single source of truth for all architectural parameters.

Every number in this table must be reproduced exactly in `PRAGMAConfig`.
The scaling test in DEVELOPMENT_PROCESS.md requires that switching from
PRAGMA-S to PRAGMA-M requires changing exactly one line:

```python
config = PRAGMAConfig.pragma_s()  # → change to pragma_m()
```

Nothing else in the training code should change.

If anything else needs to change when switching variants,
there are hardcoded architectural values in the code.
Find them. Remove them.
