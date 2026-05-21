"""End-to-end smoke test: assembler → PRAGMA → MLM loss.

Derived from PRAGMA paper Section 2.3 and DEVELOPMENT_PROCESS.md.
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

This test runs the complete pipeline on synthetic data.
No GPU. No real transactions. No real training.
Proves all components integrate without errors.

Reference: DEVELOPMENT_PROCESS.md — comparison test
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pragma_encoder.masking import MaskingStrategy
from pragma_encoder.model import PRAGMA, PRAGMAConfig
from pragma_encoder.model.assembler import EmbeddingAssembler
from pragma_encoder.tokenizer.vocabulary import VocabularySpec


def test_full_pipeline_smoke():
    """End-to-end smoke test: assembler → PRAGMA → MLM loss.

    Uses synthetic data in valid ID ranges.
    Proves the full integration works without errors.
    Does NOT verify meaningful convergence — that requires
    real data and real training (Issue 11 Part B).

    Reference: DEVELOPMENT_PROCESS.md — comparison test
    """
    config = PRAGMAConfig.pragma_s()

    vocab_spec = VocabularySpec(
        special_tokens={"PAD": 0, "MASK": 1, "EVT": 2, "SEP": 3},
        key_start=4,
        key_size=config.key_vocab_size,
        value_start=4 + config.key_vocab_size,
        value_size=config.value_vocab_size,
        total_embedding_vocab_size=4 + config.key_vocab_size
                                     + config.value_vocab_size,
        field_key_ids={},
        field_value_ranges={},
    )

    assembler = EmbeddingAssembler(vocab_spec, config)
    model = PRAGMA(config)
    masker = MaskingStrategy(config)

    model.eval()
    assembler.eval()

    batch, ne, ni, na = 2, 5, 8, 6

    # Synthetic event tokens — valid value IDs
    xe_val_ids = torch.randint(
        vocab_spec.value_start,
        vocab_spec.value_start + config.value_vocab_size,
        (batch, ne, ni),
    )
    xe_key_ids = torch.randint(
        vocab_spec.key_start,
        vocab_spec.key_start + config.key_vocab_size,
        (batch, ne, ni),
    )
    xe_pos_ids = torch.arange(ni).unsqueeze(0).unsqueeze(0).expand(batch, ne, -1)

    # Synthetic profile tokens — valid value IDs
    xa_val_ids = torch.randint(
        vocab_spec.value_start,
        vocab_spec.value_start + config.value_vocab_size,
        (batch, na),
    )
    xa_key_ids = torch.randint(
        vocab_spec.key_start,
        vocab_spec.key_start + config.key_vocab_size,
        (batch, na),
    )
    xa_pos_ids = torch.arange(na).unsqueeze(0).expand(batch, -1)

    # Temporal coordinates
    ta = torch.zeros(batch, na)
    te_usr = torch.zeros(batch, 1)
    te_events = torch.rand(batch, ne) * 100.0
    te = torch.cat([te_usr, te_events], dim=1)

    # Calendar features
    xt = torch.stack([
        torch.randint(0, 24, (batch, ne)),
        torch.randint(0, 7,  (batch, ne)),
        torch.randint(1, 32, (batch, ne)),
    ], dim=-1)

    # Apply masking
    masked_val_ids, target_ids, mlm_mask = masker.forward(
        xe_val_ids, xe_key_ids
    )

    # Assemble
    assembled = assembler.forward(
        xa_key_ids=xa_key_ids,
        xa_val_ids=xa_val_ids,
        xa_pos_ids=xa_pos_ids,
        ta=ta,
        xe_key_ids=xe_key_ids,
        xe_val_ids=masked_val_ids,
        xe_pos_ids=xe_pos_ids,
        xt=xt,
        te=te,
        target_ids=xe_val_ids,
        mask=mlm_mask,
    )

    # Validate assembler invariants
    assembled.validate(config)

    # PRAGMA forward pass
    with torch.no_grad():
        output = model.forward(
            xa=assembled.xa,
            ta=assembled.ta,
            xe=assembled.xe,
            xt=assembled.xt,
            te=assembled.te,
            mask=assembled.mlm_mask,
        )

    # Verify outputs
    assert "zh" in output
    assert output["zh"].shape == (batch, 1 + ne, config.d_model)
    assert not torch.isnan(output["zh"]).any(), "NaN in zh"

    if "logits" in output:
        assert output["logits"].shape[1] == config.value_vocab_size
        assert not torch.isnan(output["logits"]).any(), "NaN in logits"

    # Verify loss is computable
    if "logits" in output and assembled.mlm_mask.any():
        valid_targets = assembled.targets[assembled.mlm_mask]
        loss = model.mlm_head.compute_loss(
            output["logits"],
            valid_targets,
        )
        assert loss.item() > 0,         "Loss should be positive"
        assert not torch.isnan(loss),    "Loss is NaN"
        assert not torch.isinf(loss),    "Loss is infinite"
