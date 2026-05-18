"""Tests for event_valid masking in PRAGMA.forward().

event_valid is an optional (batch, ne) bool tensor.  True = real event,
False = padded.  When provided, ze for padded events is zeroed before
it enters the HistoryEncoder so padding noise cannot leak into the
history representation.

These tests are written BEFORE the implementation.  They currently fail
with TypeError (unexpected keyword argument).  Run them again after
adding event_valid to PRAGMA.forward() to confirm correctness.
"""

import torch
import pytest

from src.model import PRAGMA, PRAGMAConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg():
    """Return a tiny config so tests run quickly without a GPU."""
    return PRAGMAConfig(
        d_model=64,
        d_ffn=256,
        n_heads=4,
        profile_encoder_layers=1,
        event_encoder_layers=1,
        history_encoder_layers=1,
        max_events=16,
        max_event_tokens=8,
        key_vocab_size=64,
        value_vocab_size=128,
        dropout=0.0,
    )


def _make_inputs(config, batch=2, ne=4):
    """Return minimal float tensors matching PRAGMA.forward() contract."""
    d  = config.d_model
    na = 1
    ni = config.max_event_tokens
    return dict(
        xa=torch.randn(batch, na, d),
        ta=torch.zeros(batch, na),
        xe=torch.randn(batch, ne, ni, d),
        xt=torch.zeros(batch, ne, 3, dtype=torch.long),
        te=torch.zeros(batch, 1 + ne),
    )


@pytest.fixture(scope="module")
def model():
    cfg = _cfg()
    m = PRAGMA(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Test 1 — backward compatibility
# ---------------------------------------------------------------------------

def test_event_valid_none_is_backward_compatible(model):
    """forward() without event_valid must return the same keys as before."""
    d  = model.config.d_model
    ne = 4
    inputs = _make_inputs(model.config, ne=ne)
    with torch.no_grad():
        out = model.forward(**inputs)
    assert "zh" in out
    # [USR] at pos 0 + ne event positions
    assert out["zh"].shape == (inputs["xa"].shape[0], 1 + ne, d)
    assert "logits" not in out


# ---------------------------------------------------------------------------
# Test 2 — all-True event_valid matches no event_valid
# ---------------------------------------------------------------------------

def test_event_valid_all_true_matches_no_event_valid(model):
    """event_valid=all-True should give bit-identical zh to event_valid=None."""
    ne = 4
    inputs = _make_inputs(model.config, ne=ne)
    batch = inputs["xa"].shape[0]
    event_valid_all = torch.ones(batch, ne, dtype=torch.bool)

    with torch.no_grad():
        out_none = model.forward(**inputs)
        out_all  = model.forward(**inputs, event_valid=event_valid_all)

    torch.testing.assert_close(out_none["zh"], out_all["zh"])


# ---------------------------------------------------------------------------
# Test 3 — padding content does not leak through a False event_valid slot
# ---------------------------------------------------------------------------

def test_event_valid_pads_event_rows(model):
    """Content of a False event_valid slot must not affect zh.

    Two forward passes share the same event_valid with slot 2 masked out.
    They differ only in xe[:, 2, :, :] (the padded slot).  zh must be
    identical across both passes because ze[:, 2, :] is zeroed before
    entering HistoryEncoder in both cases.
    """
    ne = 4
    ni = model.config.max_event_tokens
    d  = model.config.d_model
    batch = 1

    inputs_a = _make_inputs(model.config, batch=batch, ne=ne)

    inputs_b = {k: v.clone() for k, v in inputs_a.items()}
    # Inject different content into the padded slot
    inputs_b["xe"] = inputs_a["xe"].clone()
    inputs_b["xe"][0, 2, :, :] = torch.randn(ni, d)

    event_valid = torch.ones(batch, ne, dtype=torch.bool)
    event_valid[0, 2] = False  # slot 2 is padding

    with torch.no_grad():
        out_a = model.forward(**inputs_a, event_valid=event_valid)
        out_b = model.forward(**inputs_b, event_valid=event_valid)

    # Position 3 in zh (= event index 2, 0-based) must be identical
    torch.testing.assert_close(out_a["zh"][:, 3, :], out_b["zh"][:, 3, :])
    # [USR] token and real events should also be identical
    torch.testing.assert_close(out_a["zh"][:, 0, :], out_b["zh"][:, 0, :])
    torch.testing.assert_close(out_a["zh"][:, 1, :], out_b["zh"][:, 1, :])
    torch.testing.assert_close(out_a["zh"][:, 2, :], out_b["zh"][:, 2, :])


# ---------------------------------------------------------------------------
# Test 4 — all events padded: no crash, no NaN/Inf
# ---------------------------------------------------------------------------

def test_event_valid_all_false_no_nan(model):
    """All events padded — zh must still be finite (all-zero ze input is valid)."""
    ne = 4
    batch = 2
    inputs = _make_inputs(model.config, batch=batch, ne=ne)
    event_valid = torch.zeros(batch, ne, dtype=torch.bool)  # all False

    with torch.no_grad():
        out = model.forward(**inputs, event_valid=event_valid)

    assert torch.isfinite(out["zh"]).all(), "zh contains NaN or Inf with all-False event_valid"


# ---------------------------------------------------------------------------
# Test 5 — event_valid + mlm mask: logits shape and finiteness
# ---------------------------------------------------------------------------

def test_event_valid_with_mlm_mask_logits(model):
    """Logit count matches masked real-event tokens; no NaN/Inf in logits."""
    ne = 4
    ni = model.config.max_event_tokens
    batch = 2

    inputs = _make_inputs(model.config, batch=batch, ne=ne)

    # Mark events 2 and 3 as padding
    event_valid = torch.ones(batch, ne, dtype=torch.bool)
    event_valid[:, 2] = False
    event_valid[:, 3] = False

    # Mask one token per real event (events 0 and 1 only)
    mask = torch.zeros(batch, ne, ni, dtype=torch.bool)
    mask[:, 0, 1] = True  # token 1 of event 0
    mask[:, 1, 0] = True  # token 0 of event 1

    # Zero out mask for padded events (matches what train_pragma.py does)
    mask = mask & event_valid.unsqueeze(-1)

    with torch.no_grad():
        out = model.forward(**inputs, event_valid=event_valid, mask=mask)

    assert "logits" in out
    expected_n_masked = mask.sum().item()
    assert out["logits"].shape == (expected_n_masked, model.config.value_vocab_size), (
        f"Expected logits shape ({expected_n_masked}, {model.config.value_vocab_size}), "
        f"got {out['logits'].shape}"
    )
    assert torch.isfinite(out["logits"]).all(), "logits contain NaN or Inf"
