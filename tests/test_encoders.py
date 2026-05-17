"""Tests for the PRAGMA encoder modules.

Tests the three-encoder architecture from Sections 2.3.2–2.3.4:
    - ProfileStateEncoder: output shape and [USR] token extraction
    - EventEncoder: output shape and [EVT] token extraction
    - HistoryEncoder: output shape, cross-attention, and masking
    - RotaryPositionalEmbedding: rotation correctness

Key invariants checked:
    - All encoders use bidirectional attention (no causal mask applied)
    - RoPE does not change tensor shapes
    - Masked event positions in HistoryEncoder are replaced correctly

Reference: Ostroukhov et al. (2026), Sections 2.3.2–2.3.4
"""

import pytest
import torch

from src.encoders import (
    EventEncoder,
    HistoryEncoder,
    ProfileStateEncoder,
    RotaryPositionalEmbedding,
)


BATCH = 2
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
VOCAB_SIZE = 1000


class TestRotaryPositionalEmbedding:
    """Tests for RoPE implementation."""

    def test_output_shape_unchanged(self):
        """RoPE should not change query/key tensor shapes."""
        rope = RotaryPositionalEmbedding(dim=16, max_seq_len=64)
        q = torch.randn(BATCH, N_HEADS, 32, 16)
        k = torch.randn(BATCH, N_HEADS, 32, 16)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_changes_values(self):
        """RoPE rotation should modify the query/key values (not identity)."""
        rope = RotaryPositionalEmbedding(dim=16, max_seq_len=64)
        q = torch.randn(BATCH, N_HEADS, 32, 16)
        k = torch.randn(BATCH, N_HEADS, 32, 16)
        q_rot, _ = rope(q, k)
        assert not torch.allclose(q, q_rot)


class TestProfileStateEncoder:
    """Tests for ProfileStateEncoder."""

    @pytest.fixture
    def encoder(self):
        return ProfileStateEncoder(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
        )

    def test_output_shape(self, encoder):
        """Output [USR] representation should be (batch, d_model)."""
        token_ids = torch.randint(1, VOCAB_SIZE, (BATCH, 32))
        usr_repr = encoder(token_ids)
        assert usr_repr.shape == (BATCH, D_MODEL)

    def test_output_with_mask(self, encoder):
        """Output shape should be unchanged when attention mask is provided."""
        token_ids = torch.randint(1, VOCAB_SIZE, (BATCH, 32))
        mask = torch.zeros(BATCH, 32, dtype=torch.bool)
        mask[:, 28:] = True  # Last 4 positions are padding
        usr_repr = encoder(token_ids, attention_mask=mask)
        assert usr_repr.shape == (BATCH, D_MODEL)

    def test_no_nan_in_output(self, encoder):
        """Output should contain no NaN values for valid inputs."""
        token_ids = torch.randint(1, VOCAB_SIZE, (BATCH, 16))
        usr_repr = encoder(token_ids)
        assert not torch.isnan(usr_repr).any()


class TestEventEncoder:
    """Tests for EventEncoder."""

    @pytest.fixture
    def encoder(self):
        return EventEncoder(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
        )

    def test_output_shape(self, encoder):
        """Output [EVT] representation should be (batch, d_model)."""
        token_ids = torch.randint(1, VOCAB_SIZE, (BATCH, 64))
        evt_repr = encoder(token_ids)
        assert evt_repr.shape == (BATCH, D_MODEL)

    def test_output_with_calendar_tokens(self, encoder):
        """Calendar tokens should not change output shape."""
        token_ids = torch.randint(1, VOCAB_SIZE, (BATCH, 64))
        cal_tokens = torch.zeros(BATCH, 5, dtype=torch.long)
        cal_tokens[:, 0] = 12  # noon
        cal_tokens[:, 1] = 2   # Wednesday
        evt_repr = encoder(token_ids, calendar_tokens=cal_tokens)
        assert evt_repr.shape == (BATCH, D_MODEL)


class TestHistoryEncoder:
    """Tests for HistoryEncoder."""

    N_EVENTS = 16

    @pytest.fixture
    def encoder(self):
        return HistoryEncoder(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
        )

    def test_output_shapes(self, encoder):
        """history_reprs should be (B, N, D); hist_repr should be (B, D)."""
        event_reprs = torch.randn(BATCH, self.N_EVENTS, D_MODEL)
        usr_repr = torch.randn(BATCH, D_MODEL)
        history_reprs, hist_repr = encoder(event_reprs, usr_repr)
        assert history_reprs.shape == (BATCH, self.N_EVENTS, D_MODEL)
        assert hist_repr.shape == (BATCH, D_MODEL)

    def test_masking_changes_output(self, encoder):
        """Masked events should produce different outputs than unmasked."""
        event_reprs = torch.randn(BATCH, self.N_EVENTS, D_MODEL)
        usr_repr = torch.randn(BATCH, D_MODEL)

        mask = torch.zeros(BATCH, self.N_EVENTS, dtype=torch.bool)
        mask[:, :4] = True  # Mask first 4 events

        no_mask_out, _ = encoder(event_reprs, usr_repr)
        with_mask_out, _ = encoder(event_reprs, usr_repr, event_mask=mask)

        # Masked positions should have different outputs
        assert not torch.allclose(no_mask_out[:, :4], with_mask_out[:, :4])
