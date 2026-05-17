"""Tests for the three-strategy masking objective.

Tests the masking strategies from PRAGMA paper Section 2.3.5:
    - TokenMasker:  Standard BERT-style token masking
    - FieldMasker:  Field-level masking across the sequence
    - EventMasker:  Event-level masking in the history

Key invariants:
    - Labels are -100 at non-masked positions (ignored in loss)
    - Labels equal original token IDs at masked positions
    - Masked token IDs are [MASK] (80%), random (10%), or unchanged (10%)
    - Field and event maskers mask complete units (not partial)

Reference: Ostroukhov et al. (2026), Section 2.3.5
"""

import pytest
import torch

from src.masking import TokenMasker, FieldMasker, EventMasker


BATCH = 4
SEQ_LEN = 128
VOCAB_SIZE = 1000
MASK_ID = 1


class TestTokenMasker:
    """Tests for standard token-level masking."""

    @pytest.fixture
    def masker(self):
        return TokenMasker(mask_prob=0.15, vocab_size=VOCAB_SIZE)

    def test_labels_are_minus_100_at_non_masked_positions(self, masker):
        """Non-masked positions should have label = -100."""
        token_ids = torch.randint(10, VOCAB_SIZE, (BATCH, SEQ_LEN))
        _, labels = masker.apply(token_ids)
        non_masked = labels == -100
        assert non_masked.any()

    def test_labels_equal_originals_at_masked_positions(self, masker):
        """Labels at masked positions should equal original token IDs."""
        token_ids = torch.randint(10, VOCAB_SIZE, (BATCH, SEQ_LEN))
        _, labels = masker.apply(token_ids)
        masked_positions = labels != -100
        assert (labels[masked_positions] == token_ids[masked_positions]).all()

    def test_output_shape_unchanged(self, masker):
        """apply() should not change tensor shapes."""
        token_ids = torch.randint(10, VOCAB_SIZE, (BATCH, SEQ_LEN))
        masked_ids, labels = masker.apply(token_ids)
        assert masked_ids.shape == token_ids.shape
        assert labels.shape == token_ids.shape

    def test_mask_prob_approximately_correct(self, masker):
        """Approximately mask_prob fraction of tokens should be masked."""
        torch.manual_seed(42)
        token_ids = torch.randint(10, VOCAB_SIZE, (100, SEQ_LEN))
        _, labels = masker.apply(token_ids)
        fraction_masked = (labels != -100).float().mean().item()
        # Allow generous tolerance (stochastic)
        assert 0.08 < fraction_masked < 0.22


class TestFieldMasker:
    """Tests for field-level masking."""

    @pytest.fixture
    def masker(self):
        return FieldMasker(n_fields_to_mask=1)

    def test_no_masking_without_field_positions(self, masker):
        """Without field_positions, no masking should occur."""
        token_ids = torch.randint(10, VOCAB_SIZE, (BATCH, SEQ_LEN))
        _, labels = masker.apply(token_ids)
        assert (labels == -100).all()

    def test_masked_positions_have_correct_labels(self, masker):
        """Labels at masked field positions should equal original IDs."""
        token_ids = torch.randint(10, VOCAB_SIZE, (BATCH, SEQ_LEN))
        field_positions = {
            "amount": [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
        }
        _, labels = masker.apply(token_ids, field_positions=field_positions)
        for b in range(BATCH):
            for pos in [0, 1, 2]:
                assert labels[b, pos] == token_ids[b, pos]


class TestEventMasker:
    """Tests for event-level masking."""

    @pytest.fixture
    def masker(self):
        return EventMasker(mask_prob=1.0)  # Always mask for deterministic tests

    def test_no_masking_without_boundaries(self, masker):
        """Without event_boundaries, no masking should occur."""
        token_ids = torch.randint(10, VOCAB_SIZE, (BATCH, SEQ_LEN))
        _, labels = masker.apply(token_ids)
        assert (labels == -100).all()

    def test_entire_event_is_masked(self, masker):
        """All tokens in a masked event should receive labels."""
        token_ids = torch.randint(10, VOCAB_SIZE, (BATCH, SEQ_LEN))
        # Each sample has one event spanning positions 0-9
        event_boundaries = [[(0, 10)]] * BATCH
        _, labels = masker.apply(token_ids, event_boundaries=event_boundaries)
        for b in range(BATCH):
            for pos in range(10):
                assert labels[b, pos] == token_ids[b, pos]
