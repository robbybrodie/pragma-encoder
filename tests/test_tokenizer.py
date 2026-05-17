"""Tests for the PRAGMA tokenisation module.

Tests the key-value-time tokenisation scheme from Section 2.2:
    - NumericalTokenizer: percentile bucket correctness
    - CategoricalTokenizer: vocabulary building and UNK handling
    - TextualTokenizer: BPE encoding
    - TemporalTokenizer: log-seconds and calendar token correctness
    - TokenizerPipeline: end-to-end event encoding

Reference: Ostroukhov et al. (2026), Section 2.2
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from src.tokenizer import (
    NumericalTokenizer,
    CategoricalTokenizer,
    TemporalTokenizer,
    TokenizerPipeline,
)
from src.tokenizer.base import BaseTokenizer


class TestNumericalTokenizer:
    """Tests for NumericalTokenizer (percentile bucketing)."""

    def test_fit_and_encode_returns_valid_bucket(self):
        """Encoded values should fall within [0, n_buckets - 1]."""
        tokenizer = NumericalTokenizer(n_buckets=10)
        data = list(range(100))
        tokenizer.fit(data)
        for v in [0, 25, 50, 75, 99]:
            bucket = tokenizer.encode(v)
            assert 0 <= bucket < tokenizer.n_buckets

    def test_missing_value_returns_special_token(self):
        """None should return n_buckets (the MISSING token)."""
        tokenizer = NumericalTokenizer(n_buckets=10)
        tokenizer.fit(list(range(100)))
        assert tokenizer.encode(None) == tokenizer.n_buckets

    def test_vocab_size(self):
        """vocab_size should be n_buckets + 1 (for MISSING)."""
        tokenizer = NumericalTokenizer(n_buckets=50)
        tokenizer.fit(list(range(100)))
        assert tokenizer.vocab_size == 51

    def test_encode_before_fit_raises(self):
        """encode() before fit() should raise RuntimeError."""
        tokenizer = NumericalTokenizer(n_buckets=10)
        with pytest.raises(RuntimeError):
            tokenizer.encode(5.0)


class TestCategoricalTokenizer:
    """Tests for CategoricalTokenizer."""

    def test_fit_builds_vocabulary(self):
        """Vocabulary should include all categories seen during fit."""
        tokenizer = CategoricalTokenizer()
        tokenizer.fit(["A", "B", "C", "A", "B"])
        assert tokenizer.encode("A") != tokenizer.encode("B")

    def test_unseen_token_maps_to_unk(self):
        """Unseen categories should map to the UNK token ID."""
        tokenizer = CategoricalTokenizer()
        tokenizer.fit(["A", "B"])
        unk_id = tokenizer.encode("UNSEEN_CATEGORY")
        assert unk_id == tokenizer._token2id[CategoricalTokenizer.UNK]

    def test_none_maps_to_missing(self):
        """None should map to the MISSING token ID."""
        tokenizer = CategoricalTokenizer()
        tokenizer.fit(["A", "B"])
        missing_id = tokenizer.encode(None)
        assert missing_id == tokenizer._token2id[CategoricalTokenizer.MISSING]

    def test_decode_round_trip(self):
        """decode(encode(v)) should return v for seen categories."""
        tokenizer = CategoricalTokenizer()
        tokenizer.fit(["GBP", "USD", "EUR"])
        for cat in ["GBP", "USD", "EUR"]:
            assert tokenizer.decode(tokenizer.encode(cat)) == cat


class TestTemporalTokenizer:
    """Tests for TemporalTokenizer (log-seconds + calendar)."""

    def test_encode_returns_six_tokens(self):
        """Temporal encoding should return exactly 6 tokens."""
        tokenizer = TemporalTokenizer()
        dt = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        tokens = tokenizer.encode(dt)
        assert len(tokens) == 6

    def test_none_returns_six_zeros(self):
        """None timestamp should return [0, 0, 0, 0, 0, 0]."""
        tokenizer = TemporalTokenizer()
        assert tokenizer.encode(None) == [0, 0, 0, 0, 0, 0]

    def test_log_bucket_in_range(self):
        """Log-seconds bucket should be in [0, n_log_buckets - 1]."""
        tokenizer = TemporalTokenizer(n_log_buckets=64)
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tokens = tokenizer.encode(dt)
        assert 0 <= tokens[0] < tokenizer.n_log_buckets

    def test_hour_token_correct(self):
        """Hour token should match the datetime's hour."""
        tokenizer = TemporalTokenizer()
        for hour in [0, 6, 12, 18, 23]:
            dt = datetime(2024, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
            tokens = tokenizer.encode(dt)
            assert tokens[1] == hour  # index 1 = hour

    def test_fit_is_noop(self):
        """fit() should return self without error (no training data needed)."""
        tokenizer = TemporalTokenizer()
        result = tokenizer.fit(["ignored"])
        assert result is tokenizer
