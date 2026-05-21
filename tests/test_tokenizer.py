"""Tests for PRAGMATokenizer (TokenizerPipeline and field tokenisers).

Derived from PRAGMA paper Section 2.2:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:    verify encode() output types and token ID ranges
  Math tests:     verify Equation 2, calendar dims, position indexing (Eq 1)
  Gradient tests: N/A — pure Python tokeniser, no trainable parameters
  Spec tests:     verify key_vocab_size=60, value_vocab_size=28000,
                  temporal scale=8, calendar_feature_dims=3

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import math
from datetime import datetime, timezone
from typing import Any, List, Union

import pytest

from pragma_encoder.model.config import PRAGMAConfig
from pragma_encoder.tokenizer.base import BaseTokenizer
from pragma_encoder.tokenizer.categorical import CategoricalTokenizer
from pragma_encoder.tokenizer.numerical import NumericalTokenizer
from pragma_encoder.tokenizer.pipeline import TokenizerPipeline
from pragma_encoder.tokenizer.temporal import TemporalTokenizer

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

# Known datetime used across calendar tests.
# 2024-03-15 14:30:00 UTC — Friday (weekday=4), day=15, hour=14.
_TS = datetime(2024, 3, 15, 14, 30, 0, tzinfo=timezone.utc)


class _MultiTokenMock(BaseTokenizer):
    """Returns [0, 1, 2] for any value — simulates a 3-token textual field.

    Used to test position indexing (Eq 1, §2.2) without requiring the
    external `tokenizers` library for BPE.
    """

    def fit(self, data: List[Any]) -> "_MultiTokenMock":
        return self

    def encode(self, value: Any) -> List[int]:
        return [0, 1, 2]

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        return "mock"

    @property
    def vocab_size(self) -> int:
        return 3


def _make_simple_pipeline() -> TokenizerPipeline:
    """Pipeline with two fitted field tokenisers: amount (numerical) and
    currency (categorical). Sufficient for shape and range tests."""
    amount = NumericalTokenizer(n_buckets=10)
    amount.fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    currency = CategoricalTokenizer()
    currency.fit(["GBP", "EUR", "USD"])

    return TokenizerPipeline(
        field_tokenizers={"amount": amount, "currency": currency}
    )


def _make_multi_token_pipeline() -> TokenizerPipeline:
    """Pipeline with one single-token field (currency) and one multi-token
    mock field (description). Used to test Equation 1 position indexing."""
    currency = CategoricalTokenizer()
    currency.fit(["GBP"])

    return TokenizerPipeline(
        field_tokenizers={
            "currency": currency,              # 1 token per value
            "description": _MultiTokenMock(),  # 3 tokens per value
        }
    )


# ---------------------------------------------------------------------------
# TestShapes — verify encode() output types and token ID ranges
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify that tokenisers return correct output types (§2.2)."""

    def test_numerical_encode_returns_single_int(self) -> None:
        """§2.2: numerical value tokeniser → exactly one token ID (int)."""
        tok = NumericalTokenizer(n_buckets=10)
        tok.fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = tok.encode(5.0)
        assert isinstance(result, int), (
            f"NumericalTokenizer.encode() must return int, got {type(result)}"
        )

    def test_categorical_encode_returns_single_int(self) -> None:
        """§2.2: categorical value tokeniser → exactly one token ID (int)."""
        tok = CategoricalTokenizer()
        tok.fit(["GBP", "EUR", "USD"])
        result = tok.encode("GBP")
        assert isinstance(result, int), (
            f"CategoricalTokenizer.encode() must return int, got {type(result)}"
        )

    def test_textual_encode_returns_list_of_ints(self) -> None:
        """§2.2: textual value tokeniser → list of ints (BPE subwords)."""
        pytest.importorskip("tokenizers", reason="tokenizers library not installed")
        from pragma_encoder.tokenizer.textual import TextualTokenizer

        tok = TextualTokenizer(vocab_size=100, max_length=16)
        corpus = [
            "metal plan", "PayPal transfer",
            "contactless payment", "AMZN purchase",
        ]
        tok.fit(corpus)
        result = tok.encode("metal plan")
        assert isinstance(result, list), (
            f"TextualTokenizer.encode() must return list, got {type(result)}"
        )
        assert len(result) > 0, "TextualTokenizer.encode() must return ≥1 token"
        assert all(isinstance(i, int) for i in result), (
            "All tokens from TextualTokenizer.encode() must be ints"
        )

    def test_key_ids_within_key_vocab_range(self) -> None:
        """§2.2: key token IDs must fall within the pipeline's key token range."""
        pipeline = _make_simple_pipeline()
        fields = [("amount", 5.0, _TS), ("currency", "GBP", _TS)]
        output = pipeline.encode_event(fields, t_seconds=0.0)

        n_fields = len(pipeline.field_tokenizers)
        key_start = pipeline.N_SPECIAL_TOKENS
        key_end = key_start + n_fields

        for kid in output.key_ids:
            assert key_start <= kid < key_end, (
                f"key_id {kid} outside key range [{key_start}, {key_end})"
            )

    def test_value_ids_within_value_vocab_range(self) -> None:
        """§2.2: value token IDs must fall within the pipeline's value range."""
        pipeline = _make_simple_pipeline()
        fields = [("amount", 5.0, _TS), ("currency", "GBP", _TS)]
        output = pipeline.encode_event(fields, t_seconds=0.0)

        n_fields = len(pipeline.field_tokenizers)
        value_start = pipeline.N_SPECIAL_TOKENS + n_fields
        value_end = pipeline.total_vocab_size

        for vid in output.value_ids:
            assert value_start <= vid < value_end, (
                f"value_id {vid} outside value range [{value_start}, {value_end})"
            )


# ---------------------------------------------------------------------------
# TestMathProperties — verify Equation 2, calendar dims, position indexing
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from the paper (§2.2, Eq 1, Eq 2)."""

    def test_temporal_transform_at_zero(self) -> None:
        """Equation 2: t'(0) = 8·ln(1 + 0/8) = 0.

        key-numbers.md: temporal_transform = 8·ln(1+t/8), §2.2
        """
        tok = TemporalTokenizer()
        result = tok.compute_temporal_coordinate(0.0)
        assert result == 0.0, (
            f"Equation 2: t'(0) must be 0.0, got {result}"
        )

    def test_temporal_transform_at_t_equals_8(self) -> None:
        """Equation 2: t'(8) = 8·ln(1 + 8/8) = 8·ln(2) ≈ 5.5452.

        key-numbers.md: temporal_transform = 8·ln(1+t/8), §2.2
        """
        tok = TemporalTokenizer()
        result = tok.compute_temporal_coordinate(8.0)
        expected = 8.0 * math.log(1.0 + 8.0 / 8.0)  # = 8·ln(2)
        assert abs(result - expected) < 1e-9, (
            f"Equation 2: t'(8) must be {expected:.6f}, got {result:.6f}"
        )

    def test_temporal_transform_compression(self) -> None:
        """Equation 2: increment from t=0→8 exceeds increment from t=8→16.

        The log transform compresses large temporal gaps — the property that
        makes it suitable as input to RoPE positional encoding (§2.2).
        """
        tok = TemporalTokenizer()
        inc_early = (
            tok.compute_temporal_coordinate(8.0)
            - tok.compute_temporal_coordinate(0.0)
        )
        inc_late = (
            tok.compute_temporal_coordinate(16.0)
            - tok.compute_temporal_coordinate(8.0)
        )
        assert inc_early > inc_late, (
            f"Equation 2 log compression: increment 0→8 ({inc_early:.4f}) "
            f"must exceed increment 8→16 ({inc_late:.4f})"
        )

    def test_calendar_features_exactly_three_dims(self) -> None:
        """§2.2: calendar features are [hour, day_of_week, day_of_month] — 3 values.

        key-numbers.md: calendar_feature_dims = 3, §2.2
        Month and quarter are NOT in the paper.
        """
        tok = TemporalTokenizer()
        features = tok.extract_calendar_features(_TS)
        assert len(features) == 3, (
            f"§2.2: calendar features must have exactly 3 dims, got {len(features)}"
        )

    def test_calendar_hour_range(self) -> None:
        """§2.2: hour ∈ [0, 23]."""
        tok = TemporalTokenizer()
        features = tok.extract_calendar_features(_TS)
        hour = features[0]
        assert 0 <= hour <= 23, f"hour must be in [0, 23], got {hour}"
        assert hour == 14.0, f"_TS has hour=14, got {hour}"

    def test_calendar_day_of_week_range(self) -> None:
        """§2.2: day_of_week ∈ [0, 6] (0=Monday, 6=Sunday)."""
        tok = TemporalTokenizer()
        features = tok.extract_calendar_features(_TS)
        dow = features[1]
        assert 0 <= dow <= 6, f"day_of_week must be in [0, 6], got {dow}"
        assert dow == 4.0, f"_TS is a Friday (weekday=4), got {dow}"

    def test_calendar_day_of_month_range(self) -> None:
        """§2.2: day_of_month ∈ [1, 31]."""
        tok = TemporalTokenizer()
        features = tok.extract_calendar_features(_TS)
        dom = features[2]
        assert 1 <= dom <= 31, f"day_of_month must be in [1, 31], got {dom}"
        assert dom == 15.0, f"_TS has day=15, got {dom}"

    def test_position_indexing_restarts_per_field(self) -> None:
        """§2.2 / Equation 1: position counter resets to 0 at each new field.

        'Currency: GBP'      → positions [0]
        'Description: x y z' → positions [0, 1, 2]  ← restarts at 0
        """
        pipeline = _make_multi_token_pipeline()
        fields = [
            ("currency", "GBP", _TS),         # 1 token  → position [0]
            ("description", "anything", _TS),  # 3 tokens → positions [0, 1, 2]
        ]
        output = pipeline.encode_event(fields, t_seconds=0.0)

        assert output.position_ids[0] == 0, (
            "First field (currency) must start at position 0"
        )
        assert output.position_ids[1] == 0, (
            "Second field (description) must RESTART at position 0 (Eq 1)"
        )
        assert output.position_ids[2] == 1, "description: second token at position 1"
        assert output.position_ids[3] == 2, "description: third token at position 2"

    def test_key_token_replicated_per_value_token(self) -> None:
        """§2.2 / Equation 1: key token appears once per value token.

        A 3-token description field must produce 3 key tokens + 3 value tokens.
        """
        pipeline = _make_multi_token_pipeline()
        fields = [("description", "anything", _TS)]
        output = pipeline.encode_event(fields, t_seconds=0.0)

        desc_key_id = pipeline._key_token_ids["description"]
        assert len(output.key_ids) == 3, (
            f"3-token field must produce 3 key tokens, got {len(output.key_ids)}"
        )
        assert all(k == desc_key_id for k in output.key_ids), (
            "All key tokens must be the description key ID (replicated, Eq 1)"
        )
        assert len(output.value_ids) == 3, (
            f"3-token field must produce 3 value tokens, got {len(output.value_ids)}"
        )

    def test_numerical_zero_gets_dedicated_bucket(self) -> None:
        """§2.2: exactly-zero values get a dedicated bucket separate from
        the percentile bucket that small non-zero values fall into."""
        tok = NumericalTokenizer(n_buckets=10)
        tok.fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        zero_id = tok.encode(0.0)
        nonzero_id = tok.encode(1.0)

        assert zero_id == tok.ZERO_ID, (
            f"encode(0.0) must return ZERO_ID ({tok.ZERO_ID}), got {zero_id}"
        )
        assert zero_id != nonzero_id, (
            f"encode(0.0) ({zero_id}) must differ from encode(1.0) ({nonzero_id})"
        )
        assert tok.encode(0.0) == zero_id, "Zero encoding must be deterministic"


# ---------------------------------------------------------------------------
# TestGradientFlow — N/A for pure Python tokeniser
# ---------------------------------------------------------------------------
# PRAGMATokenizer is a pure Python component with no trainable parameters.
# Gradient flow tests are not applicable.


# ---------------------------------------------------------------------------
# TestPaperSpecifications — verify exact values from key-numbers.md
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact values from the paper (all sourced from key-numbers.md)."""

    def test_key_vocab_size_matches_config(self) -> None:
        """§2.2, key-numbers.md: key_vocab_size = 60.

        The config documents the paper's target vocabulary size for field types.
        """
        config = PRAGMAConfig.pragma_s()
        assert config.key_vocab_size == 60, (  # key-numbers.md: §2.2
            f"key_vocab_size must be 60 (§2.2), got {config.key_vocab_size}"
        )

    def test_value_vocab_size_matches_config(self) -> None:
        """§2.2, key-numbers.md: value_vocab_size = 28,000.

        The config documents the paper's target vocabulary size for values.
        """
        config = PRAGMAConfig.pragma_s()
        assert config.value_vocab_size == 28_000, (  # key-numbers.md: §2.2
            f"value_vocab_size must be 28,000 (§2.2), got {config.value_vocab_size}"
        )

    def test_temporal_transform_scale_is_8(self) -> None:
        """key-numbers.md: temporal_transform_scale = 8, §2.2.

        Verified by: t'(8) = 8·ln(2). Dividing by ln(2) recovers scale = 8.
        """
        tok = TemporalTokenizer()
        t_prime_at_8 = tok.compute_temporal_coordinate(8.0)
        inferred_scale = t_prime_at_8 / math.log(2.0)
        assert abs(inferred_scale - 8.0) < 1e-9, (
            f"key-numbers.md: temporal scale must be 8, inferred {inferred_scale:.6f}"
        )

    def test_calendar_feature_dims_is_3(self) -> None:
        """key-numbers.md: calendar_feature_dims = 3, §2.2.

        Exactly 3 calendar features: hour, day_of_week, day_of_month.
        Month and quarter are NOT in the paper.
        """
        assert TemporalTokenizer.N_CALENDAR_DIMS == 3, (  # key-numbers.md: §2.2
            "key-numbers.md: calendar_feature_dims = 3 (§2.2)"
        )
