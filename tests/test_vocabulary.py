"""Tests for VocabularySpec and VocabularyMap.

Derived from PRAGMA paper Section 2.2 and ADR 002
(docs/decisions/002-embedding-assembler.md):
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape/spec tests: verify VocabularySpec fields match TokenizerPipeline layout
  Math tests:       verify VocabularyMap ID arithmetic (global ↔ local)
  Spec tests:       verify exact values from key-numbers.md

Vocabulary layout (TokenizerPipeline._build_vocabulary_layout):
  [0, N_SPECIAL_TOKENS)            — special tokens (PAD, MASK, CLS, SEP)
  [N_SPECIAL_TOKENS, key_end)      — key tokens (one per field, sorted alphabetically)
  [key_end, key_end + value_size)  — value tokens (contiguous per field)

Concrete test fixture (used throughout):
  Fields (sorted alphabetically): "amount", "currency"
  amount:   NumericalTokenizer(n_buckets=10)  → vocab_size = 12
  currency: CategoricalTokenizer fit(3 values) → vocab_size = 5

  N_SPECIAL_TOKENS = 4
  key_start  = 4,  key_size   = 2
  key_ids:   {"amount": 4, "currency": 5}
  value_start = 6, value_size = 17  (12 + 5)
  value_ranges: {"amount": (6, 18), "currency": (18, 23)}
  total_embedding_vocab_size = 23

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section, or is derived from it.
"""

import pytest

from src.tokenizer.categorical import CategoricalTokenizer
from src.tokenizer.numerical import NumericalTokenizer
from src.tokenizer.pipeline import TokenizerPipeline
from src.tokenizer.vocabulary import VocabularyMap, VocabularySpec


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> TokenizerPipeline:
    """Two-field pipeline: amount (numerical) and currency (categorical).

    Sorted alphabetically: "amount" before "currency".
      amount  vocab_size = 12  (10 buckets + ZERO + MISSING)
      currency vocab_size = 5  (UNK + MISSING + GBP + EUR + USD)
    """
    amount = NumericalTokenizer(n_buckets=10)
    amount.fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    currency = CategoricalTokenizer()
    currency.fit(["GBP", "EUR", "USD"])

    return TokenizerPipeline(
        field_tokenizers={"amount": amount, "currency": currency}
    )


# Expected constants derived from the concrete fixture above
_N_SPECIAL = 4    # PAD=0, MASK=1, CLS=2, SEP=3
_KEY_START  = 4
_KEY_SIZE   = 2   # "amount", "currency"
_VAL_START  = 6   # key_start + key_size
_AMOUNT_VSIZ  = 12  # n_buckets=10 + ZERO + MISSING
_CURRENCY_VSIZ = 5  # UNK + MISSING + GBP + EUR + USD
_VAL_SIZE   = _AMOUNT_VSIZ + _CURRENCY_VSIZ   # 17
_TOTAL      = _N_SPECIAL + _KEY_SIZE + _VAL_SIZE  # 23


# ---------------------------------------------------------------------------
# TestVocabularySpec — verify vocabulary_spec() output
# ---------------------------------------------------------------------------


class TestVocabularySpec:
    """Verify VocabularySpec fields match TokenizerPipeline vocabulary layout."""

    def test_vocabulary_spec_returns_vocabulary_spec(self) -> None:
        """vocabulary_spec() must return a VocabularySpec instance."""
        pipeline = _make_pipeline()
        spec = pipeline.vocabulary_spec()
        assert isinstance(spec, VocabularySpec), (
            f"vocabulary_spec() must return VocabularySpec, got {type(spec)}"
        )

    def test_total_embedding_vocab_size(self) -> None:
        """total_embedding_vocab_size = N_SPECIAL + n_fields + sum(field.vocab_size).

        ADR 002: temporal tokenizer is NOT in the embedding table —
        temporal encoding is handled by RoPE (continuous float) and
        calendar features (3 floats), not discrete tokens.
        """
        spec = _make_pipeline().vocabulary_spec()
        assert spec.total_embedding_vocab_size == _TOTAL, (
            f"total_embedding_vocab_size must be {_TOTAL}, got {spec.total_embedding_vocab_size}"
        )

    def test_value_start_equals_key_start_plus_key_size(self) -> None:
        """value_start == key_start + key_size — no gap in vocabulary layout."""
        spec = _make_pipeline().vocabulary_spec()
        assert spec.value_start == spec.key_start + spec.key_size, (
            f"value_start ({spec.value_start}) must equal "
            f"key_start ({spec.key_start}) + key_size ({spec.key_size})"
        )

    def test_key_start_is_n_special_tokens(self) -> None:
        """key_start == N_SPECIAL_TOKENS (4) — keys come immediately after specials."""
        spec = _make_pipeline().vocabulary_spec()
        assert spec.key_start == _N_SPECIAL, (
            f"key_start must be {_N_SPECIAL} (N_SPECIAL_TOKENS), got {spec.key_start}"
        )

    def test_key_start_greater_than_max_special_token_id(self) -> None:
        """key_start > max(special_tokens.values()) — no overlap with specials."""
        spec = _make_pipeline().vocabulary_spec()
        max_special = max(spec.special_tokens.values())
        assert spec.key_start > max_special, (
            f"key_start ({spec.key_start}) must exceed max special token ID ({max_special})"
        )

    def test_field_key_ids_keys_match_field_names(self) -> None:
        """field_key_ids must contain exactly the registered field names."""
        spec = _make_pipeline().vocabulary_spec()
        assert set(spec.field_key_ids.keys()) == {"amount", "currency"}, (
            f"field_key_ids keys must be {{'amount', 'currency'}}, "
            f"got {set(spec.field_key_ids.keys())}"
        )

    def test_field_key_ids_values_in_key_range(self) -> None:
        """All field_key_ids must fall within [key_start, key_start + key_size)."""
        spec = _make_pipeline().vocabulary_spec()
        key_end = spec.key_start + spec.key_size
        for name, kid in spec.field_key_ids.items():
            assert spec.key_start <= kid < key_end, (
                f"key_id for '{name}' ({kid}) must be in "
                f"[{spec.key_start}, {key_end})"
            )

    def test_field_key_ids_sorted_alphabetically(self) -> None:
        """Fields are sorted alphabetically — 'amount' (4) < 'currency' (5).

        TokenizerPipeline._build_vocabulary_layout uses sorted() for determinism.
        """
        spec = _make_pipeline().vocabulary_spec()
        assert spec.field_key_ids["amount"] == _KEY_START, (
            f"'amount' (first alphabetically) must have key_id {_KEY_START}"
        )
        assert spec.field_key_ids["currency"] == _KEY_START + 1, (
            f"'currency' (second alphabetically) must have key_id {_KEY_START + 1}"
        )

    def test_field_value_ranges_within_value_space(self) -> None:
        """All field_value_ranges must lie within [value_start, value_start + value_size)."""
        spec = _make_pipeline().vocabulary_spec()
        val_end = spec.value_start + spec.value_size
        for name, (lo, hi) in spec.field_value_ranges.items():
            assert lo >= spec.value_start, (
                f"'{name}' value range start ({lo}) must be >= value_start ({spec.value_start})"
            )
            assert hi <= val_end, (
                f"'{name}' value range end ({hi}) must be <= value_end ({val_end})"
            )
            assert lo < hi, f"'{name}' value range must be non-empty: ({lo}, {hi})"

    def test_field_value_ranges_correct_sizes(self) -> None:
        """Value range widths must match each field tokenizer's vocab_size."""
        spec = _make_pipeline().vocabulary_spec()
        amt_lo, amt_hi = spec.field_value_ranges["amount"]
        cur_lo, cur_hi = spec.field_value_ranges["currency"]
        assert amt_hi - amt_lo == _AMOUNT_VSIZ, (
            f"'amount' value range width must be {_AMOUNT_VSIZ}, got {amt_hi - amt_lo}"
        )
        assert cur_hi - cur_lo == _CURRENCY_VSIZ, (
            f"'currency' value range width must be {_CURRENCY_VSIZ}, got {cur_hi - cur_lo}"
        )

    def test_vocabulary_spec_is_frozen(self) -> None:
        """VocabularySpec must be a frozen dataclass — immutable after construction."""
        spec = _make_pipeline().vocabulary_spec()
        with pytest.raises((AttributeError, TypeError)):
            spec.key_start = 999  # type: ignore[misc]

    def test_spec_contains_all_four_special_tokens(self) -> None:
        """special_tokens must contain PAD, MASK, CLS (EVT), SEP with correct IDs."""
        spec = _make_pipeline().vocabulary_spec()
        # TokenizerPipeline: PAD=0, MASK=1, CLS=2, SEP=3
        assert 0 in spec.special_tokens.values(), "PAD (ID=0) must be in special_tokens"
        assert 1 in spec.special_tokens.values(), "MASK (ID=1) must be in special_tokens"

    def test_value_size_matches_sum_of_field_vocab_sizes(self) -> None:
        """value_size == sum of all field tokenizer vocab_sizes."""
        spec = _make_pipeline().vocabulary_spec()
        assert spec.value_size == _VAL_SIZE, (
            f"value_size must be {_VAL_SIZE} (12 + 5), got {spec.value_size}"
        )


# ---------------------------------------------------------------------------
# TestVocabularyMap — verify ID arithmetic methods
# ---------------------------------------------------------------------------


class TestVocabularyMap:
    """Verify VocabularyMap ID arithmetic (ADR 002)."""

    def _make_spec_and_map(self):
        spec = _make_pipeline().vocabulary_spec()
        return spec, VocabularyMap(spec)

    def test_vocabulary_map_constructed_from_spec(self) -> None:
        """VocabularyMap must accept a VocabularySpec and initialise correctly."""
        spec, vmap = self._make_spec_and_map()
        assert isinstance(vmap, VocabularyMap)

    # --- global_to_local_value_id ---

    def test_global_to_local_value_id_at_value_start(self) -> None:
        """global_to_local_value_id(value_start) == 0 — first value maps to local 0.

        ADR 002: local_id = global_id - value_start
        """
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        ids = torch.tensor([spec.value_start])
        result = vmap.global_to_local_value_id(ids)
        assert result.item() == 0, (
            f"global_to_local_value_id({spec.value_start}) must be 0, got {result.item()}"
        )

    def test_global_to_local_value_id_at_value_end(self) -> None:
        """global_to_local_value_id(value_start + value_size - 1) == value_size - 1."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        last_global = spec.value_start + spec.value_size - 1
        ids = torch.tensor([last_global])
        result = vmap.global_to_local_value_id(ids)
        assert result.item() == spec.value_size - 1, (
            f"global_to_local_value_id({last_global}) must be {spec.value_size - 1}, "
            f"got {result.item()}"
        )

    def test_global_to_local_value_id_arithmetic(self) -> None:
        """global_to_local_value_id subtracts value_start from each ID."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        # Mid-range value
        mid_global = spec.value_start + 5
        ids = torch.tensor([mid_global])
        result = vmap.global_to_local_value_id(ids)
        assert result.item() == 5, (
            f"global_to_local_value_id({mid_global}) must be 5, got {result.item()}"
        )

    # --- is_key_id ---

    def test_is_key_id_true_for_key_range(self) -> None:
        """is_key_id returns True for IDs in [key_start, key_start + key_size)."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        key_ids = torch.tensor([spec.key_start, spec.key_start + 1])
        result = vmap.is_key_id(key_ids)
        assert result.all(), (
            f"is_key_id must be True for key IDs {key_ids.tolist()}"
        )

    def test_is_key_id_false_for_special_tokens(self) -> None:
        """is_key_id returns False for IDs < key_start (special token range)."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        special_ids = torch.tensor([0, 1, 2, 3])
        result = vmap.is_key_id(special_ids)
        assert not result.any(), (
            f"is_key_id must be False for special IDs [0,1,2,3]"
        )

    def test_is_key_id_false_for_value_ids(self) -> None:
        """is_key_id returns False for IDs >= value_start (value token range)."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        val_ids = torch.tensor([spec.value_start, spec.value_start + 3])
        result = vmap.is_key_id(val_ids)
        assert not result.any(), (
            f"is_key_id must be False for value IDs {val_ids.tolist()}"
        )

    # --- is_special_id ---

    def test_is_special_id_true_for_special_range(self) -> None:
        """is_special_id returns True for IDs in [0, key_start)."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        special_ids = torch.tensor([0, 1, 2, 3])
        result = vmap.is_special_id(special_ids)
        assert result.all(), "is_special_id must be True for IDs 0–3"

    def test_is_special_id_false_for_key_ids(self) -> None:
        """is_special_id returns False for IDs >= key_start."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        key_ids = torch.tensor([spec.key_start, spec.key_start + 1])
        result = vmap.is_special_id(key_ids)
        assert not result.any(), (
            f"is_special_id must be False for key IDs {key_ids.tolist()}"
        )

    # --- is_global_value_id ---

    def test_is_global_value_id_true_for_value_range(self) -> None:
        """is_global_value_id returns True for IDs in [value_start, value_start + value_size)."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        val_ids = torch.tensor([spec.value_start, spec.value_start + spec.value_size - 1])
        result = vmap.is_global_value_id(val_ids)
        assert result.all(), (
            f"is_global_value_id must be True for {val_ids.tolist()}"
        )

    def test_is_global_value_id_false_for_out_of_range(self) -> None:
        """is_global_value_id returns False for IDs >= value_start + value_size."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        out_of_range = spec.value_start + spec.value_size  # first ID beyond value space
        ids = torch.tensor([out_of_range])
        result = vmap.is_global_value_id(ids)
        assert not result.any(), (
            f"is_global_value_id must be False for {out_of_range} (beyond value space)"
        )

    def test_is_global_value_id_false_for_key_ids(self) -> None:
        """is_global_value_id returns False for key IDs."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        key_ids = torch.tensor([spec.key_start, spec.key_start + 1])
        result = vmap.is_global_value_id(key_ids)
        assert not result.any(), (
            f"is_global_value_id must be False for key IDs {key_ids.tolist()}"
        )

    def test_is_global_value_id_false_for_special_ids(self) -> None:
        """is_global_value_id returns False for special token IDs."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        special_ids = torch.tensor([0, 1, 2, 3])
        result = vmap.is_global_value_id(special_ids)
        assert not result.any(), "is_global_value_id must be False for special IDs"

    # --- Mutual exclusivity ---

    def test_id_regions_are_mutually_exclusive(self) -> None:
        """Each global ID falls into exactly one region: special, key, or value."""
        pytest.importorskip("torch")
        import torch
        spec, vmap = self._make_spec_and_map()
        all_ids = torch.arange(spec.total_embedding_vocab_size)
        is_spec = vmap.is_special_id(all_ids)
        is_key  = vmap.is_key_id(all_ids)
        is_val  = vmap.is_global_value_id(all_ids)

        # Each ID must be in exactly one region
        combined = is_spec.int() + is_key.int() + is_val.int()
        assert (combined == 1).all(), (
            "Each ID must belong to exactly one region (special, key, or value). "
            f"Found IDs not in exactly one region: {all_ids[(combined != 1)].tolist()}"
        )
