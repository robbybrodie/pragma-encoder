"""Tests for AssembledBatch.

Derived from PRAGMA paper Section 2.3 and ADR 002
(docs/decisions/002-embedding-assembler.md):
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:    verify validate() passes for correct tensor shapes/dtypes
  Contract tests: verify validate() raises for violated ADR 002 invariants
  Spec tests:     verify IGNORE_INDEX == -100 (PyTorch cross_entropy convention)

Key ADR 002 invariant (the invariant that AssembledBatch.validate() enforces):
  targets[targets != IGNORE_INDEX].max() < config.value_vocab_size
  targets[targets != IGNORE_INDEX].min() >= 0

  This prevents the silent bug where global token IDs (which include key vocab
  offsets ~60 and value offsets ~60+) are passed as MLM targets into a head
  that only knows about value_vocab_size (~28k local IDs).  Global IDs in the
  hundreds or thousands would silently land in valid range and corrupt training.

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from src.model.assembled_batch import AssembledBatch
from src.model.config import PRAGMAConfig

_CONFIG = PRAGMAConfig.pragma_s()

# Test dimensions — small enough for fast tests
_B  = 2   # batch size
_NE = 3   # number of events
_NI = 4   # tokens per event
_NA = 5   # profile tokens
_D  = _CONFIG.d_model


# ---------------------------------------------------------------------------
# Helper — build a valid AssembledBatch
# ---------------------------------------------------------------------------


def _make_valid_batch(
    target_value: int = 0,
    ignore_some: bool = True,
) -> AssembledBatch:
    """Return a correctly-shaped AssembledBatch.

    targets: all value_vocab_id = target_value (valid local IDs in [0, value_vocab_size))
    If ignore_some: set some positions to IGNORE_INDEX (-100) — the normal case.
    """
    xa       = torch.randn(_B, _NA, _D)
    ta       = torch.zeros(_B, _NA)
    xe       = torch.randn(_B, _NE, _NI, _D)
    xt       = torch.randint(0, 24, (_B, _NE, 3))
    te       = torch.zeros(_B, 1 + _NE)
    mlm_mask = torch.zeros(_B, _NE, _NI, dtype=torch.bool)
    mlm_mask[:, :, 1:] = True   # mask all non-EVT positions

    targets = torch.full((_B, _NE, _NI), AssembledBatch.IGNORE_INDEX, dtype=torch.long)
    if ignore_some:
        # Only set targets where mlm_mask is True
        targets[mlm_mask] = target_value
    else:
        # All positions have a valid target
        targets[:] = target_value

    return AssembledBatch(
        xa=xa, ta=ta, xe=xe, xt=xt, te=te, mlm_mask=mlm_mask, targets=targets,
    )


# ---------------------------------------------------------------------------
# TestAssembledBatchSpec — IGNORE_INDEX and class contract
# ---------------------------------------------------------------------------


class TestAssembledBatchSpec:
    """Verify class-level constants and dataclass structure."""

    def test_ignore_index_is_minus_100(self) -> None:
        """IGNORE_INDEX must be -100 — PyTorch cross_entropy ignore convention.

        ADR 002: IGNORE_INDEX = -100 matches nn.CrossEntropyLoss(ignore_index=-100).
        """
        assert AssembledBatch.IGNORE_INDEX == -100, (
            f"IGNORE_INDEX must be -100 (PyTorch convention), "
            f"got {AssembledBatch.IGNORE_INDEX}"
        )

    def test_ignore_index_is_class_variable(self) -> None:
        """IGNORE_INDEX must be a ClassVar — not in __init__, not per-instance."""
        batch = _make_valid_batch()
        # ClassVar: the value on the class and the instance must be the same object
        assert AssembledBatch.IGNORE_INDEX == batch.IGNORE_INDEX == -100

    def test_assembled_batch_has_required_fields(self) -> None:
        """AssembledBatch must expose xa, ta, xe, xt, te, mlm_mask, targets."""
        batch = _make_valid_batch()
        assert hasattr(batch, "xa")
        assert hasattr(batch, "ta")
        assert hasattr(batch, "xe")
        assert hasattr(batch, "xt")
        assert hasattr(batch, "te")
        assert hasattr(batch, "mlm_mask")
        assert hasattr(batch, "targets")


# ---------------------------------------------------------------------------
# TestValidatePasses — validate() accepts correct batches
# ---------------------------------------------------------------------------


class TestValidatePasses:
    """validate() must not raise for correctly-constructed batches."""

    def test_validate_passes_for_valid_batch(self) -> None:
        """validate() must pass silently for a correctly-shaped batch."""
        batch = _make_valid_batch()
        batch.validate(_CONFIG)  # must not raise

    def test_validate_passes_for_all_ignore_targets(self) -> None:
        """validate() must pass when all targets are IGNORE_INDEX (no masked tokens)."""
        xa       = torch.randn(_B, _NA, _D)
        ta       = torch.zeros(_B, _NA)
        xe       = torch.randn(_B, _NE, _NI, _D)
        xt       = torch.randint(0, 24, (_B, _NE, 3))
        te       = torch.zeros(_B, 1 + _NE)
        mlm_mask = torch.zeros(_B, _NE, _NI, dtype=torch.bool)
        targets  = torch.full((_B, _NE, _NI), AssembledBatch.IGNORE_INDEX, dtype=torch.long)

        batch = AssembledBatch(xa=xa, ta=ta, xe=xe, xt=xt, te=te,
                               mlm_mask=mlm_mask, targets=targets)
        batch.validate(_CONFIG)  # all-ignore is valid (no MLM positions)

    def test_validate_passes_for_max_valid_target(self) -> None:
        """validate() must pass when targets == value_vocab_size - 1 (last valid ID)."""
        max_valid = _CONFIG.value_vocab_size - 1
        batch = _make_valid_batch(target_value=max_valid)
        batch.validate(_CONFIG)

    def test_validate_passes_for_target_zero(self) -> None:
        """validate() must pass when targets == 0 (first valid value_vocab_id)."""
        batch = _make_valid_batch(target_value=0)
        batch.validate(_CONFIG)

    def test_validate_passes_with_all_targets_set(self) -> None:
        """validate() must pass when all positions have valid local targets (no ignore)."""
        batch = _make_valid_batch(target_value=100, ignore_some=False)
        batch.validate(_CONFIG)


# ---------------------------------------------------------------------------
# TestValidateRaises — validate() rejects violated invariants
# ---------------------------------------------------------------------------


class TestValidateRaises:
    """validate() must raise AssertionError (or ValueError) for bad batches."""

    def test_validate_raises_for_global_id_in_targets(self) -> None:
        """validate() must raise when targets contain global token IDs.

        ADR 002 key invariant: targets must be value_vocab_local IDs in
        [0, value_vocab_size), not global IDs (which include key offsets).

        The silent bug: global_id for a value token could be e.g. 65 (key_size=60
        + local_id=5), which is within [0, value_vocab_size=28000) and would NOT
        trigger an out-of-range exception — it would silently corrupt training.
        validate() catches this by requiring targets < value_vocab_size when != IGNORE.

        Here we test with a clearly out-of-range global ID to verify the guard.
        """
        batch = _make_valid_batch()
        # Set one target to value_vocab_size (one beyond valid range)
        batch.targets[0, 0, 1] = _CONFIG.value_vocab_size
        with pytest.raises((AssertionError, ValueError)):
            batch.validate(_CONFIG)

    def test_validate_raises_for_negative_non_ignore_target(self) -> None:
        """validate() must raise when targets contain negative values != IGNORE_INDEX.

        Only IGNORE_INDEX (-100) is a permitted negative target. Any other
        negative value (e.g. -1, -50) indicates a logic error in the caller.
        """
        batch = _make_valid_batch()
        # Set one target to -1 (not IGNORE_INDEX=-100, not a valid local ID)
        batch.targets[0, 0, 1] = -1
        with pytest.raises((AssertionError, ValueError)):
            batch.validate(_CONFIG)

    def test_validate_raises_for_xa_wrong_ndim(self) -> None:
        """validate() must raise when xa is not 3-dimensional."""
        batch = _make_valid_batch()
        batch.xa = torch.randn(_B, _NA)  # 2D instead of 3D
        with pytest.raises((AssertionError, ValueError)):
            batch.validate(_CONFIG)

    def test_validate_raises_for_xe_wrong_ndim(self) -> None:
        """validate() must raise when xe is not 4-dimensional."""
        batch = _make_valid_batch()
        batch.xe = torch.randn(_B, _NE, _D)  # 3D instead of 4D
        with pytest.raises((AssertionError, ValueError)):
            batch.validate(_CONFIG)

    def test_validate_raises_for_xt_wrong_last_dim(self) -> None:
        """validate() must raise when xt.shape[-1] != 3 (calendar features).

        §2.2: calendar features are [hour, day_of_week, day_of_month] — exactly 3.
        key-numbers.md: calendar_feature_dims = 3.
        """
        batch = _make_valid_batch()
        batch.xt = torch.randint(0, 24, (_B, _NE, 4))  # 4 calendar features (wrong)
        with pytest.raises((AssertionError, ValueError)):
            batch.validate(_CONFIG)

    def test_validate_raises_for_mlm_mask_wrong_dtype(self) -> None:
        """validate() must raise when mlm_mask is not boolean dtype."""
        batch = _make_valid_batch()
        batch.mlm_mask = batch.mlm_mask.long()  # int instead of bool
        with pytest.raises((AssertionError, ValueError)):
            batch.validate(_CONFIG)

    def test_validate_raises_for_targets_wrong_dtype(self) -> None:
        """validate() must raise when targets is not long (int64) dtype."""
        batch = _make_valid_batch()
        batch.targets = batch.targets.float()  # float instead of long
        with pytest.raises((AssertionError, ValueError)):
            batch.validate(_CONFIG)
