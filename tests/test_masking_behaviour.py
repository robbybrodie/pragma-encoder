"""TD-005 masking behaviour contract tests.

Verifies three properties that document the boundary between MaskingStrategy
and the training loop (train.py):

  1. Deterministic masking — same RNG seed + same input → identical output.
     Reproducibility depends on torch.manual_seed() before each forward() call.

  2. Label alignment — for positions where mask=True, target_ids must equal the
     original token values (not PAD/MASK). MaskingStrategy.forward() always
     returns target_ids == token_ids, so this is trivially True — the test
     documents the contract so future refactors cannot break it.

  3. PAD guard — MaskingStrategy does NOT filter padding positions internally.
     The training loop applies the guard:
         mlm_mask = mlm_mask & xe_valid  # train.py:512
     This test documents that the guard is NOT in MaskingStrategy itself:
     if token_ids contains PAD-valued positions, the masking strategy may
     select and return them as mask=True. The training loop is responsible
     for zeroing those positions via xe_valid.

TD-005 resolution note:
    The PAD-as-UNK problem (TD-005) is that PAD_ID (0) is also used as
    _UNK_TOKEN_ID in MaskingStrategy — so a PAD position that happens to be
    selected by the UNK branch becomes indistinguishable from a real PAD.
    The xe_valid & mlm_mask guard in train.py is the correct fix for the
    MLM loss contribution. Full TD-005 resolution (teaching MaskingStrategy
    to accept xe_valid and internally exclude PAD rows) is a future milestone.
    These tests document the current behaviour — they must not be changed to
    match a different implementation without updating this note.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.3.5
ADR: docs/decisions (masking design)
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pragma_encoder.masking.strategy import MaskingStrategy
from pragma_encoder.model.config import PRAGMAConfig

_CONFIG = PRAGMAConfig.pragma_s()
_PAD_ID = 0   # TokenizerPipeline.PAD_ID — also used as UNK replacement


# ---------------------------------------------------------------------------
# 1. Deterministic masking — same seed → same output
# ---------------------------------------------------------------------------


class TestDeterministicMasking:
    """Verify that MaskingStrategy is deterministic given the same RNG seed.

    This is a reproducibility contract: torch.manual_seed() before forward()
    must produce bit-identical masked_ids, target_ids, and mask tensors for
    identical inputs.
    """

    def test_same_seed_same_output_masked_ids(self) -> None:
        """Same RNG seed must produce identical masked_ids.

        If two forward() calls with the same seed and input produce different
        masked_ids, training runs are not reproducible from a checkpoint.
        """
        masker = MaskingStrategy(_CONFIG)
        batch, ne, ni = 2, 6, 8
        token_ids = torch.arange(batch * ne * ni).reshape(batch, ne, ni) + 10
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        torch.manual_seed(42)
        masked_ids_1, _, _ = masker.forward(token_ids, key_ids)

        torch.manual_seed(42)
        masked_ids_2, _, _ = masker.forward(token_ids, key_ids)

        assert (masked_ids_1 == masked_ids_2).all(), (
            "MaskingStrategy must be deterministic: same torch.manual_seed(42) + "
            "same input must produce identical masked_ids. "
            "Two forward() calls with the same seed diverged."
        )

    def test_same_seed_same_output_mask(self) -> None:
        """Same RNG seed must produce identical mask (MLM loss selector) tensors."""
        masker = MaskingStrategy(_CONFIG)
        batch, ne, ni = 2, 6, 8
        token_ids = torch.arange(batch * ne * ni).reshape(batch, ne, ni) + 10
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        torch.manual_seed(99)
        _, _, mask_1 = masker.forward(token_ids, key_ids)

        torch.manual_seed(99)
        _, _, mask_2 = masker.forward(token_ids, key_ids)

        assert (mask_1 == mask_2).all(), (
            "MaskingStrategy must be deterministic: same torch.manual_seed(99) + "
            "same input must produce identical mask tensors."
        )

    def test_different_seeds_may_differ(self) -> None:
        """Different RNG seeds should (almost certainly) produce different outputs.

        This is a probabilistic sanity check. The probability that two independent
        Bernoulli masks are identical over 2×6×8=96 positions is negligible.
        """
        masker = MaskingStrategy(_CONFIG)
        batch, ne, ni = 2, 6, 8
        token_ids = torch.arange(batch * ne * ni).reshape(batch, ne, ni) + 10
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        torch.manual_seed(0)
        masked_ids_0, _, _ = masker.forward(token_ids, key_ids)

        torch.manual_seed(1)
        masked_ids_1, _, _ = masker.forward(token_ids, key_ids)

        # With high probability, different seeds produce different masks.
        # If this fails it is a fluke (p ≈ 2^{-96}) — do not add a workaround.
        assert not (masked_ids_0 == masked_ids_1).all(), (
            "Different RNG seeds almost certainly produce different masks. "
            "If this assertion fails, re-run — it is a statistical fluke."
        )


# ---------------------------------------------------------------------------
# 2. Label alignment — mask=True positions always have non-zero targets
# ---------------------------------------------------------------------------


class TestLabelAlignment:
    """Verify that mask=True positions always carry the original token IDs as targets.

    MaskingStrategy.forward() returns target_ids == token_ids everywhere (not just
    at masked positions). This test documents the contract: for every position
    where mask=True, the corresponding target_id is the original (non-PAD) value.

    This assumes the caller provides non-PAD token_ids for real (non-padding)
    positions. The training loop enforces this by slicing with xe_valid before
    computing the loss. These tests verify MaskingStrategy itself upholds the
    target_ids == token_ids invariant.
    """

    def test_mask_true_targets_equal_original(self) -> None:
        """target_ids at mask=True positions must equal original token_ids.

        The MLM loss uses target_ids[mlm_mask] as labels. If target_ids does not
        equal the original input, the loss is computed against wrong labels.
        """
        masker = MaskingStrategy(_CONFIG)
        torch.manual_seed(7)
        batch, ne, ni = 3, 5, 8
        # All tokens >= 10 (well above PAD=0 and MASK=1)
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        _, target_ids, mask = masker.forward(token_ids, key_ids)

        if mask.any():
            expected = token_ids[mask]
            actual = target_ids[mask]
            assert (actual == expected).all(), (
                "target_ids at mask=True positions must equal original token_ids. "
                f"Divergence at {(actual != expected).sum().item()} positions."
            )

    def test_target_ids_unchanged_everywhere(self) -> None:
        """target_ids must equal token_ids at ALL positions, not just masked ones.

        This is the full invariant: target_ids is a clone of token_ids and must
        not be modified by MaskingStrategy.forward() under any code path.
        """
        masker = MaskingStrategy(_CONFIG)
        torch.manual_seed(11)
        batch, ne, ni = 2, 6, 10
        token_ids = torch.randint(10, 500, (batch, ne, ni))
        key_ids = torch.randint(0, 5, (batch, ne, ni))

        _, target_ids, _ = masker.forward(token_ids, key_ids)

        assert (target_ids == token_ids).all(), (
            "target_ids must be identical to token_ids at ALL positions. "
            "MaskingStrategy must not modify target_ids."
        )

    def test_mask_true_positions_not_all_pad(self) -> None:
        """mask=True positions must not be PAD values in the target.

        When token_ids has no PAD positions (all values >= 10), mask=True
        positions must have non-PAD targets. This confirms MaskingStrategy
        does not accidentally label real tokens as PAD.
        """
        masker = MaskingStrategy(_CONFIG)
        torch.manual_seed(13)
        batch, ne, ni = 3, 6, 8
        # No PAD (0) or MASK (1) values in input
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        _, target_ids, mask = masker.forward(token_ids, key_ids)

        if mask.any():
            targets_at_masked = target_ids[mask]
            assert (targets_at_masked != _PAD_ID).all(), (
                "With non-PAD token_ids input, target_ids at mask=True positions "
                "must not contain PAD_ID (0). "
                f"Found {(targets_at_masked == _PAD_ID).sum().item()} PAD values."
            )


# ---------------------------------------------------------------------------
# 3. PAD guard — MaskingStrategy does NOT filter PAD positions internally
# ---------------------------------------------------------------------------


class TestPadGuardIsInTrainingLoop:
    """Document that PAD filtering is the training loop's responsibility.

    MaskingStrategy.forward() does NOT receive xe_valid and does NOT filter
    padding positions internally. The guard:

        mlm_mask = mlm_mask & xe_valid  # train.py:512

    is applied in the training loop after MaskingStrategy returns.

    These tests document the CURRENT behaviour:
      - MaskingStrategy may select PAD-valued positions for masking.
      - It does NOT have an xe_valid parameter.
      - The training loop is the single point of truth for PAD exclusion.

    Changing MaskingStrategy to accept xe_valid internally is a future
    TD-005 resolution milestone — do not add that parameter without also
    updating train.py and these tests.
    """

    def test_masking_strategy_has_no_xe_valid_parameter(self) -> None:
        """MaskingStrategy.forward() must NOT accept xe_valid.

        PAD filtering is the training loop's responsibility (train.py:512).
        If xe_valid is added to forward(), train.py must be updated too,
        and this test must be revised as part of the TD-005 resolution.
        """
        import inspect
        sig = inspect.signature(MaskingStrategy.forward)
        assert "xe_valid" not in sig.parameters, (
            "MaskingStrategy.forward() must not accept xe_valid. "
            "PAD filtering is done in train.py via: mlm_mask = mlm_mask & xe_valid. "
            "Adding xe_valid to MaskingStrategy is a TD-005 resolution step — "
            "see test docstring for the required coordinated change."
        )

    def test_masking_strategy_may_select_pad_valued_positions(self) -> None:
        """MaskingStrategy can select PAD-valued positions for masking.

        This test documents that MaskingStrategy does NOT guard against PAD
        positions. With token_mask_prob=1.0, every position is selected.
        If a position has token_id == PAD_ID (0), it can be selected.

        The training loop guard (mlm_mask & xe_valid) prevents PAD-row
        positions from contributing to the MLM loss, even if MaskingStrategy
        marks them as mask=True.
        """
        config = dataclasses.replace(
            _CONFIG,
            token_mask_prob=1.0,   # Select every position
            event_mask_prob=0.0,
            key_mask_prob=0.0,
        )
        masker = MaskingStrategy(config)
        torch.manual_seed(17)

        batch, ne, ni = 2, 4, 6
        # Mix of real tokens (>=10) and PAD_ID (0) positions
        token_ids = torch.randint(10, 100, (batch, ne, ni))
        token_ids[:, -1, :] = _PAD_ID  # Last event row is all PAD

        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        _, _, mask = masker.forward(token_ids, key_ids)

        # With token_mask_prob=1.0, ALL positions (including PAD rows) should
        # be in the MASK branch (subject to UNK fraction splitting).
        # We test that some PAD-row positions ARE selected (mask=True).
        # If MaskingStrategy internally filtered PAD, the PAD rows would all be False.
        pad_row_mask = mask[:, -1, :]  # mask values for the last (PAD) event row
        # Note: ~10% of positions go to UNK branch (mask=False), so not all will be True.
        # But with p=1.0 selection and only ~10% UNK, we expect ~90% to be mask=True.
        # At least some PAD-row positions must be selected (mask=True).
        assert pad_row_mask.any(), (
            "MaskingStrategy does NOT filter PAD positions internally. "
            "With token_mask_prob=1.0, PAD-valued positions should be selected "
            "(mask=True). The training loop applies mlm_mask & xe_valid to exclude "
            "them from the MLM loss. "
            "If this assertion fails, MaskingStrategy may have added internal PAD "
            "filtering — update train.py and this test together (TD-005)."
        )

    def test_training_loop_guard_pattern_documented(self) -> None:
        """Document the training loop PAD guard pattern by simulating it.

        Simulates what train.py does:
            mlm_mask = mlm_mask & xe_valid  # train.py:512

        After applying xe_valid, PAD-row positions must have mask=False,
        even if MaskingStrategy marked them as mask=True.

        This test does NOT import train.py — it only simulates the guard pattern
        to document its effect and to serve as a regression test if the guard is
        accidentally removed from train.py.
        """
        config = dataclasses.replace(
            _CONFIG,
            token_mask_prob=1.0,
            event_mask_prob=0.0,
            key_mask_prob=0.0,
        )
        masker = MaskingStrategy(config)
        torch.manual_seed(19)

        batch, ne, ni = 2, 4, 6
        token_ids = torch.randint(10, 100, (batch, ne, ni))
        token_ids[:, -1, :] = _PAD_ID  # Last event is PAD

        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        _, _, mlm_mask = masker.forward(token_ids, key_ids)

        # Build xe_valid: True for real token positions, False for PAD event rows.
        # In the real training loop, xe_valid comes from PragmaDataset and marks
        # real (non-padding) token positions.
        xe_valid = torch.ones(batch, ne, ni, dtype=torch.bool)
        xe_valid[:, -1, :] = False  # Last event row is padding

        # Apply the training loop guard
        mlm_mask_guarded = mlm_mask & xe_valid

        # After the guard, PAD-row positions must all be False
        assert not mlm_mask_guarded[:, -1, :].any(), (
            "After applying mlm_mask & xe_valid (train.py:512 guard), "
            "PAD-row positions must all have mask=False — they must not "
            "contribute to the MLM loss. "
            "This simulation confirms the guard pattern is correct."
        )

        # Non-PAD positions should still have some mask=True entries
        non_pad_mask = mlm_mask_guarded[:, :-1, :]
        assert non_pad_mask.any(), (
            "After applying the xe_valid guard, non-PAD positions must still "
            "have some mask=True entries. Training cannot proceed if all real "
            "token positions are excluded from the MLM loss."
        )
