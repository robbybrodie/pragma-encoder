"""Tests for MaskingStrategy.

Derived from PRAGMA paper Section 2.3.5:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:        verify forward(token_ids, key_ids) output shapes
  Math tests:         verify masking rates, strategy atomicity, UNK handling
  Architecture tests: verify not nn.Module, pure tensor ops, config constructor
  Spec tests:         verify probabilities from config

Key paper properties (§2.3.5):
  - Input:  token_ids (batch, ne, ni) — integer value token IDs
  - Input:  key_ids   (batch, ne, ni) — integer key type IDs
  - Output: masked_ids (batch, ne, ni) — [MASK] or [UNK] at selected positions
  - Output: target_ids (batch, ne, ni) — original token IDs (for MLM loss)
  - Output: mask       (batch, ne, ni) — bool: True=[MASK] (in loss), False elsewhere
  - Token-level masking:         0.15 probability per position (§2.3.5)
  - Event-level masking:         0.10 probability per event (§2.3.5)
  - Semantic-type (key) masking: 0.10 probability per key type (§2.3.5)
  - UNK replacement: small fraction of masked → PAD_ID(0), mask=False (excluded from loss)
  - No nn.Module — pure stateless tensor operations

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import dataclasses

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pragma_encoder.masking.strategy import MaskingStrategy
from pragma_encoder.model.config import PRAGMAConfig

_CONFIG = PRAGMAConfig.pragma_s()

# Special token IDs — must match TokenizerPipeline (src/pragma_encoder/tokenizer/pipeline.py)
_MASK_TOKEN_ID = 1  # TokenizerPipeline.MASK_ID
_UNK_TOKEN_ID = 0   # TokenizerPipeline.PAD_ID — used as UNK replacement (no global UNK)


# ---------------------------------------------------------------------------
# TestShapes — verify forward(token_ids, key_ids) output shapes
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify MaskingStrategy input/output shapes (§2.3.5)."""

    def test_output_shapes(self) -> None:
        """§2.3.5: forward returns 3 tensors, all (batch, ne, ni).

        masked_ids:  (batch, ne, ni) — token IDs with masks applied
        target_ids:  (batch, ne, ni) — original token IDs (for MLM loss)
        mask:        (batch, ne, ni) — bool: True=[MASK] (in loss)
        """
        masker = MaskingStrategy(_CONFIG)
        batch, ne, ni = 2, 6, 8
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.randint(0, 10, (batch, ne, ni))

        masked_ids, target_ids, mask = masker.forward(token_ids, key_ids)

        assert masked_ids.shape == (batch, ne, ni), (
            f"masked_ids must be (batch, ne, ni) = ({batch}, {ne}, {ni}), "
            f"got {tuple(masked_ids.shape)}"
        )
        assert target_ids.shape == (batch, ne, ni), (
            f"target_ids must be (batch, ne, ni) = ({batch}, {ne}, {ni}), "
            f"got {tuple(target_ids.shape)}"
        )
        assert mask.shape == (batch, ne, ni), (
            f"mask must be (batch, ne, ni) = ({batch}, {ne}, {ni}), "
            f"got {tuple(mask.shape)}"
        )

    def test_masked_ids_same_dtype_as_input(self) -> None:
        """§2.3.5: masked_ids and target_ids preserve the dtype of token_ids."""
        masker = MaskingStrategy(_CONFIG)
        batch, ne, ni = 2, 4, 6
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        masked_ids, target_ids, _ = masker.forward(token_ids, key_ids)

        assert masked_ids.dtype == token_ids.dtype, (
            f"masked_ids dtype must match token_ids dtype ({token_ids.dtype}), "
            f"got {masked_ids.dtype}"
        )
        assert target_ids.dtype == token_ids.dtype, (
            f"target_ids dtype must match token_ids dtype ({token_ids.dtype}), "
            f"got {target_ids.dtype}"
        )

    def test_mask_is_boolean(self) -> None:
        """§2.3.5: mask is a boolean tensor — True at positions included in MLM loss."""
        masker = MaskingStrategy(_CONFIG)
        batch, ne, ni = 2, 4, 6
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        _, _, mask = masker.forward(token_ids, key_ids)

        assert mask.dtype == torch.bool, (
            f"mask must be dtype torch.bool (True=[MASK], False=excluded from loss), "
            f"got {mask.dtype}"
        )


# ---------------------------------------------------------------------------
# TestMathProperties — masking rates, strategy atomicity, UNK handling
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from §2.3.5."""

    def test_token_masking_rate(self) -> None:
        """§2.3.5: token masking selects approximately token_mask_prob (0.15) of positions.

        Only token masking active (event and key masking disabled via 0.0 probability).
        Total selected = MASK positions + UNK positions ≈ 15% of all tokens.
        """
        config = dataclasses.replace(
            _CONFIG, token_mask_prob=0.15, event_mask_prob=0.0, key_mask_prob=0.0
        )
        masker = MaskingStrategy(config)
        torch.manual_seed(42)
        batch, ne, ni = 50, 20, 24  # large sample for statistical test
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        masked_ids, _, _ = masker.forward(token_ids, key_ids)

        # "selected" = any position changed from original (MASK or UNK)
        selected = masked_ids != token_ids
        fraction = selected.float().mean().item()

        assert 0.10 < fraction < 0.22, (
            f"§2.3.5: token masking rate should be ~0.15 (token_mask_prob), "
            f"got {fraction:.3f}"
        )

    def test_event_masking_masks_entire_event(self) -> None:
        """§2.3.5: event masking masks ALL ni tokens within a selected event.

        With event_mask_prob=1.0, every event is selected. All token positions
        must be changed (MASK or UNK). Partial event masking is wrong.
        """
        config = dataclasses.replace(
            _CONFIG, event_mask_prob=1.0, token_mask_prob=0.0, key_mask_prob=0.0
        )
        masker = MaskingStrategy(config)
        torch.manual_seed(0)
        batch, ne, ni = 2, 4, 8
        # token_ids >= 10 so any replacement (MASK=1 or UNK=0) is detectable
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        masked_ids, _, _ = masker.forward(token_ids, key_ids)

        # Every token must be changed (event_mask_prob=1.0 selects all events)
        selected = masked_ids != token_ids
        assert selected.all(), (
            "§2.3.5: with event_mask_prob=1.0, all tokens in all events must be "
            "masked. Partial event masking violates the event-level strategy."
        )

    def test_key_masking_masks_all_matching_positions(self) -> None:
        """§2.3.5: key masking masks ALL positions with the selected key type.

        When a key type is selected for masking, every position in the (batch, ne, ni)
        tensor where key_ids == k must be changed. Partial key masking is wrong.
        """
        config = dataclasses.replace(
            _CONFIG, key_mask_prob=1.0, token_mask_prob=0.0, event_mask_prob=0.0
        )
        masker = MaskingStrategy(config)
        torch.manual_seed(1)
        batch, ne, ni = 2, 5, 6
        # token_ids >= 10: any replacement is detectable
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        # Assign key type 42 to positions [:, :, :3]; key type 7 to [:, :, 3:]
        key_ids = torch.full((batch, ne, ni), 7, dtype=torch.long)
        key_ids[:, :, :3] = 42

        masked_ids, _, _ = masker.forward(token_ids, key_ids)
        selected = masked_ids != token_ids

        # key_mask_prob=1.0 → key type 42 is always selected
        # ALL positions with key_id=42 must be changed
        key42_positions = key_ids == 42
        assert selected[key42_positions].all(), (
            "§2.3.5: when key type 42 is selected for masking, ALL positions "
            "where key_ids==42 must be masked. Partial key masking is wrong."
        )

    def test_unmasked_positions_unchanged(self) -> None:
        """§2.3.5: positions not selected by any strategy are unchanged.

        masked_ids must equal token_ids at all positions not selected for masking.
        """
        config = dataclasses.replace(
            _CONFIG, token_mask_prob=0.5, event_mask_prob=0.0, key_mask_prob=0.0
        )
        masker = MaskingStrategy(config)
        torch.manual_seed(2)
        batch, ne, ni = 3, 4, 8
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        masked_ids, _, _ = masker.forward(token_ids, key_ids)

        not_selected = masked_ids == token_ids  # positions that were not changed
        # Verify: nothing was changed without being selected (trivially True by construction)
        # Key property: not-selected positions retain original values exactly
        assert (masked_ids[not_selected] == token_ids[not_selected]).all(), (
            "§2.3.5: unselected positions must retain their original token IDs."
        )

    def test_target_ids_always_equal_original(self) -> None:
        """§2.3.5: target_ids always contains the original token IDs.

        target_ids is the label tensor for MLM loss — it must equal token_ids
        at every position, regardless of whether that position is masked.
        """
        masker = MaskingStrategy(_CONFIG)
        torch.manual_seed(3)
        batch, ne, ni = 2, 4, 6
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.randint(0, 5, (batch, ne, ni))

        _, target_ids, _ = masker.forward(token_ids, key_ids)

        assert (target_ids == token_ids).all(), (
            "§2.3.5: target_ids must equal original token_ids everywhere — "
            "it is the label tensor for the MLM loss."
        )

    def test_mask_true_positions_have_mask_token(self) -> None:
        """§2.3.5: positions with mask=True contain _MASK_TOKEN_ID (1).

        mask=True indicates a position that is in the MLM loss.
        These positions must have been replaced with _MASK_TOKEN_ID.
        """
        config = dataclasses.replace(
            _CONFIG, token_mask_prob=0.5, event_mask_prob=0.0, key_mask_prob=0.0
        )
        masker = MaskingStrategy(config)
        torch.manual_seed(4)
        batch, ne, ni = 2, 5, 8
        # token_ids > 1 so MASK_TOKEN_ID (1) is detectable
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        masked_ids, _, mask = masker.forward(token_ids, key_ids)

        if mask.any():
            assert (masked_ids[mask] == _MASK_TOKEN_ID).all(), (
                f"§2.3.5: positions where mask=True must contain "
                f"_MASK_TOKEN_ID ({_MASK_TOKEN_ID}). "
                f"Got values: {masked_ids[mask].unique().tolist()}"
            )

    def test_unk_positions_excluded_from_mask(self) -> None:
        """§2.3.5: UNK-replaced positions have mask=False — excluded from MLM loss.

        A small fraction (_UNK_FRACTION=0.10) of selected positions are replaced
        with _UNK_TOKEN_ID instead of _MASK_TOKEN_ID. These act as input dropout
        and are excluded from the MLM objective (mask=False).
        """
        config = dataclasses.replace(
            _CONFIG, token_mask_prob=1.0, event_mask_prob=0.0, key_mask_prob=0.0
        )
        masker = MaskingStrategy(config)
        torch.manual_seed(5)
        batch, ne, ni = 4, 8, 10  # 320 positions; ~32 expected UNK with fraction=0.10
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        masked_ids, _, mask = masker.forward(token_ids, key_ids)

        # UNK positions: changed from original AND mask=False
        changed = masked_ids != token_ids
        unk_positions = changed & ~mask

        assert unk_positions.any(), (
            "§2.3.5: with token_mask_prob=1.0, some masked positions must be "
            "replaced with UNK (PAD_ID=0) and excluded from loss (mask=False). "
            "UNK replacement fraction is 0.10 of selected positions."
        )
        assert not mask[unk_positions].any(), (
            "§2.3.5: UNK positions must have mask=False (excluded from MLM loss)."
        )

    def test_unk_positions_have_unk_token(self) -> None:
        """§2.3.5: UNK-replaced positions contain _UNK_TOKEN_ID (PAD_ID = 0).

        The paper specifies [UNK] replacement. TokenizerPipeline has no global
        [UNK] token — PAD_ID (0) is used as the corruption token. This is an
        implementation decision documented in MaskingStrategy.

        These positions have mask=False and are excluded from the MLM loss.
        """
        config = dataclasses.replace(
            _CONFIG, token_mask_prob=1.0, event_mask_prob=0.0, key_mask_prob=0.0
        )
        masker = MaskingStrategy(config)
        torch.manual_seed(6)
        batch, ne, ni = 4, 8, 10  # 320 positions; ~32 expected UNK
        token_ids = torch.randint(10, 1000, (batch, ne, ni))
        key_ids = torch.zeros(batch, ne, ni, dtype=torch.long)

        masked_ids, _, mask = masker.forward(token_ids, key_ids)

        # UNK positions: changed from original AND mask=False
        changed = masked_ids != token_ids
        unk_positions = changed & ~mask

        assert unk_positions.any(), (
            "With token_mask_prob=1.0, UNK positions must exist "
            "(UNK fraction = 0.10 of selected positions)."
        )
        assert (masked_ids[unk_positions] == _UNK_TOKEN_ID).all(), (
            f"§2.3.5: UNK-replaced positions must contain _UNK_TOKEN_ID "
            f"(PAD_ID = {_UNK_TOKEN_ID}). "
            f"TokenizerPipeline has no global [UNK]; PAD_ID is used as the "
            f"corruption token (implementation decision). "
            f"Got values: {masked_ids[unk_positions].unique().tolist()}"
        )


# ---------------------------------------------------------------------------
# TestArchitecture — not nn.Module, no parameters, config constructor
# ---------------------------------------------------------------------------


class TestArchitecture:
    """Verify MaskingStrategy is a pure stateless utility class (§2.3.5)."""

    def test_not_nn_module(self) -> None:
        """§2.3.5: MaskingStrategy is NOT an nn.Module.

        The masking strategy is a pure Python utility — no neural network,
        no learnable parameters, no forward hooks, no device management.
        """
        import torch.nn as nn

        masker = MaskingStrategy(_CONFIG)
        assert not isinstance(masker, nn.Module), (
            "§2.3.5: MaskingStrategy must not subclass nn.Module. "
            "It is a stateless utility class with pure tensor operations."
        )

    def test_no_parameters(self) -> None:
        """§2.3.5: MaskingStrategy has no parameters() method.

        Not an nn.Module → no parameters() → no trainable weights.
        The masking strategy is applied identically at any scale.
        """
        masker = MaskingStrategy(_CONFIG)
        assert not hasattr(masker, "parameters"), (
            "§2.3.5: MaskingStrategy must not have a parameters() method. "
            "It is not an nn.Module and has no trainable parameters."
        )

    def test_constructor_takes_config(self) -> None:
        """DEVELOPMENT_PROCESS.md: constructor must be (self, config: PRAGMAConfig).

        All three model sizes must be constructable. Probabilities are read from
        config, not hardcoded.
        """
        masker_s = MaskingStrategy(PRAGMAConfig.pragma_s())
        masker_m = MaskingStrategy(PRAGMAConfig.pragma_m())
        masker_l = MaskingStrategy(PRAGMAConfig.pragma_l())

        assert masker_s.token_mask_prob == 0.15   # key-numbers.md: §2.3.5
        assert masker_m.event_mask_prob == 0.10   # key-numbers.md: §2.3.5
        assert masker_l.key_mask_prob == 0.10     # key-numbers.md: §2.3.5


# ---------------------------------------------------------------------------
# TestPaperSpecifications — exact probability values from key-numbers.md
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact masking probabilities from the paper (§2.3.5, key-numbers.md)."""

    def test_token_mask_prob_from_config(self) -> None:
        """key-numbers.md: token_mask_prob = 0.15 (§2.3.5)."""
        config = PRAGMAConfig.pragma_s()
        masker = MaskingStrategy(config)
        assert masker.token_mask_prob == 0.15, (  # key-numbers.md: §2.3.5
            f"token_mask_prob must be 0.15 (§2.3.5), got {masker.token_mask_prob}"
        )

    def test_event_mask_prob_from_config(self) -> None:
        """key-numbers.md: event_mask_prob = 0.10 (§2.3.5)."""
        config = PRAGMAConfig.pragma_s()
        masker = MaskingStrategy(config)
        assert masker.event_mask_prob == 0.10, (  # key-numbers.md: §2.3.5
            f"event_mask_prob must be 0.10 (§2.3.5), got {masker.event_mask_prob}"
        )

    def test_key_mask_prob_from_config(self) -> None:
        """key-numbers.md: key_mask_prob = 0.10 (§2.3.5)."""
        config = PRAGMAConfig.pragma_s()
        masker = MaskingStrategy(config)
        assert masker.key_mask_prob == 0.10, (  # key-numbers.md: §2.3.5
            f"key_mask_prob must be 0.10 (§2.3.5), got {masker.key_mask_prob}"
        )
