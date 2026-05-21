"""MaskingStrategy — three-strategy masked event modelling (§2.3.5).

Implements the unified masking strategy from PRAGMA paper Section 2.3.5.

PRAGMA extends the standard MLM objective with three masking granularities
applied jointly in a single forward pass:

    Token masking   (token_mask_prob = 0.15): Bernoulli per position.
    Event masking   (event_mask_prob = 0.10): Bernoulli per event, ALL ni
                    tokens within a selected event are masked together.
    Key-type masking (key_mask_prob  = 0.10): Bernoulli per unique key type;
                    ALL positions across (batch, ne, ni) where key_ids == k
                    are masked together when key type k is selected.

The three strategies are ORed (union) before applying corruption:
    selected = token_selected | event_selected | key_selected

Corruption of selected positions:
    (1 - _UNK_FRACTION) → [MASK] token (_MASK_TOKEN_ID = 1); mask=True
    _UNK_FRACTION       → [UNK]  token (_UNK_TOKEN_ID  = 0); mask=False

The mask output is a boolean tensor:
    True  = position replaced with [MASK] — included in MLM loss
    False = position unchanged OR replaced with [UNK] — excluded from loss

Implementation notes:
    - MaskingStrategy is NOT nn.Module. It is a plain Python class with no
      learnable parameters. It can be constructed and called from any context
      without PyTorch device management.
    - _UNK_TOKEN_ID = TokenizerPipeline.PAD_ID = 0. There is no global [UNK]
      token in TokenizerPipeline; PAD_ID is used as the corruption token.
      This is an implementation decision (the paper specifies [UNK] corruption
      but TokenizerPipeline only defines PAD, MASK, CLS, SEP as specials).
    - _UNK_FRACTION = 0.10 is a module constant, not a PRAGMAConfig field.
      The paper does not specify this fraction explicitly; 0.10 is adopted as
      the implementation default consistent with the MLM literature.
    - Overlap: positions selected by more than one strategy are masked once
      (union). This is the simplest correct interpretation.
    - Key-type selection: one Bernoulli sample PER UNIQUE KEY TYPE across the
      entire (batch, ne, ni) tensor. All positions sharing that key type are
      masked together regardless of batch or event index.

Reference: Ostroukhov et al. (2026), Section 2.3.5
"""

from typing import Tuple

import torch

from pragma_encoder.model.config import PRAGMAConfig
from pragma_encoder.tokenizer.pipeline import TokenizerPipeline

# Token IDs — sourced from TokenizerPipeline
_MASK_TOKEN_ID: int = TokenizerPipeline.MASK_ID  # 1 — [MASK] replacement
_UNK_TOKEN_ID:  int = TokenizerPipeline.PAD_ID   # 0 — [UNK] replacement (no global UNK)

# Fraction of selected positions replaced with [UNK] instead of [MASK].
# These positions are excluded from the MLM loss (mask=False).
# Implementation decision — not a PRAGMAConfig field.
_UNK_FRACTION: float = 0.10


class MaskingStrategy:
    """Unified three-strategy masking for PRAGMA masked event modelling (§2.3.5).

    Applies token-level, event-level, and semantic-type (key) masking in a
    single forward pass. All three strategies are ORed (union) to produce the
    final set of selected positions.

    This class is NOT an nn.Module. It has no learnable parameters and no
    parameters() method. It is a pure Python utility with stateless tensor
    operations.

    Args:
        config: PRAGMAConfig — provides token_mask_prob, event_mask_prob,
                key_mask_prob. All three probabilities are read from config;
                none are hardcoded in this class.

    Attributes:
        token_mask_prob: Probability of masking each token position (§2.3.5).
        event_mask_prob: Probability of masking each event (§2.3.5).
        key_mask_prob:   Probability of masking each unique key type (§2.3.5).
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        self.token_mask_prob: float = config.token_mask_prob
        self.event_mask_prob: float = config.event_mask_prob
        self.key_mask_prob:   float = config.key_mask_prob

    # ------------------------------------------------------------------
    # Private strategy methods — each returns a bool (batch, ne, ni) mask
    # ------------------------------------------------------------------

    def _token_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Bernoulli masking at each token position independently.

        Selects each position with probability token_mask_prob.

        Args:
            token_ids: (batch, ne, ni) — integer token IDs (values not used).

        Returns:
            selected: (batch, ne, ni) bool — True at positions to mask.
        """
        prob = torch.full(
            token_ids.shape, self.token_mask_prob,
            dtype=torch.float, device=token_ids.device,
        )
        return torch.bernoulli(prob).bool()

    def _event_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Event-level masking — whole events selected atomically.

        Each event (ne dimension) is selected independently with probability
        event_mask_prob. When an event is selected, ALL ni token positions
        within that event are masked.

        Args:
            token_ids: (batch, ne, ni) — shape used; values not used.

        Returns:
            selected: (batch, ne, ni) bool — True at all positions of
                      selected events; False for unselected events.
        """
        batch, ne, ni = token_ids.shape
        prob = torch.full(
            (batch, ne), self.event_mask_prob,
            dtype=torch.float, device=token_ids.device,
        )
        event_selected = torch.bernoulli(prob).bool()  # (batch, ne)
        # Expand: every token position in a selected event is True
        return event_selected.unsqueeze(-1).expand(batch, ne, ni)

    def _key_mask(self, key_ids: torch.Tensor) -> torch.Tensor:
        """Semantic-type (key) masking — all positions sharing a key type.

        For each unique key type k in key_ids, a single Bernoulli sample
        determines whether the entire key type is selected. When selected,
        ALL positions across (batch, ne, ni) where key_ids == k are masked.

        Args:
            key_ids: (batch, ne, ni) — integer key type IDs.

        Returns:
            selected: (batch, ne, ni) bool — True at all positions whose
                      key type was selected for masking.
        """
        selected = torch.zeros_like(key_ids, dtype=torch.bool)
        unique_keys = key_ids.unique()  # type: ignore[no-untyped-call]
        for k in unique_keys:
            if torch.bernoulli(torch.tensor(self.key_mask_prob)).item():
                selected |= (key_ids == k)
        return selected

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,  # (batch, ne, ni) — integer token IDs
        key_ids:   torch.Tensor,  # (batch, ne, ni) — integer key type IDs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply three-strategy masking in a single forward pass.

        Combines token-level, event-level, and key-type masking via union.
        A fraction (_UNK_FRACTION = 0.10) of selected positions are replaced
        with [UNK] (PAD_ID = 0) as input dropout; the remainder are replaced
        with [MASK] (MASK_ID = 1) and included in the MLM loss.

        Args:
            token_ids: Input token IDs. Shape: (batch, ne, ni). Integer dtype.
            key_ids:   Semantic type IDs per token position. Shape: (batch, ne, ni).

        Returns:
            masked_ids:  (batch, ne, ni) — token_ids with [MASK] or [UNK] at
                         selected positions; unchanged elsewhere. Same dtype as
                         token_ids.
            target_ids:  (batch, ne, ni) — original token_ids unchanged. Used
                         as the label tensor for the MLM loss. Always equals
                         token_ids.
            mask:        (batch, ne, ni) bool — True at positions replaced with
                         [MASK] (included in MLM loss). False at [UNK] positions
                         and at unmasked positions (excluded from loss).
        """
        # target_ids = original token IDs, unmodified (MLM label tensor)
        target_ids = token_ids.clone()
        masked_ids = token_ids.clone()

        # Union of all three strategies → set of positions to corrupt
        token_selected = self._token_mask(token_ids)
        event_selected = self._event_mask(token_ids)
        key_selected   = self._key_mask(key_ids)
        selected = token_selected | event_selected | key_selected

        # Split selected into [MASK] positions (in loss) and [UNK] positions (not in loss)
        unk_gate = torch.rand(token_ids.shape, device=token_ids.device) < _UNK_FRACTION
        unk_selected  = selected & unk_gate   # True → replace with UNK, exclude from loss
        mask_selected = selected & ~unk_gate  # True → replace with MASK, include in loss

        # Apply corruption
        masked_ids[mask_selected] = _MASK_TOKEN_ID  # 1 — [MASK]
        masked_ids[unk_selected]  = _UNK_TOKEN_ID   # 0 — [UNK] (PAD_ID)

        # mask: True = in MLM loss ([MASK] positions only)
        mask = mask_selected

        return masked_ids, target_ids, mask
