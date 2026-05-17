"""Masking package — three-strategy masked event modelling (§2.3.5).

Implements the unified MaskingStrategy described in PRAGMA paper Section 2.3.5.
PRAGMA extends the standard MLM objective with three masking granularities:

    1. Token masking:    Mask individual token positions (Bernoulli per position).
    2. Event masking:    Mask all tokens of entire events atomically.
    3. Key-type masking: Mask all tokens sharing the same semantic key type.

The three strategies are applied jointly in a single forward() call and ORed
(union) to produce the final selected positions.

MaskingStrategy is NOT an nn.Module — it is a plain Python utility class with
no learnable parameters. It is constructed with a PRAGMAConfig and can be
used in any training or evaluation context.

Reference: Ostroukhov et al. (2026), Section 2.3.5
"""

from typing import Tuple

import torch

from .strategy import MaskingStrategy


class MaskingStrategyProtocol:
    """Interface contract for MaskingStrategy (§2.3.5).

    Describes the forward() signature for three-strategy masked event
    modelling. MaskingStrategy is NOT an nn.Module — this protocol documents
    the expected API for callers and type checkers.

    Input:
        token_ids: (batch, ne, ni) — integer token IDs
        key_ids:   (batch, ne, ni) — integer semantic type IDs

    Output:
        masked_ids:  (batch, ne, ni) — token IDs with [MASK] or [UNK] applied
        target_ids:  (batch, ne, ni) — original token IDs (for MLM loss)
        mask:        (batch, ne, ni) bool — True at [MASK] positions (in loss)
    """

    def forward(
        self,
        token_ids: torch.Tensor,  # (batch, ne, ni)
        key_ids:   torch.Tensor,  # (batch, ne, ni)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply three-strategy masking in a single forward pass (§2.3.5)."""
        ...


__all__ = [
    "MaskingStrategy",
    "MaskingStrategyProtocol",
]
