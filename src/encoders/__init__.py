"""Encoder package — three-encoder architecture from PRAGMA Section 2.3.

Implements the three separate bidirectional Transformer encoders described
in PRAGMA paper Sections 2.3.2–2.3.4:

    ProfileStateEncoder — static customer attributes + RoPE (Section 2.3.2)
    EventEncoder        — per-transaction fields + calendar embeddings (2.3.3)
    HistoryEncoder      — sequence of event representations + RoPE (2.3.4)

Key architectural constraints (from the paper):
    - All encoders are BIDIRECTIONAL (not causal)
    - RoPE uses continuous temporal coordinates (Equation 2), not integer indices
    - No shared weights between the three encoders

Reference: Ostroukhov et al. (2026), Sections 2.3.2–2.3.4
"""

from typing import Protocol, Tuple

import torch

from .event_encoder import EventEncoder
from .history_encoder import HistoryEncoder
from .profile_state_encoder import ProfileStateEncoder
from .rope import RoPEEncoding


class RoPEEncodingProtocol(Protocol):
    """Interface contract for RoPEEncoding (§2.3.2, Equation 9).

    RoPE rotates query and key vectors using continuous temporal coordinates
    (log-seconds from Equation 2), not sequential integer indices.

    Key property: the dot product between rotated q at tm and rotated k at tn
    depends only on the relative distance (tn − tm), not absolute positions.
    """

    def forward(
        self,
        q: torch.Tensor,          # (batch, n_heads, seq_len, head_dim)
        k: torch.Tensor,          # (batch, n_heads, seq_len, head_dim)
        positions: torch.Tensor,  # (batch, seq_len) or (seq_len,) — log-seconds
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # rotated (q, k) — same shapes
        """Apply RoPE rotation using continuous temporal coordinates (Eq 9)."""
        ...


__all__ = [
    "ProfileStateEncoder",
    "EventEncoder",
    "HistoryEncoder",
    "RoPEEncoding",
    "RoPEEncodingProtocol",
]
