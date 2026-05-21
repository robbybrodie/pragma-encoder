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


class ProfileStateEncoderProtocol(Protocol):
    """Interface contract for ProfileStateEncoder (§2.3.2, Equation 4).

    Bidirectional Transformer encoder for static customer profile state.
    Accepts pre-embedded float tensors (NOT integer token IDs).
    The [USR] token is already at position 0 of xa (prepended by caller).
    Returns the full output sequence za — caller extracts za[:,0:1,:].

    Key constraints:
        - Bidirectional attention (is_causal=False) — NEVER causal
        - RoPE applied to Q and K using temporal coordinates ta
        - No nn.Embedding — embedding is done externally (Equation 1)
        - No self.usr_token — [USR] is prepended by the caller
    """

    def forward(
        self,
        xa: torch.Tensor,  # (batch, na, d_model) — pre-embedded; [USR] at pos 0
        ta: torch.Tensor,  # (batch, na) — temporal coordinates (log-seconds, Eq 2)
    ) -> torch.Tensor:     # (batch, na, d_model) — full encoder output za
        """Encode profile state token embeddings with temporal RoPE (Eq 4)."""
        ...


class EventEncoderProtocol(Protocol):
    """Interface contract for EventEncoder (§2.3.3, Equations 3 and 5).

    Bidirectional Transformer encoder for event token sequences.
    Each event processed independently (no cross-event attention).
    Accepts pre-embedded float tensors (NOT integer token IDs).
    The [EVT] token is already at position 0 of each event (prepended by caller).
    Returns TWO tensors: z_hat_e (token-level) and ze (calendar-augmented [EVT]).

    Key constraints:
        - No RoPE — calendar features handle temporal encoding (CLAUDE.md §3)
        - Calendar (Eq 3): sincos → 2-layer MLP → zt; added AFTER encoder
        - ze = z'e + zt (addition, NOT concatenation)
        - No nn.Embedding — embedding done externally (Equation 1)
        - No self.evt_token — [EVT] prepended by the caller
        - Independence: events processed separately (reshape to batch*ne)
    """

    def forward(
        self,
        xe: torch.Tensor,  # (batch, ne, ni, d_model) — pre-embedded; [EVT] at pos 0
        xt: torch.Tensor,  # (batch, ne, 3) — calendar features (integers)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # z_hat_e: (batch, ne, ni, d_model) — token-level encoder output
        # ze:      (batch, ne, d_model)      — calendar-augmented [EVT] tokens
        """Encode event sequences independently with calendar augmentation (Eq 3, 5)."""
        ...


class HistoryEncoderProtocol(Protocol):
    """Interface contract for HistoryEncoder (§2.3.4, Equations 6 and 7).

    Bidirectional Transformer encoder for the full transaction history.
    Receives the concatenated [USR:EVT] sequence z (Equation 6) assembled by
    the caller. Produces full contextualised output zh (Equation 7).

    Key constraints:
        - z is assembled by caller: z[:,0:1,:] = za, z[:,1:,:] = ze (Eq 6)
        - te[:,0] = 0.0 for [USR]; te[:,1:] = log-seconds for [EVT] (Eq 2)
        - Bidirectional self-attention (is_causal=False) — NEVER causal
        - RoPE applied to Q and K using te in every attention layer (Eq 9)
        - Pure self-attention — NO cross-attention sublayer
        - No self.summary_token — [USR] at position 0 is the user-level representation
        - No self.inline_mask_emb — masking done externally by MaskingStrategy
        - Returns full zh — caller slices zh[:,0,:] or zh[:,1:,:]
    """

    def forward(
        self,
        z:  torch.Tensor,  # (batch, 1+ne, d_model) — [USR:EVT] concatenated (Eq 6)
        te: torch.Tensor,  # (batch, 1+ne) — temporal coordinates (log-seconds, Eq 2)
    ) -> torch.Tensor:     # (batch, 1+ne, d_model) — zh, full encoder output (Eq 7)
        """Encode the [USR:EVT] history sequence with temporal RoPE (Eq 7)."""
        ...


__all__ = [
    "ProfileStateEncoder",
    "EventEncoder",
    "HistoryEncoder",
    "RoPEEncoding",
    "RoPEEncodingProtocol",
    "ProfileStateEncoderProtocol",
    "EventEncoderProtocol",
    "HistoryEncoderProtocol",
]
