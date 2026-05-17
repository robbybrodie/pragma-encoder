"""Encoder package — three-encoder architecture from PRAGMA Section 2.3.

Implements the three separate bidirectional Transformer encoders described
in PRAGMA paper Sections 2.3.2–2.3.4:

    ProfileStateEncoder — static customer attributes + RoPE (Section 2.3.2)
    EventEncoder        — per-transaction fields + calendar embeddings (2.3.3)
    HistoryEncoder      — sequence of event representations + RoPE (2.3.4)

Key architectural constraints (from the paper):
    - All encoders are BIDIRECTIONAL (not causal)
    - All encoders use RoPE positional encoding (not sinusoidal/absolute)
    - No shared weights between the three encoders

Reference: Ostroukhov et al. (2026), Sections 2.3.2–2.3.4
"""

from .event_encoder import EventEncoder
from .history_encoder import HistoryEncoder
from .profile_state_encoder import ProfileStateEncoder
from .rope import RotaryPositionalEmbedding

__all__ = [
    "ProfileStateEncoder",
    "EventEncoder",
    "HistoryEncoder",
    "RotaryPositionalEmbedding",
]
