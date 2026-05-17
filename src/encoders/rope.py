"""Rotary Positional Embedding (RoPE) implementation.

Implements the RoPE positional encoding used throughout the PRAGMA
encoder stack (Sections 2.3.2, 2.3.4).

RoPE encodes position information by rotating query and key vectors in
the attention mechanism, rather than adding absolute position embeddings
to token representations. This has two advantages for PRAGMA:

    1. Length generalisation: RoPE enables the model to handle sequence
       lengths at inference that differ from those seen during training.
       This is important for the History Encoder which processes variable
       lengths of transaction history.

    2. Relative position awareness: Attention scores become a function
       of relative position, capturing the temporal ordering of events
       without requiring absolute position tokens.

The PRAGMA paper uses RoPE in the Profile State Encoder (for life-long
event timestamps) and the History Encoder (for the event sequence).
The Event Encoder uses calendar token embeddings instead (Section 2.3.3).

Reference: Ostroukhov et al. (2026), Sections 2.3.2 and 2.3.4
Original RoPE paper: Su et al. (2022), arXiv:2104.09864
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for transformer attention.

    Applies position-dependent rotation to query and key vectors
    in multi-head attention. Compatible with bidirectional attention.

    Args:
        dim: Dimensionality of each attention head (d_model // n_heads).
        max_seq_len: Maximum sequence length to pre-compute. Default: 4096.
        base: Base for the geometric frequency progression. Default: 10000.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute inverse frequencies: θ_i = 1 / base^(2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute the cos/sin rotation cache."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE rotation to query and key tensors.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim).
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim).

        Returns:
            Rotated (q, k) tensors with the same shape.
        """
        seq_len = q.shape[2]

        # Extend cache if necessary
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :]  # (1, 1, seq_len, dim)
        sin = self.sin_cached[:, :, :seq_len, :]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot.to(q.dtype), k_rot.to(k.dtype)
