"""Rotary Positional Embedding (RoPE) for the PRAGMA encoder stack.

Implements the RoPE positional encoding described in PRAGMA paper Sections
2.3.2 and 2.3.4, following the original formulation of Su et al. (2024).

RoPE encodes position information by rotating query and key vectors in the
attention mechanism using continuous temporal coordinates (log-seconds),
rather than sequential integer indices.

Key property (Equation 9):
    attention(qm, kn) ∝ qm^T kn = q^T R(tm)^T R(tn) k = q^T R(tn − tm) k

    The dot product depends only on the RELATIVE temporal distance (tn − tm),
    not on absolute positions. Closer events produce higher dot-product
    similarity. This is what gives PRAGMA temporal awareness.

Usage in PRAGMA:
    ProfileStateEncoder — RoPE on ta (log-seconds since life-long event)
    HistoryEncoder      — RoPE on te (log-seconds to most recent event)
    EventEncoder        — does NOT use RoPE (uses calendar features instead)

Critical deviation from naïve RoPE:
    Standard RoPE implementations use integer positions 0, 1, 2, ...
    PRAGMA uses CONTINUOUS temporal coordinates from Equation 2:
        t' = 8·ln(1 + t/8)
    These are passed directly to forward() as a float tensor.
    The sequential integer cache approach would encode token order,
    not temporal distance — a silent architectural error.

Separation from within-field PosEmb (ADR 003):
    RoPE operates on temporal coordinates. Within-field positions
    use standard sine/cosine PosEmb (Equation 1). These must not mix.

Reference: Ostroukhov et al. (2026), Sections 2.3.2 and 2.3.4
Original RoPE paper: Su et al. (2024), arXiv:2104.09864
"""

from typing import Tuple

import torch
import torch.nn as nn

from pragma_encoder.model.config import PRAGMAConfig


class RoPEEncoding(nn.Module):
    """Rotary Positional Embedding — applies temporal coordinate rotations.

    Rotates query and key vectors by angles derived from continuous temporal
    coordinates (log-seconds), so that attention scores reflect temporal
    proximity rather than token-index order.

    Has NO trainable parameters — inv_freq is a fixed buffer. Rotation
    angles are determined entirely by the temporal coordinates and the
    geometric frequency progression (base=10000, Su et al. 2024).

    Args:
        config: PRAGMAConfig instance. head_dim is derived as
                config.d_model // config.n_heads (= 64 for all PRAGMA sizes).
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        super().__init__()
        self.head_dim: int = config.d_model // config.n_heads  # 64 — key-numbers.md Table 1

        # Inverse frequencies: θ_i = 1 / 10000^(2i / head_dim)
        # Shape: (head_dim // 2,) — one frequency per dimension pair.
        # Registered as a buffer (not a Parameter) so it moves with .to(device)
        # but receives no gradient and appears in no optimiser update.
        inv_freq = 1.0 / (
            10_000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension into the first half.

        For x = [x1 | x2] (two equal halves):
            _rotate_half(x) = [-x2 | x1]

        Combined with the cos/sin application this produces the 2D rotation:
            [x1*cos - x2*sin, x2*cos + x1*sin]
        for each (x[i], x[i + head_dim/2]) dimension pair.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,          # (batch, n_heads, seq_len, head_dim)
        k: torch.Tensor,          # (batch, n_heads, seq_len, head_dim)
        positions: torch.Tensor,  # (batch, seq_len) or (seq_len,) — temporal coords (log-seconds)
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # rotated (q, k) — same shapes as input
        """Apply RoPE rotation using continuous temporal coordinates.

        Computes rotation angles directly from the provided temporal
        coordinates (log-seconds). This encodes temporal distance, not
        token-index order — the key distinction from integer-position RoPE.

        Equation 9 property:
            q_rot[m]^T k_rot[n] = q^T R(tn − tm) k
            (dot product depends only on relative temporal distance)

        Args:
            q:         Query tensor.  Shape: (batch, n_heads, seq_len, head_dim)
            k:         Key tensor.    Shape: (batch, n_heads, seq_len, head_dim)
            positions: Temporal coordinates in log-seconds (continuous floats
                       from Equation 2: t' = 8·ln(1+t/8)).
                       Shape: (batch, seq_len) or (seq_len,)

        Returns:
            Tuple of rotated (q, k) tensors, same shapes as input.
        """
        # Normalise to (batch, seq_len) if a 1-D positions vector is passed
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # (1, seq_len)

        # Compute per-position, per-frequency rotation angles.
        # freqs: (batch, seq_len, head_dim // 2)
        freqs = torch.einsum("bs,d->bsd", positions.float(), self.inv_freq)

        # Duplicate frequencies for both halves of head_dim (cos/sin interleave).
        # emb: (batch, seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Expand over the n_heads dimension for broadcasting with (batch, n_heads, seq_len, head_dim).  # noqa: E501
        cos = emb.cos().unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = emb.sin().unsqueeze(1)  # (batch, 1, seq_len, head_dim)

        # Apply rotation: x_rot = x * cos + rotate_half(x) * sin
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot.to(q.dtype), k_rot.to(k.dtype)
