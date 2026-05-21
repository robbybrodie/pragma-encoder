"""Event Encoder — bidirectional Transformer with sincos calendar MLP.

Implements the Event Encoder described in PRAGMA paper Section 2.3.3.

The Event Encoder processes pre-embedded event token sequences independently
(one event at a time) and produces two outputs:
  - z_hat_e: token-level embeddings for all events (for MLM head)
  - ze:      calendar-augmented [EVT] tokens (for History Encoder)

Contract with callers:
    INPUT:
        xe: (batch, ne, ni, d_model) — event token embeddings.
            Pre-embedded via Equation 1 (E(k) + E(v) + PosEmb).
            The [EVT] token is ALREADY prepended at position 0 of each event.
            These are FLOAT embeddings, not raw integer token IDs.

        xt: (batch, ne, 3) — calendar features.
            Three integers per event: [hour_of_day, day_of_week, day_of_month].
            key-numbers.md: calendar_feature_dims=3, §2.2.

    OUTPUT:
        z_hat_e: (batch, ne, ni, d_model) — full token-level encoder output.
            Used by the MLM head during pre-training (Equation 8).
            The caller extracts z_hat_e[:,:,0,:] for the [EVT] token.
            This encoder does NOT slice — it returns the full z_hat_e.

        ze: (batch, ne, d_model) — calendar-augmented [EVT] tokens.
            Computed as ze = z'e + zt where:
              z'e = z_hat_e[:,:,0,:] — [EVT] token at position 0
              zt  = CalendarMLP(sincos(xt)) — Equation 3

Key design decisions from Section 2.3.3:
    - Bidirectional self-attention (is_causal=False) — NEVER causal
    - NO RoPE — calendar features handle temporal encoding (CLAUDE.md §3)
    - Calendar pipeline (Equation 3): xt → sincos → 2-layer MLP → zt
    - sincos BEFORE MLP — not after
    - Calendar added AFTER encoder: ze = z'e + zt — NOT before or inside
    - Pre-norm LayerNorm (Xiong et al., 2020) — norm_first pattern
    - GELU activation (Hendrycks et al., 2016)
    - Dropout = 0.1 (§2.3)
    - config.event_encoder_layers layers (5 / 16 / 45 for S / M / L)

Independence constraint (§2.3.3):
    The paper uses a FlashAttention varlen kernel to process all events in a
    single forward pass while preventing cross-event attention.

    KNOWN DEVIATION from paper production infrastructure:
    We implement the equivalent by reshaping xe from (batch, ne, ni, d_model)
    to (batch*ne, ni, d_model) before the encoder. Each event occupies its own
    batch position — the standard attention mechanism then produces no cross-event
    attention without requiring FlashAttention. We reshape back to
    (batch, ne, ni, d_model) after the encoder.

Separation of concerns (CLAUDE.md, ADR 003):
    - Embedding (Equation 1) is done OUTSIDE this class by the pipeline
    - [EVT] token is prepended OUTSIDE this class by the caller
    - Calendar uses sincos (fixed) + learned MLP, NOT nn.Embedding
    - This encoder does NOT add the [EVT] token itself

Reference: Ostroukhov et al. (2026), Section 2.3.3, Equations 3 and 5
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pragma_encoder.model.config import PRAGMAConfig


class _CalendarMLP(nn.Module):
    """Calendar feature embedding via sincos + 2-layer MLP (Equation 3).

    Converts raw calendar integer features xt ∈ R^(ne×3) into dense embeddings
    zt ∈ R^(ne×d) using a fixed sincos transform followed by a learned 2-layer MLP:

        xt (batch, ne, 3)  — integer calendar values
        → sincos           — each of 3 scalars → (sin, cos) pair → 6 values
        → Linear(6, d)     — layer 1
        → GELU
        → Linear(d, d)     — layer 2
        → zt (batch, ne, d_model)

    The sincos step normalises each calendar dimension by its known cycle
    period (hour/24, dow/7, dom/31) before applying sin/cos, giving each
    dimension a fixed period equal to the real calendar cycle (§2.3.3:
    "Periods fixed to known calendar cycles").

    Two layers: key-numbers.md: calendar_feature_embedding_layers=2, §2.3.3.
    sincos BEFORE the MLP — not after.

    Args:
        config: PRAGMAConfig — provides d_model.
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        super().__init__()
        # sincos doubles 3 calendar scalars to 6 inputs for the MLP.
        # Layer 1: 6 → d_model. Layer 2: d_model → d_model.
        # key-numbers.md: calendar_feature_embedding_layers=2, §2.3.3
        self.mlp = nn.Sequential(
            nn.Linear(6, config.d_model),        # layer 1: sincos(3) → d_model
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),  # layer 2: d_model → d_model
        )

    def forward(
        self,
        xt: torch.Tensor,  # (batch, ne, 3) — integer calendar features
    ) -> torch.Tensor:     # (batch, ne, d_model) — zt
        """Embed calendar features using sincos + 2-layer MLP (Equation 3).

        Normalises each calendar dimension by its known cycle period before
        applying sin/cos, so that the embedding is periodic with the correct
        cycle length (§2.3.3: "Periods fixed to known calendar cycles"):
            hour of day   → 2π · h   / 24
            day of week   → 2π · dow / 7
            day of month  → 2π · dom / 31

        Args:
            xt: Calendar features. Shape: (batch, ne, 3) — integers.
                Values: [hour_of_day, day_of_week, day_of_month].

        Returns:
            zt: Calendar embeddings. Shape: (batch, ne, d_model).
        """
        # Known calendar cycle periods — fixed, not learned (§2.3.3, Equation 3)
        periods = torch.tensor(
            [24.0, 7.0, 31.0], dtype=torch.float32, device=xt.device
        )  # (3,) — hour/24, dow/7, dom/31

        # Normalise to [0, 2π) before sin/cos so each dimension is periodic
        # with the correct calendar cycle length.
        xt_normed = xt.float() * (2.0 * math.pi) / periods  # (batch, ne, 3)

        # sincos: each normalised scalar → (sin, cos) pair → 6 values per event
        sincos = torch.cat(
            [torch.sin(xt_normed), torch.cos(xt_normed)],
            dim=-1,
        )  # (batch, ne, 6)
        return self.mlp(sincos)  # type: ignore[no-any-return]


class _EventAttention(nn.Module):
    """Multi-head self-attention for EventEncoder (no RoPE).

    Bidirectional attention within a single event's token sequence.
    No temporal coordinate rotation — EventEncoder uses calendar features
    instead of RoPE (CLAUDE.md §3, §2.3.3).

    Input is (batch*ne, ni, d_model) — events flattened into the batch dimension
    to enforce the independence constraint. Each event in the flattened batch
    attends only to its own tokens.

    Args:
        config: PRAGMAConfig — provides d_model, n_heads, dropout.
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        super().__init__()
        self.n_heads: int = config.n_heads
        self.head_dim: int = config.d_model // config.n_heads  # 64 — key-numbers.md
        self.d_model: int = config.d_model
        self._dropout: float = config.dropout

        # QKV and output projections — bias=True (standard)
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        x: torch.Tensor,                           # (batch*ne, ni, d_model) — flattened events
        attn_mask: Optional[torch.Tensor] = None,  # (batch*ne, 1, 1, ni) bool — True=attend
    ) -> torch.Tensor:                             # (batch*ne, ni, d_model)
        bne, ni, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bne, ni, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = _to_heads(q), _to_heads(k), _to_heads(v)

        # Bidirectional attention — is_causal=False.
        # PRAGMA is encoder-only. NEVER use causal masking here. (CLAUDE.md)
        # attn_mask: boolean key-padding mask (batch*ne, 1, 1, ni).
        #   True  = position is real (attend normally)
        #   False = position is padding (set to -inf before softmax)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self._dropout if self.training else 0.0,
            is_causal=False,
        )  # (batch*ne, n_heads, ni, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bne, ni, self.d_model)
        return self.out_proj(attn_out)  # type: ignore[no-any-return]


class _EventEncoderLayer(nn.Module):
    """One pre-norm Transformer encoder layer for EventEncoder (no RoPE).

    Pre-norm (Xiong et al., 2020) pattern:
        x = x + dropout(attn(norm1(x)))   # attention sub-layer
        x = x + dropout(ff(norm2(x)))     # feed-forward sub-layer

    Activation: GELU (§2.3, key-numbers.md).
    Normalisation: pre-norm LayerNorm (§2.3, key-numbers.md).
    No RoPE — EventEncoder uses calendar features for temporal encoding.

    Args:
        config: PRAGMAConfig — provides all hyperparameters.
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attn = _EventAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ffn),
            nn.GELU(),
            nn.Linear(config.d_ffn, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,                           # (batch*ne, ni, d_model)
        attn_mask: Optional[torch.Tensor] = None,  # (batch*ne, 1, 1, ni) bool — True=attend
    ) -> torch.Tensor:                             # (batch*ne, ni, d_model)
        # Pre-norm attention + residual (§2.3: pre-norm LayerNorm)
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask=attn_mask))
        # Pre-norm feed-forward + residual (§2.3: GELU activation)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class EventEncoder(nn.Module):
    """Bidirectional Transformer encoder for financial event token sequences.

    Processes pre-embedded event token sequences xe (with [EVT] already at
    position 0 of each event) independently and produces two outputs:
      - z_hat_e: full token-level encoder output (used by MLM head)
      - ze:      calendar-augmented [EVT] tokens (used by History Encoder)

    The calendar embedding (Equation 3) is applied AFTER the encoder:
      z'e = z_hat_e[:,:,0,:]   — [EVT] token at position 0 of each event
      zt  = MLP(sincos(xt))    — calendar feature embedding
      ze  = z'e + zt           — addition, NOT concatenation

    Independence constraint — KNOWN DEVIATION from paper production:
    The paper uses a FlashAttention varlen kernel. We implement the equivalent
    by reshaping xe from (batch, ne, ni, d_model) to (batch*ne, ni, d_model)
    before the encoder. Each event is a separate batch item — no cross-event
    attention occurs. We reshape back after the encoder.

    Args:
        config: PRAGMAConfig — single source of truth for all hyperparameters.
                config.event_encoder_layers controls the stack depth
                (5 for PRAGMA-S, 16 for PRAGMA-M, 45 for PRAGMA-L — Table 1).
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        super().__init__()
        self.d_model: int = config.d_model  # exposed for callers and tests

        # Encoder stack — no RoPE (calendar handles temporal encoding, §2.3.3)
        # Depth from config.event_encoder_layers (5 / 16 / 45 — Table 1)
        self.layers = nn.ModuleList([
            _EventEncoderLayer(config)
            for _ in range(config.event_encoder_layers)
        ])

        # Final layer norm applied to the full token-level output
        self.norm = nn.LayerNorm(config.d_model)

        # Input dropout applied to pre-embedded xe before the encoder stack
        self.dropout = nn.Dropout(config.dropout)

        # Calendar feature embedding (Equation 3): sincos + 2-layer MLP
        # key-numbers.md: calendar_feature_embedding_layers=2, §2.3.3
        self.calendar_mlp = _CalendarMLP(config)

    def forward(
        self,
        xe: torch.Tensor,                          # (batch, ne, ni, d_model) — pre-embedded; [EVT] at pos 0  # noqa: E501
        xt: torch.Tensor,                          # (batch, ne, 3) — calendar features (integers)
        xe_valid: Optional[torch.Tensor] = None,   # (batch, ne, ni) bool — True=real token, False=padding  # noqa: E501
    ) -> Tuple[torch.Tensor, torch.Tensor]:        # (z_hat_e, ze)
        """Encode event token sequences independently with calendar augmentation.

        Args:
            xe: Event token embeddings. Shape: (batch, ne, ni, d_model).
                Float tensor — pre-embedded via Equation 1 by the pipeline.
                The [EVT] token is already at position 0 of each event.
            xt: Calendar features. Shape: (batch, ne, 3) — integers.
                Three values per event: [hour_of_day, day_of_week, day_of_month].
                key-numbers.md: calendar_feature_dims=3, §2.2.
            xe_valid: Token-level validity mask. Shape: (batch, ne, ni) bool. Optional.
                      True = real token, False = padding. When provided, padding
                      positions are masked out from attention (set to -inf before
                      softmax) so they cannot influence real token outputs. Positions
                      beyond the real token count for each event should be False.

        Returns:
            z_hat_e: Full token-level encoder output. Shape: (batch, ne, ni, d_model).
                     Used by MLM head during pre-training (Equation 8).
                     Caller slices z_hat_e[:,:,0,:] for [EVT] tokens.
            ze:      Calendar-augmented [EVT] tokens. Shape: (batch, ne, d_model).
                     ze = z'e + zt (Equation 3). Used by History Encoder input.
        """
        batch, ne, ni, _ = xe.shape

        # Independence constraint: flatten events into batch dimension.
        # (batch, ne, ni, d_model) → (batch*ne, ni, d_model)
        # Each event is now a separate batch item — no cross-event attention.
        # KNOWN DEVIATION: paper uses FlashAttention varlen; we use reshape.
        x = self.dropout(xe.view(batch * ne, ni, self.d_model))

        # Build attention key-padding mask from xe_valid when provided.
        # xe_valid: (batch, ne, ni) → (batch*ne, ni) → (batch*ne, 1, 1, ni)
        # Boolean mask: True = real token (attend), False = padding (-inf).
        attn_mask: Optional[torch.Tensor] = None
        if xe_valid is not None:
            attn_mask = xe_valid.view(batch * ne, ni).unsqueeze(1).unsqueeze(2)

        # Pass through each encoder layer — bidirectional, no RoPE
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # Final layer norm over all token positions
        x = self.norm(x)

        # Restore event structure: (batch*ne, ni, d_model) → (batch, ne, ni, d_model)
        z_hat_e = x.view(batch, ne, ni, self.d_model)

        # z'e: [EVT] token at position 0 of each event (batch, ne, d_model)
        # The caller prepended [EVT] at position 0 — we slice it here for ze.
        z_prime_e = z_hat_e[:, :, 0, :]  # (batch, ne, d_model)

        # zt: calendar feature embedding (Equation 3)
        # xt → sincos → 2-layer MLP → zt
        zt = self.calendar_mlp(xt)  # (batch, ne, d_model)

        # ze: calendar-augmented [EVT] tokens (§2.3.3, Equation 3)
        # Addition — NOT concatenation. ze and z'e have the same shape.
        ze = z_prime_e + zt  # (batch, ne, d_model)

        return z_hat_e, ze
