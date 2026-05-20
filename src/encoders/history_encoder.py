"""History Encoder — bidirectional Transformer with RoPE on event timestamps.

Implements the History Encoder described in PRAGMA paper Section 2.3.4.

The History Encoder processes the concatenated sequence z = [za : ze] (Equation 6)
assembled by the caller and produces a full contextualised output zh (Equation 7).

Contract with callers:
    INPUT:
        z:  (batch, 1+ne, d_model) — concatenated [USR:EVT] sequence.
            Assembled by the caller BEFORE calling forward():
              z[:,0:1,:]  = za — the [USR] token from ProfileStateEncoder
              z[:,1:,:]   = ze — [EVT] tokens from EventEncoder
            The HistoryEncoder receives z already assembled.
            It does NOT concatenate za and ze itself.

        te: (batch, 1+ne) — temporal coordinates in log-seconds.
            te[:,0]   = 0.0 — [USR] position (no timestamp; RoPE identity)
            te[:,1:]  = log-seconds to most recent event per §2.3.4
            Computed via Equation 2: t' = 8·ln(1 + t/8).
            Fed into RoPEEncoding for every attention layer.

    OUTPUT:
        zh: (batch, 1+ne, d_model) — full history encoder output sequence.
            zh[:,0,:]  = [USR] representation (user level)
            zh[:,1:,:] = [EVT] representations (per-event level)
            The caller slices as needed — this encoder does NOT slice.

Key design decisions from Section 2.3.4:
    - Bidirectional self-attention (is_causal=False) — NEVER causal
    - RoPE applied to Q and K in EVERY attention layer, using te
    - Pure self-attention — NO cross-attention sublayer
    - [USR] at position 0 conditions [EVT] tokens through self-attention naturally
    - Pre-norm LayerNorm (Xiong et al., 2020) — norm_first pattern
    - GELU activation (Hendrycks et al., 2016)
    - Dropout = 0.1 (§2.3)
    - config.history_encoder_layers layers (2 / 6 / 18 for S / M / L)

Separation of concerns (CLAUDE.md, ADR 003):
    - Concatenation z=[za:ze] is done OUTSIDE by the caller (Equation 6)
    - No dedicated summary token — [USR] at position 0 is the user-level representation
    - No inline mask embedding — masking is done OUTSIDE by MaskingStrategy
    - RoPE (temporal on te) is SEPARATE from within-field PosEmb (Equation 1)

Profile conditioning mechanism:
    The stub used cross-attention from [EVT] positions to the [USR] vector.
    The paper specifies NO separate cross-attention sublayer.
    Instead, [USR] is prepended at position 0 of z. Bidirectional self-attention
    then allows every [EVT] position to attend to [USR] directly — profile
    information conditions history encoding through the standard attention
    mechanism without any extra architectural components.

Reference: Ostroukhov et al. (2026), Section 2.3.4, Equations 6 and 7
RoPE: Su et al. (2024), arXiv:2104.09864
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.rope import RoPEEncoding
from src.model.config import PRAGMAConfig


class _RoPEMultiheadAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K.

    Applies temporal coordinate rotations (Equation 9) to query and key
    vectors before computing bidirectional scaled dot-product attention.
    RoPE is applied once per attention layer using the temporal coordinates
    te passed from HistoryEncoder.forward().

    Identical pattern to ProfileStateEncoder._RoPEMultiheadAttention —
    the two encoders share the same attention mechanism, but have independent
    weights and use different temporal coordinates (ta vs te).

    Args:
        config: PRAGMAConfig — provides d_model, n_heads, dropout.
        rope:   Shared RoPEEncoding instance (no trainable parameters).
    """

    def __init__(self, config: PRAGMAConfig, rope: RoPEEncoding) -> None:
        super().__init__()
        self.n_heads: int = config.n_heads
        self.head_dim: int = config.d_model // config.n_heads  # 64 — key-numbers.md
        self.d_model: int = config.d_model
        self.rope: RoPEEncoding = rope
        self._dropout: float = config.dropout

        # QKV and output projections — bias=True (standard)
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        x: torch.Tensor,                           # (batch, 1+ne, d_model)
        te: torch.Tensor,                          # (batch, 1+ne) — temporal coordinates (log-seconds)  # noqa: E501
        attn_mask: Optional[torch.Tensor] = None,  # (batch, 1, 1, 1+ne) bool — True=attend
    ) -> torch.Tensor:                             # (batch, 1+ne, d_model)
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = _to_heads(q)  # (batch, n_heads, 1+ne, head_dim)
        k = _to_heads(k)
        v = _to_heads(v)

        # Apply RoPE rotation to Q and K using temporal coordinates.
        # te[:,0] = 0 → identity rotation for [USR] position (RoPE at t=0 is I).
        # te[:,1:] = log-seconds → relative temporal distance for [EVT] positions.
        # Equation 9: dot(q_rot[m], k_rot[n]) = q^T R(tn - tm) k.
        q, k = self.rope(q, k, te)

        # Bidirectional scaled dot-product attention — is_causal=False.
        # PRAGMA is encoder-only. NEVER use causal masking here. (CLAUDE.md)
        # attn_mask: boolean key-padding mask (batch, 1, 1, 1+ne).
        #   True  = position is real (attend normally)
        #   False = position is padding (set to -inf before softmax)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self._dropout if self.training else 0.0,
            is_causal=False,
        )  # (batch, n_heads, 1+ne, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(attn_out)


class _HistoryEncoderLayer(nn.Module):
    """One pre-norm Transformer encoder layer with RoPE attention.

    Implements the standard pre-norm (Xiong et al., 2020) pattern:
        z = z + dropout(attn(norm1(z), te))   # attention sub-layer
        z = z + dropout(ff(norm2(z)))          # feed-forward sub-layer

    No cross-attention sub-layer — pure self-attention only (§2.3.4).
    [USR] at position 0 of z conditions [EVT] tokens through self-attention.
    Activation: GELU (§2.3, key-numbers.md).

    Args:
        config: PRAGMAConfig — provides all hyperparameters.
        rope:   Shared RoPEEncoding instance passed in from HistoryEncoder.
    """

    def __init__(self, config: PRAGMAConfig, rope: RoPEEncoding) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attn = _RoPEMultiheadAttention(config, rope)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ffn),
            nn.GELU(),
            nn.Linear(config.d_ffn, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        z: torch.Tensor,                           # (batch, 1+ne, d_model)
        te: torch.Tensor,                          # (batch, 1+ne) — temporal coordinates
        attn_mask: Optional[torch.Tensor] = None,  # (batch, 1, 1, 1+ne) bool — True=attend
    ) -> torch.Tensor:                             # (batch, 1+ne, d_model)
        # Pre-norm attention + residual (§2.3: pre-norm LayerNorm)
        z = z + self.dropout(self.attn(self.norm1(z), te, attn_mask=attn_mask))
        # Pre-norm feed-forward + residual (§2.3: GELU activation)
        z = z + self.dropout(self.ff(self.norm2(z)))
        return z


class HistoryEncoder(nn.Module):
    """Bidirectional Transformer encoder for the full transaction history.

    Processes the concatenated [USR:EVT] sequence z (Equation 6) with temporal
    RoPE on te, producing a full contextualised output zh (Equation 7).

    The caller assembles z = [za : ze] before calling forward():
      - z[:,0:1,:] = za — [USR] token from ProfileStateEncoder
      - z[:,1:,:]  = ze — [EVT] tokens from EventEncoder

    This encoder does NOT concatenate za and ze — that is the caller's job.
    This encoder does NOT slice zh — the caller extracts what it needs:
      - zh[:,0,:]  → [USR] representation → embedding probe / fine-tuning head
      - zh[:,1:,:] → [EVT] representations → MLM head

    Profile conditioning: [USR] at position 0 naturally conditions all [EVT]
    positions through bidirectional self-attention. No cross-attention is needed.

    Args:
        config: PRAGMAConfig — single source of truth for all hyperparameters.
                config.history_encoder_layers controls the stack depth
                (2 for PRAGMA-S, 6 for PRAGMA-M, 18 for PRAGMA-L — Table 1).
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        super().__init__()
        self.d_model: int = config.d_model  # exposed for callers and tests

        # Shared RoPEEncoding — no trainable parameters; inv_freq is a buffer.
        # Uses te (event temporal coordinates), not ta (profile temporal coordinates).
        self.rope = RoPEEncoding(config)

        # Stack of encoder layers — depth from config (Table 1)
        self.layers = nn.ModuleList([
            _HistoryEncoderLayer(config, self.rope)
            for _ in range(config.history_encoder_layers)
        ])

        # Final layer norm applied to the full output sequence
        self.norm = nn.LayerNorm(config.d_model)

        # Input dropout applied to z before the encoder stack
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        z: torch.Tensor,                            # (batch, 1+ne, d_model) — [USR:EVT] assembled by caller (Eq 6)  # noqa: E501
        te: torch.Tensor,                           # (batch, 1+ne) — temporal coordinates (log-seconds, Eq 2)  # noqa: E501
        event_valid: Optional[torch.Tensor] = None, # (batch, ne) bool — True=real event, False=padding  # noqa: E501
    ) -> torch.Tensor:                              # (batch, 1+ne, d_model) — zh, full encoder output (Eq 7)  # noqa: E501
        """Encode the concatenated [USR:EVT] history sequence with temporal RoPE.

        Args:
            z:  Concatenated input sequence. Shape: (batch, 1+ne, d_model).
                Assembled by caller as z = [za : ze] per Equation 6.
                z[:,0:1,:] = za — [USR] token from ProfileStateEncoder.
                z[:,1:,:]  = ze — [EVT] tokens from EventEncoder.
            te: Temporal coordinates in log-seconds. Shape: (batch, 1+ne).
                te[:,0]  = 0.0 — [USR] position (no timestamp per §2.3.4).
                te[:,1:] = log-seconds to most recent event (Equation 2).
            event_valid: Event-level validity mask. Shape: (batch, ne) bool. Optional.
                         True = real event, False = padding. When provided, padding
                         event positions are masked out from attention (set to -inf before
                         softmax) so their Q/K/V bias contributions cannot leak into
                         real event representations (DEF-005b).
                         The [USR] position (index 0) is always treated as real.

        Returns:
            zh: Full history encoder output. Shape: (batch, 1+ne, d_model).
                zh[:,0,:]  = [USR] representation (user-level).
                zh[:,1:,:] = [EVT] representations (per-event level).
                Caller slices as needed — this encoder returns the full zh.
        """
        # Build attention key-padding mask from event_valid when provided.
        # [USR] at position 0 is always real — prepend a True column.
        # key_valid: (batch, 1+ne) → (batch, 1, 1, 1+ne) for SDPA broadcast.
        # True = real key position (attend), False = padding key (-inf logit).
        attn_mask: Optional[torch.Tensor] = None
        if event_valid is not None:
            batch = z.shape[0]
            usr_valid = torch.ones(batch, 1, dtype=torch.bool, device=z.device)
            key_valid = torch.cat([usr_valid, event_valid], dim=1)  # (batch, 1+ne)
            attn_mask = key_valid.unsqueeze(1).unsqueeze(2)          # (batch, 1, 1, 1+ne)

        # Apply input dropout to embeddings
        zh = self.dropout(z)

        # Pass through each encoder layer — RoPE applied at every layer
        for layer in self.layers:
            zh = layer(zh, te, attn_mask=attn_mask)

        # Final layer norm over the full sequence
        return self.norm(zh)
