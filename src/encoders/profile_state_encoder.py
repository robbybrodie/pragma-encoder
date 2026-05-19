"""Profile State Encoder — bidirectional Transformer with RoPE.

Implements the Profile State Encoder described in PRAGMA paper Section 2.3.2.

The Profile State Encoder processes static customer profile fields
(plan, region, account tenure, balance quantile) alongside optional
life-long event timestamps. It produces a dense sequence of contextualised
profile token embeddings.

Contract with callers:
    INPUT:
        xa: (batch, na, d_model) — profile state token embeddings.
            Already embedded via Equation 1 (E(k) + E(v) + PosEmb).
            The [USR] token is ALREADY prepended at position 0 by the caller.
            These are FLOAT embeddings, not raw integer token IDs.

        ta: (batch, na) — temporal coordinates in log-seconds.
            Computed via Equation 2: t' = 8·ln(1 + t/8).
            Fed into RoPEEncoding for every attention layer.
            Non-temporal profile fields use t' = 0.

    OUTPUT:
        za: (batch, na, d_model) — full encoder output sequence.
            The caller extracts za[:,0:1,:] for the [USR] token.
            This encoder does NOT slice — it returns the full za.

Key design decisions from Section 2.3.2:
    - Bidirectional self-attention (is_causal=False) — NEVER causal
    - RoPE applied to Q and K in EVERY attention layer, using ta
    - Pre-norm LayerNorm (Xiong et al., 2020) — norm_first pattern
    - GELU activation (Hendrycks et al., 2016)
    - Dropout = 0.1 (§2.3)
    - config.profile_encoder_layers layers (1 / 3 / 9 for S / M / L)

Separation of concerns (CLAUDE.md, ADR 003):
    - Embedding (Equation 1) is done OUTSIDE this class by the pipeline
    - [USR] token is prepended OUTSIDE this class by the caller
    - RoPE (temporal) is SEPARATE from within-field PosEmb (Equation 1)
    - This encoder does NOT add the [USR] token itself

Reference: Ostroukhov et al. (2026), Section 2.3.2, Equation 4
RoPE: Su et al. (2024), arXiv:2104.09864
"""

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
    ta passed from ProfileStateEncoder.forward().

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
        x: torch.Tensor,   # (batch, na, d_model)
        ta: torch.Tensor,  # (batch, na) — temporal coordinates (log-seconds)
    ) -> torch.Tensor:     # (batch, na, d_model)
        batch, na, _ = x.shape

        # Linear projections: (batch, na, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, n_heads, na, head_dim) for multi-head attention
        def _to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, na, self.n_heads, self.head_dim).transpose(1, 2)

        q = _to_heads(q)  # (batch, n_heads, na, head_dim)
        k = _to_heads(k)
        v = _to_heads(v)

        # Apply RoPE rotation to Q and K using temporal coordinates.
        # ta: (batch, na) matches positions shape expected by RoPEEncoding.forward().
        # Equation 9: dot(q_rot[m], k_rot[n]) = q^T R(tn - tm) k — relative distance only.
        q, k = self.rope(q, k, ta)

        # Bidirectional scaled dot-product attention — is_causal=False.
        # PRAGMA is encoder-only. NEVER use causal masking here.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self._dropout if self.training else 0.0,
            is_causal=False,  # bidirectional — non-negotiable (CLAUDE.md)
        )  # (batch, n_heads, na, head_dim)

        # Reshape back to (batch, na, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, na, self.d_model)
        return self.out_proj(attn_out)  # type: ignore[no-any-return]


class _ProfileEncoderLayer(nn.Module):
    """One pre-norm Transformer encoder layer with RoPE attention.

    Implements the standard pre-norm (Xiong et al., 2020) pattern:
        xa = xa + dropout(attn(norm1(xa), ta))   # attention sub-layer
        xa = xa + dropout(ff(norm2(xa)))          # feed-forward sub-layer

    Activation: GELU (§2.3, key-numbers.md).
    Normalisation: pre-norm LayerNorm (§2.3, key-numbers.md).

    Args:
        config: PRAGMAConfig — provides all hyperparameters.
        rope:   Shared RoPEEncoding instance passed in from ProfileStateEncoder.
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
        xa: torch.Tensor,  # (batch, na, d_model)
        ta: torch.Tensor,  # (batch, na) — temporal coordinates
    ) -> torch.Tensor:     # (batch, na, d_model)
        # Pre-norm attention + residual (§2.3: pre-norm LayerNorm)
        xa = xa + self.dropout(self.attn(self.norm1(xa), ta))
        # Pre-norm feed-forward + residual (§2.3: GELU activation)
        xa = xa + self.dropout(self.ff(self.norm2(xa)))
        return xa


class ProfileStateEncoder(nn.Module):
    """Bidirectional Transformer encoder for static customer profile state.

    Processes pre-embedded profile field tokens xa (with [USR] already at
    position 0) and temporal coordinates ta via RoPE, producing a full
    contextualised sequence za ∈ R^(na × d).

    The caller extracts za[:,0:1,:] (the [USR] token) for the History Encoder.
    This class returns the complete za sequence — it does NOT slice.

    Args:
        config: PRAGMAConfig — single source of truth for all hyperparameters.
                config.profile_encoder_layers controls the stack depth
                (1 for PRAGMA-S, 3 for PRAGMA-M, 9 for PRAGMA-L — Table 1).
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        super().__init__()
        self.d_model: int = config.d_model  # exposed for callers and tests

        # Shared RoPEEncoding — no trainable parameters; inv_freq is a buffer.
        # Shared across all encoder layers (same frequency progression everywhere).
        self.rope = RoPEEncoding(config)

        # Stack of encoder layers — depth from config (Table 1)
        self.layers = nn.ModuleList([
            _ProfileEncoderLayer(config, self.rope)
            for _ in range(config.profile_encoder_layers)
        ])

        # Final layer norm applied to the full output sequence
        self.norm = nn.LayerNorm(config.d_model)

        # Input dropout applied to the pre-embedded xa before the encoder stack
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        xa: torch.Tensor,  # (batch, na, d_model) — pre-embedded; [USR] at pos 0
        ta: torch.Tensor,  # (batch, na) — temporal coordinates (log-seconds, Eq 2)
    ) -> torch.Tensor:     # (batch, na, d_model) — za, full encoder output
        """Encode profile state token embeddings with temporal RoPE.

        Args:
            xa: Profile state token embeddings. Shape: (batch, na, d_model).
                Float tensor — pre-embedded via Equation 1 by the pipeline.
                The [USR] token is already at position 0 (prepended by caller).
            ta: Temporal coordinates in log-seconds from Equation 2.
                Shape: (batch, na). Non-temporal fields use t'=0.

        Returns:
            za: Full encoder output sequence. Shape: (batch, na, d_model).
                The caller extracts za[:,0:1,:] for the [USR] token.
        """
        # Apply input dropout to embeddings (standard practice, as in BERT)
        za = self.dropout(xa)

        # Pass through each encoder layer — RoPE applied at every layer
        for layer in self.layers:
            za = layer(za, ta)

        # Final layer norm over the full sequence
        return self.norm(za)  # type: ignore[no-any-return]
