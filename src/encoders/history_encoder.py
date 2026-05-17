"""History Encoder.

Implements the History Encoder described in PRAGMA paper Section 2.3.4.

The History Encoder is a bidirectional Transformer that processes the
sequence of per-event [EVT] vectors produced by the Event Encoder.
It models the customer's transaction history as a sequence and produces
contextualised event representations.

Architecture:
    Input:  Sequence of [EVT] vectors (one per transaction) from the
            Event Encoder, conditioned on the [USR] vector from the
            Profile State Encoder via cross-attention.
    Output: Contextualised representations for every position in the
            history sequence, plus a [HIST] summary token.

Key design choices from the paper:
    - Bidirectional — the full history window is visible (MLM training).
    - RoPE positional encoding on the event sequence positions.
    - Cross-attention from history positions to the [USR] profile vector.
      This allows the model to modulate history interpretation based on
      static customer attributes (e.g. plan tier, region).
    - The MASK token replaces event representations during MLM training.

The History Encoder produces the final contextualised event embeddings
that are used for downstream task probing (Section 3.1.1) and fine-tuning
(Section 3.1.2).

Reference: Ostroukhov et al. (2026), Section 2.3.4
"""

import torch
import torch.nn as nn

from .rope import RotaryPositionalEmbedding


class HistoryEncoderLayer(nn.Module):
    """Single History Encoder layer with self-attention + cross-attention.

    Implements pre-norm transformer layer with:
        1. Bidirectional self-attention over the event sequence (with RoPE).
        2. Cross-attention to the [USR] profile representation.
        3. Feed-forward network.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward intermediate dimension.
        dropout: Dropout probability.
        rope: Shared RoPE instance.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        rope: RotaryPositionalEmbedding,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope = rope

        # Self-attention (bidirectional)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention to [USR] profile vector
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Pre-norm layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        usr_repr: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for a single history encoder layer.

        Args:
            x: (batch, seq_len, d_model) — event sequence.
            usr_repr: (batch, d_model) — [USR] profile vector.
            key_padding_mask: (batch, seq_len) — True for padding positions.

        Returns:
            (batch, seq_len, d_model) — updated event sequence.
        """
        # Self-attention with pre-norm (bidirectional — no causal mask)
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.drop(x) + residual

        # Cross-attention to [USR] profile representation
        residual = x
        x = self.norm2(x)
        usr_kv = usr_repr.unsqueeze(1)  # (B, 1, D) — single key/value
        x, _ = self.cross_attn(x, usr_kv, usr_kv)
        x = self.drop(x) + residual

        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.drop(self.ff(x)) + residual

        return x


class HistoryEncoder(nn.Module):
    """Bidirectional Transformer encoder for the full transaction history.

    Processes a sequence of per-event [EVT] vectors from the Event Encoder,
    conditioned on the [USR] vector from the Profile State Encoder.

    Args:
        d_model: Hidden dimension (must match Event Encoder). Default: 256.
        n_heads: Number of attention heads. Default: 8.
        n_layers: Number of History Encoder layers. Default: 6.
        d_ff: Feed-forward intermediate dimension. Default: 1024.
        dropout: Dropout probability. Default: 0.1.
        max_history_len: Maximum events in a history sequence. Default: 512.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_history_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model

        # RoPE for event sequence positions
        self.rope = RotaryPositionalEmbedding(
            dim=d_model // n_heads, max_seq_len=max_history_len
        )

        # Stack of HistoryEncoderLayers
        self.layers = nn.ModuleList([
            HistoryEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                rope=self.rope,
            )
            for _ in range(n_layers)
        ])

        # [HIST] summary token — prepended to the event sequence
        self.hist_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.hist_token, std=0.02)

        # Mask token for MLM — replaces masked event representations
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        event_reprs: torch.Tensor,
        usr_repr: torch.Tensor,
        event_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the full transaction history.

        Args:
            event_reprs: (batch, n_events, d_model) — [EVT] vectors from
                         the Event Encoder. May contain masked positions.
            usr_repr: (batch, d_model) — [USR] vector from the Profile
                      State Encoder.
            event_mask: (batch, n_events) — True for positions to replace
                        with the [MASK] embedding (MLM training).
            key_padding_mask: (batch, n_events) — True for padding positions.

        Returns:
            Tuple of:
                history_reprs: (batch, n_events, d_model) — contextualised
                               event representations at all positions.
                hist_repr: (batch, d_model) — the [HIST] summary token output.
        """
        batch_size, n_events, _ = event_reprs.shape

        # Replace masked events with the [MASK] embedding
        if event_mask is not None:
            mask_emb = self.mask_embedding.expand(batch_size, n_events, -1)
            event_reprs = torch.where(
                event_mask.unsqueeze(-1), mask_emb, event_reprs
            )

        # Prepend [HIST] token
        hist = self.hist_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([hist, event_reprs], dim=1)           # (B, N+1, D)

        # Extend padding mask for [HIST] token
        if key_padding_mask is not None:
            hist_pad = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=key_padding_mask.device
            )
            key_padding_mask = torch.cat([hist_pad, key_padding_mask], dim=1)

        # Apply History Encoder layers
        for layer in self.layers:
            x = layer(x, usr_repr, key_padding_mask=key_padding_mask)

        x = self.layer_norm(x)

        hist_repr = x[:, 0, :]       # (B, D) — [HIST] summary
        history_reprs = x[:, 1:, :]  # (B, N, D) — contextualised events

        return history_reprs, hist_repr
