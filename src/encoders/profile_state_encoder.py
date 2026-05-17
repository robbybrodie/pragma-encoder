"""Profile State Encoder.

Implements the Profile State Encoder described in PRAGMA paper Section 2.3.2.

The Profile State Encoder is a bidirectional Transformer that processes
static customer attributes (plan, region, account tenure, balance quantile)
alongside life-long event timestamps encoded via RoPE.

Architecture:
    Input:  Tokenised static customer profile fields + optional summary
            timestamps of the customer's life-long event history.
    Output: The [USR] token representation, a dense vector summarising
            the customer's static state. This is passed to the History
            Encoder as a conditioning signal.

Key design choices from the paper:
    - Bidirectional (not causal) — the profile is fully observed.
    - RoPE positional encoding on timestamp positions.
    - The [USR] token plays the same role as [CLS] in BERT.
    - Separate from the Event and History encoders — no shared weights.

The Profile State Encoder output serves as the initial hidden state
injected into the History Encoder's cross-attention mechanism.

Reference: Ostroukhov et al. (2026), Section 2.3.2
"""

import torch
import torch.nn as nn

from .rope import RotaryPositionalEmbedding


class ProfileStateEncoder(nn.Module):
    """Bidirectional Transformer encoder for static customer profile state.

    Processes tokenised profile fields and produces a [USR] representation
    summarising the customer's static attributes.

    Args:
        vocab_size: Size of the profile field vocabulary.
        d_model: Hidden dimension of the Transformer. Default: 256.
        n_heads: Number of attention heads. Default: 8.
        n_layers: Number of Transformer encoder layers. Default: 4.
        d_ff: Feed-forward intermediate dimension. Default: 1024.
        dropout: Dropout probability. Default: 0.1.
        max_seq_len: Maximum profile sequence length. Default: 512.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding table for profile fields
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)

        # RoPE for timestamp positions in the profile sequence
        self.rope = RotaryPositionalEmbedding(
            dim=d_model // n_heads, max_seq_len=max_seq_len
        )

        # Bidirectional Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable for deep models)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # [USR] token — prepended to the profile sequence
        self.usr_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.usr_token, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        profile_token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a customer profile and return the [USR] representation.

        Args:
            profile_token_ids: (batch, seq_len) token IDs from the
                               profile field tokeniser.
            attention_mask: (batch, seq_len) boolean mask where True
                            indicates padding positions to ignore.

        Returns:
            usr_repr: (batch, d_model) — the [USR] token output,
                      summarising the customer's static profile state.
        """
        batch_size = profile_token_ids.shape[0]

        # Embed profile tokens
        x = self.dropout(self.embedding(profile_token_ids))  # (B, S, D)

        # Prepend [USR] token
        usr = self.usr_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([usr, x], dim=1)  # (B, S+1, D)

        # Extend attention mask for the prepended [USR] token
        if attention_mask is not None:
            usr_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=attention_mask.device
            )
            attention_mask = torch.cat([usr_mask, attention_mask], dim=1)

        # Bidirectional self-attention (no causal mask)
        x = self.encoder(x, src_key_padding_mask=attention_mask)
        x = self.layer_norm(x)

        # Return the [USR] token output (position 0)
        usr_repr = x[:, 0, :]  # (B, D)
        return usr_repr
