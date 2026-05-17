"""Event Encoder.

Implements the Event Encoder described in PRAGMA paper Section 2.3.3.

The Event Encoder is a bidirectional Transformer that processes the
token sequence for a single financial event (transaction). Each event
is represented as a flat sequence of (key, value, time) tokens produced
by the TokenizerPipeline.

Architecture:
    Input:  Token sequence for one event: [EVT] + field tokens + time tokens.
    Output: The [EVT] token representation — a dense vector summarising
            the event. These per-event vectors are assembled into a sequence
            and passed to the History Encoder.

Key design choices from the paper:
    - Bidirectional — the full event is visible during encoding.
    - Calendar token embeddings for temporal fields (not RoPE).
      Calendar embeddings capture periodic patterns (hour-of-day,
      day-of-week, etc.) that RoPE does not explicitly model.
    - Separate embedding tables for key tokens and value tokens.
    - [EVT] token plays the same role as [CLS] in BERT.

The Event Encoder is applied independently to each event in the
customer's history before the History Encoder processes the sequence.

Reference: Ostroukhov et al. (2026), Section 2.3.3
"""

import torch
import torch.nn as nn


class CalendarEmbedding(nn.Module):
    """Dedicated embedding table for calendar tokens.

    Encodes periodic temporal features (hour, day-of-week, day-of-month,
    month, quarter) with learned embeddings rather than position encodings.
    This allows the model to learn that hour=12 (noon) and hour=13 (1pm)
    are similar without imposing a strict linear ordering.

    Args:
        n_hours: Number of hour-of-day values. Default: 24.
        n_dow: Number of day-of-week values. Default: 7.
        n_dom: Number of day-of-month values. Default: 31.
        n_months: Number of month values. Default: 12.
        n_quarters: Number of quarter values. Default: 4.
        d_model: Embedding dimension.
    """

    def __init__(
        self,
        d_model: int,
        n_hours: int = 24,
        n_dow: int = 7,
        n_dom: int = 31,
        n_months: int = 12,
        n_quarters: int = 4,
    ):
        super().__init__()
        self.hour_emb = nn.Embedding(n_hours, d_model)
        self.dow_emb = nn.Embedding(n_dow, d_model)
        self.dom_emb = nn.Embedding(n_dom, d_model)
        self.month_emb = nn.Embedding(n_months, d_model)
        self.quarter_emb = nn.Embedding(n_quarters, d_model)

        self.projection = nn.Linear(5 * d_model, d_model)

    def forward(self, calendar_tokens: torch.Tensor) -> torch.Tensor:
        """Embed calendar tokens and project to d_model.

        Args:
            calendar_tokens: (batch, 5) — [hour, dow, dom, month, quarter].

        Returns:
            Calendar embedding of shape (batch, d_model).
        """
        h = self.hour_emb(calendar_tokens[:, 0])
        d = self.dow_emb(calendar_tokens[:, 1])
        dm = self.dom_emb(calendar_tokens[:, 2])
        m = self.month_emb(calendar_tokens[:, 3])
        q = self.quarter_emb(calendar_tokens[:, 4])
        combined = torch.cat([h, d, dm, m, q], dim=-1)
        return self.projection(combined)


class EventEncoder(nn.Module):
    """Bidirectional Transformer encoder for a single financial event.

    Processes the flat token sequence of one transaction and produces
    an [EVT] summary vector.

    Args:
        vocab_size: Total token vocabulary size (from TokenizerPipeline).
        d_model: Hidden dimension. Default: 256.
        n_heads: Number of attention heads. Default: 8.
        n_layers: Number of Transformer encoder layers. Default: 4.
        d_ff: Feed-forward intermediate dimension. Default: 1024.
        dropout: Dropout probability. Default: 0.1.
        max_seq_len: Maximum tokens per event. Default: 128.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding (key tokens + value tokens share this table)
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)

        # Calendar embedding for temporal periodic features
        self.calendar_embedding = CalendarEmbedding(d_model=d_model)

        # Bidirectional Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # [EVT] token — prepended to the event token sequence
        self.evt_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.evt_token, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        event_token_ids: torch.Tensor,
        calendar_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a single event and return the [EVT] representation.

        Args:
            event_token_ids: (batch, seq_len) token IDs for the event.
            calendar_tokens: (batch, 5) calendar feature tokens
                             [hour, dow, dom, month, quarter].
                             Added to the [EVT] token embedding.
            attention_mask: (batch, seq_len) padding mask.

        Returns:
            evt_repr: (batch, d_model) — the [EVT] token output.
        """
        batch_size = event_token_ids.shape[0]

        # Embed event tokens
        x = self.dropout(self.embedding(event_token_ids))  # (B, S, D)

        # Prepend [EVT] token, optionally enriched with calendar embedding
        evt = self.evt_token.expand(batch_size, -1, -1)  # (B, 1, D)
        if calendar_tokens is not None:
            cal_emb = self.calendar_embedding(calendar_tokens)  # (B, D)
            evt = evt + cal_emb.unsqueeze(1)

        x = torch.cat([evt, x], dim=1)  # (B, S+1, D)

        # Extend attention mask for [EVT] token
        if attention_mask is not None:
            evt_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=attention_mask.device
            )
            attention_mask = torch.cat([evt_mask, attention_mask], dim=1)

        # Bidirectional self-attention (no causal mask)
        x = self.encoder(x, src_key_padding_mask=attention_mask)
        x = self.layer_norm(x)

        # Return the [EVT] token output (position 0)
        evt_repr = x[:, 0, :]  # (B, D)
        return evt_repr
