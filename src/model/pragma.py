"""Full PRAGMA model — three-encoder assembly.

Implements the complete PRAGMA foundation model described in Section 2.3
of the PRAGMA paper, composing the three encoder branches:

    1. Profile State Encoder (Section 2.3.2) — static customer attributes.
    2. Event Encoder        (Section 2.3.3) — per-transaction encoding.
    3. History Encoder      (Section 2.3.4) — sequential history modelling.

Forward pass:
    Given a batch of (profile, events) inputs:
    1. Profile State Encoder → [USR] representation per customer.
    2. Event Encoder applied to each event independently → [EVT] vectors.
    3. History Encoder conditions on [USR] and processes [EVT] sequence
       → contextualised history representations + [HIST] summary.

During pretraining, the MLMHead is applied to the History Encoder
output to predict masked event tokens.

During fine-tuning (Section 3.1.2), the History Encoder output is
passed to a LoRA-adapted classification head.

Reference: Ostroukhov et al. (2026), Section 2.3
"""

import torch
import torch.nn as nn

from ..encoders import EventEncoder, HistoryEncoder, ProfileStateEncoder
from .config import PRAGMAConfig
from .mlm_head import MLMHead


class PRAGMA(nn.Module):
    """Full PRAGMA foundation model.

    Composes ProfileStateEncoder, EventEncoder, and HistoryEncoder
    into the complete three-encoder architecture from Section 2.3.

    Args:
        config: PRAGMAConfig instance specifying architecture hyperparameters.
    """

    def __init__(self, config: PRAGMAConfig):
        super().__init__()
        self.config = config

        # Profile State Encoder — static customer profile (Section 2.3.2)
        self.profile_encoder = ProfileStateEncoder(
            vocab_size=config.profile_vocab_size,
            d_model=config.d_model,
            n_heads=config.profile_n_heads,
            n_layers=config.profile_n_layers,
            d_ff=config.profile_d_ff,
            dropout=config.dropout,
            max_seq_len=config.profile_max_seq_len,
        )

        # Event Encoder — per-transaction encoding (Section 2.3.3)
        self.event_encoder = EventEncoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.event_n_heads,
            n_layers=config.event_n_layers,
            d_ff=config.event_d_ff,
            dropout=config.dropout,
            max_seq_len=config.event_max_seq_len,
        )

        # History Encoder — sequential history modelling (Section 2.3.4)
        self.history_encoder = HistoryEncoder(
            d_model=config.d_model,
            n_heads=config.history_n_heads,
            n_layers=config.history_n_layers,
            d_ff=config.history_d_ff,
            dropout=config.dropout,
            max_history_len=config.history_max_seq_len,
        )

        # MLM head for pretraining (Section 2.3.5)
        self.mlm_head = MLMHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
        )

    def forward(
        self,
        profile_token_ids: torch.Tensor,
        event_token_ids: torch.Tensor,
        calendar_tokens: torch.Tensor | None = None,
        profile_attention_mask: torch.Tensor | None = None,
        event_attention_mask: torch.Tensor | None = None,
        history_key_padding_mask: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full PRAGMA forward pass.

        Args:
            profile_token_ids: (batch, profile_seq_len) — profile tokens.
            event_token_ids: (batch, n_events, event_seq_len) — event tokens.
            calendar_tokens: (batch, n_events, 5) — calendar tokens per event.
            profile_attention_mask: (batch, profile_seq_len) — profile padding.
            event_attention_mask: (batch, n_events, event_seq_len) — event padding.
            history_key_padding_mask: (batch, n_events) — history padding.
            event_mask: (batch, n_events) — True for masked events (MLM).

        Returns:
            Dict with keys:
                'history_reprs':  (batch, n_events, d_model)
                'hist_repr':      (batch, d_model)
                'usr_repr':       (batch, d_model)
                'mlm_logits':     (batch, n_events, vocab_size) — if event_mask given
        """
        batch_size, n_events, event_seq_len = event_token_ids.shape

        # Step 1: Profile State Encoder → [USR]
        usr_repr = self.profile_encoder(
            profile_token_ids,
            attention_mask=profile_attention_mask,
        )  # (B, D)

        # Step 2: Event Encoder → [EVT] for each event independently
        # Flatten events into batch dimension for parallel processing
        event_ids_flat = event_token_ids.view(batch_size * n_events, event_seq_len)

        cal_flat = None
        if calendar_tokens is not None:
            cal_flat = calendar_tokens.view(batch_size * n_events, 5)

        attn_flat = None
        if event_attention_mask is not None:
            attn_flat = event_attention_mask.view(batch_size * n_events, event_seq_len)

        evt_reprs_flat = self.event_encoder(
            event_ids_flat, calendar_tokens=cal_flat, attention_mask=attn_flat
        )  # (B * N, D)

        evt_reprs = evt_reprs_flat.view(batch_size, n_events, -1)  # (B, N, D)

        # Step 3: History Encoder → contextualised history
        history_reprs, hist_repr = self.history_encoder(
            event_reprs=evt_reprs,
            usr_repr=usr_repr,
            event_mask=event_mask,
            key_padding_mask=history_key_padding_mask,
        )  # (B, N, D), (B, D)

        output = {
            "history_reprs": history_reprs,
            "hist_repr": hist_repr,
            "usr_repr": usr_repr,
        }

        # MLM logits — only computed when event_mask is provided (pretraining)
        if event_mask is not None:
            # Apply MLM head to masked event positions only
            masked_reprs = history_reprs[event_mask]  # (n_masked, D)
            if masked_reprs.numel() > 0:
                output["mlm_logits"] = self.mlm_head(masked_reprs)

        return output

    def get_event_embeddings(
        self,
        profile_token_ids: torch.Tensor,
        event_token_ids: torch.Tensor,
        calendar_tokens: torch.Tensor | None = None,
        history_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract contextualised event embeddings for downstream use.

        Convenience method for embedding extraction at inference time.
        No masking is applied.

        Args:
            profile_token_ids: (batch, profile_seq_len)
            event_token_ids: (batch, n_events, event_seq_len)
            calendar_tokens: (batch, n_events, 5)
            history_key_padding_mask: (batch, n_events)

        Returns:
            (batch, n_events, d_model) — contextualised event embeddings.
        """
        with torch.no_grad():
            output = self.forward(
                profile_token_ids=profile_token_ids,
                event_token_ids=event_token_ids,
                calendar_tokens=calendar_tokens,
                history_key_padding_mask=history_key_padding_mask,
            )
        return output["history_reprs"]
