"""Three-strategy masking for PRAGMA pretraining.

Implements the three masking strategies from PRAGMA paper Section 2.3.5.

The PRAGMA pretraining objective is masked event modelling (MEM), which
extends masked language modelling (MLM) with domain-specific masking
granularities appropriate for structured financial sequences.

Strategies:
    TokenMasker:  Standard BERT-style masking of individual tokens.
                  Applies mask_prob to each token independently.
                  80% → [MASK], 10% → random token, 10% → unchanged.

    FieldMasker:  Masks all tokens belonging to a chosen field type
                  across the entire sequence. For example, mask all
                  'amount_local' tokens. Forces the model to learn
                  inter-field relationships (e.g. infer amount from
                  merchant category and channel).

    EventMasker:  Masks all tokens belonging to entire event positions.
                  Forces the model to impute missing transactions from
                  surrounding context. This is the most challenging
                  strategy and is critical for the History Encoder.

Reference: Ostroukhov et al. (2026), Section 2.3.5
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch


MASK_TOKEN_ID = 1  # Must match TokenizerPipeline.MASK_ID


class MaskingStrategy(ABC):
    """Abstract base class for PRAGMA masking strategies."""

    @abstractmethod
    def apply(
        self,
        token_ids: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to a token sequence.

        Args:
            token_ids: (batch, seq_len) input token IDs.
            **kwargs:  Strategy-specific arguments.

        Returns:
            Tuple of:
                masked_ids: (batch, seq_len) — token IDs with masks applied.
                labels:     (batch, seq_len) — original token IDs at masked
                            positions, -100 elsewhere (ignored in loss).
        """
        ...


class TokenMasker(MaskingStrategy):
    """Standard BERT-style token masking (Section 2.3.5, Strategy 1).

    Randomly selects mask_prob fraction of non-special tokens and applies:
        80% → [MASK] token
        10% → random token from vocabulary
        10% → unchanged (but still predicted)

    Args:
        mask_prob: Probability of masking each token. Default: 0.15.
        vocab_size: Total vocabulary size (for random token replacement).
        mask_token_id: ID of the [MASK] token. Default: 1.
        special_token_ids: Set of special token IDs to never mask. Default: {0}.
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        vocab_size: int = 50_000,
        mask_token_id: int = MASK_TOKEN_ID,
        special_token_ids: Optional[set] = None,
    ):
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.special_token_ids = special_token_ids or {0}

    def apply(
        self,
        token_ids: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply token-level masking.

        Args:
            token_ids: (batch, seq_len)

        Returns:
            (masked_ids, labels)
        """
        masked_ids = token_ids.clone()
        labels = torch.full_like(token_ids, -100)

        # Select tokens to predict
        probability_matrix = torch.full(token_ids.shape, self.mask_prob)
        for special_id in self.special_token_ids:
            probability_matrix[token_ids == special_id] = 0.0

        selected = torch.bernoulli(probability_matrix).bool()
        labels[selected] = token_ids[selected]

        # 80% → [MASK]
        mask_indices = selected & (torch.rand_like(probability_matrix) < 0.8)
        masked_ids[mask_indices] = self.mask_token_id

        # 10% → random token
        random_indices = selected & ~mask_indices & (torch.rand_like(probability_matrix) < 0.5)
        random_tokens = torch.randint(len(self.special_token_ids), self.vocab_size, token_ids.shape)
        masked_ids[random_indices] = random_tokens[random_indices]

        # Remaining 10% → unchanged (labels still set, so they are predicted)
        return masked_ids, labels


class FieldMasker(MaskingStrategy):
    """Field-level masking — mask all tokens of a chosen field type (Strategy 2).

    Selects one or more field types at random and masks all tokens belonging
    to that field across the entire sequence. Requires field position metadata
    from the TokenizerPipeline.

    Args:
        mask_token_id: ID of the [MASK] token. Default: 1.
        n_fields_to_mask: Number of field types to mask per sample. Default: 1.
    """

    def __init__(
        self,
        mask_token_id: int = MASK_TOKEN_ID,
        n_fields_to_mask: int = 1,
    ):
        self.mask_token_id = mask_token_id
        self.n_fields_to_mask = n_fields_to_mask

    def apply(
        self,
        token_ids: torch.Tensor,
        field_positions: Optional[Dict[str, List[List[int]]]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply field-level masking.

        Args:
            token_ids: (batch, seq_len)
            field_positions: Dict mapping field_name → list of (per-sample)
                             lists of token positions for that field.
                             If None, falls back to TokenMasker behaviour.

        Returns:
            (masked_ids, labels)
        """
        masked_ids = token_ids.clone()
        labels = torch.full_like(token_ids, -100)

        if field_positions is None:
            return masked_ids, labels

        field_names = list(field_positions.keys())
        chosen = random.sample(field_names, min(self.n_fields_to_mask, len(field_names)))

        for field_name in chosen:
            positions_per_sample = field_positions[field_name]
            for batch_idx, positions in enumerate(positions_per_sample):
                for pos in positions:
                    labels[batch_idx, pos] = token_ids[batch_idx, pos]
                    masked_ids[batch_idx, pos] = self.mask_token_id

        return masked_ids, labels


class EventMasker(MaskingStrategy):
    """Event-level masking — mask all tokens of entire events (Strategy 3).

    Selects mask_prob fraction of events and masks all their tokens.
    Operates on the event representation level in the History Encoder
    (replacing [EVT] vectors with the [MASK] embedding).

    Args:
        mask_prob: Probability of masking each event. Default: 0.15.
        mask_token_id: ID of the [MASK] token. Default: 1.
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = MASK_TOKEN_ID,
    ):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

    def apply(
        self,
        token_ids: torch.Tensor,
        event_boundaries: Optional[List[List[Tuple[int, int]]]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply event-level masking.

        Args:
            token_ids: (batch, seq_len)
            event_boundaries: Per-sample list of (start, end) token ranges
                              for each event. If None, no masking is applied.

        Returns:
            (masked_ids, labels)
        """
        masked_ids = token_ids.clone()
        labels = torch.full_like(token_ids, -100)

        if event_boundaries is None:
            return masked_ids, labels

        for batch_idx, boundaries in enumerate(event_boundaries):
            for start, end in boundaries:
                if random.random() < self.mask_prob:
                    for pos in range(start, end):
                        labels[batch_idx, pos] = token_ids[batch_idx, pos]
                        masked_ids[batch_idx, pos] = self.mask_token_id

        return masked_ids, labels
