"""AssembledBatch — typed output container for the EmbeddingAssembler.

Implements the AssembledBatch contract from ADR 002
(docs/decisions/002-embedding-assembler.md).

AssembledBatch holds all tensors needed for one PRAGMA.forward() call plus
the MLM targets.  The validate() method enforces the ADR 002 key invariant:
targets must contain value-vocab-local IDs (range [0, value_vocab_size)), not
global embedding-table IDs.

Key invariant (ADR 002):
    targets[targets != IGNORE_INDEX].max() < config.value_vocab_size
    targets[targets != IGNORE_INDEX].min() >= 0

    The silent bug this prevents: if a caller passes global_token_ids as MLM
    targets, IDs such as key_offset+local_id might still fall within
    [0, value_vocab_size=28000) and silently corrupt training without any
    out-of-range exception.  validate() makes this violation explicit.

ID naming convention (ADR 002):
    global_token_id  — embedding table index [0, total_embedding_vocab_size)
    value_vocab_id   — MLM target            [0, value_vocab_size)
    IGNORE_INDEX     — sentinel (-100)        positions NOT in MLM loss

Tensor shapes:
    xa:       (batch, na, d_model)     — profile state embeddings
    ta:       (batch, na)              — profile temporal coords (log-seconds)
    xe:       (batch, ne, ni, d_model) — event token embeddings ([EVT] at pos 0)
    xt:       (batch, ne, 3)           — calendar features: [hour, dow, dom]
    te:       (batch, 1+ne)            — history temporal coords (0.0 for [USR])
    mlm_mask: (batch, ne, ni) bool     — True at positions in MLM loss
    targets:  (batch, ne, ni) long     — value_vocab_id or IGNORE_INDEX (-100)

Reference: Ostroukhov et al. (2026), Section 2.3; ADR 002
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from pragma_encoder.model.config import PRAGMAConfig


@dataclass
class AssembledBatch:
    """Typed output of EmbeddingAssembler — inputs for PRAGMA.forward() + MLM targets.

    All tensor fields are parallel and share the same batch dimension.
    validate() must be called before passing to PRAGMA.forward() to catch
    the global-ID-as-target bug early (ADR 002).

    Attributes:
        xa:       Profile state token embeddings. Shape: (batch, na, d_model).
        ta:       Profile temporal coordinates (log-seconds). Shape: (batch, na).
                  Passed through to ProfileStateEncoder for RoPE.
                  0.0 for static profile attributes; log-seconds for life-long events.
        xe:       Event token embeddings. Shape: (batch, ne, ni, d_model).
                  [EVT] token is at position 0 of each event.
        xt:       Calendar features (integer). Shape: (batch, ne, 3).
                  Values: [hour_of_day, day_of_week, day_of_month] (§2.2).
        te:       History temporal coordinates. Shape: (batch, 1+ne).
                  te[:,0] = 0.0 for [USR]; te[:,1:] = log-seconds per event.
        mlm_mask: Token-level MLM mask. Shape: (batch, ne, ni). Bool.
                  True = position is in MLM loss.
        targets:  MLM target IDs. Shape: (batch, ne, ni). Long.
                  value_vocab_id in [0, value_vocab_size) at masked positions,
                  IGNORE_INDEX (-100) everywhere else.
        IGNORE_INDEX: Sentinel value for non-MLM positions (-100).
                  Matches PyTorch nn.CrossEntropyLoss(ignore_index=-100).
    """

    xa:       torch.Tensor  # (batch, na, d_model)
    ta:       torch.Tensor  # (batch, na)              — profile temporal coords
    xe:       torch.Tensor  # (batch, ne, ni, d_model)
    xt:       torch.Tensor  # (batch, ne, 3)
    te:       torch.Tensor  # (batch, 1+ne)
    mlm_mask: torch.Tensor  # (batch, ne, ni) bool
    targets:  torch.Tensor  # (batch, ne, ni) long; value_vocab_id or IGNORE_INDEX

    IGNORE_INDEX: ClassVar[int] = -100  # ClassVar: not in __init__, not per-instance

    def validate(self, config: PRAGMAConfig) -> None:
        """Assert that all tensors conform to the PRAGMA contract (ADR 002).

        Raises:
            AssertionError: if any invariant is violated.

        Checked invariants:
            - xa is 3-dimensional: (batch, na, d_model)
            - xe is 4-dimensional: (batch, ne, ni, d_model)
            - xt is 3-dimensional and xt.shape[-1] == 3  (§2.2: 3 calendar features)
            - te is 2-dimensional: (batch, 1+ne)
            - mlm_mask.dtype == torch.bool
            - targets.dtype == torch.long
            - ADR 002 key invariant: all non-ignore targets in [0, value_vocab_size)
        """
        assert self.xa.dim() == 3, (
            f"xa must be 3-dimensional (batch, na, d_model), got shape {tuple(self.xa.shape)}"
        )
        assert self.ta.dim() == 2, (
            f"ta must be 2-dimensional (batch, na), got shape {tuple(self.ta.shape)}"
        )
        assert self.xe.dim() == 4, (
            f"xe must be 4-dimensional (batch, ne, ni, d_model), got shape {tuple(self.xe.shape)}"
        )
        assert self.xt.dim() == 3 and self.xt.shape[-1] == 3, (
            f"xt must be 3-dimensional with shape[-1]==3 (§2.2: [hour, dow, dom]), "
            f"got shape {tuple(self.xt.shape)}"
        )
        assert self.te.dim() == 2, (
            f"te must be 2-dimensional (batch, 1+ne), got shape {tuple(self.te.shape)}"
        )
        assert self.mlm_mask.dtype == torch.bool, (
            f"mlm_mask must be bool dtype, got {self.mlm_mask.dtype}"
        )
        assert self.targets.dtype == torch.long, (
            f"targets must be long (int64) dtype, got {self.targets.dtype}"
        )

        # Shape consistency — all fields must agree on batch, na, ne, ni, d_model
        batch, na, d = self.xa.shape
        _, ne, ni, d2 = self.xe.shape

        assert d == config.d_model, (
            f"xa d_model mismatch: {d} != {config.d_model}"
        )
        assert d2 == config.d_model, (
            f"xe d_model mismatch: {d2} != {config.d_model}"
        )
        assert self.ta.shape == (batch, na), (
            f"ta shape {tuple(self.ta.shape)} != ({batch}, {na})"
        )
        assert self.xt.shape == (batch, ne, 3), (
            f"xt shape {tuple(self.xt.shape)} != ({batch}, {ne}, 3)"
        )
        assert self.te.shape == (batch, 1 + ne), (
            f"te shape {tuple(self.te.shape)} != ({batch}, {1 + ne})"
        )
        assert self.mlm_mask.shape == (batch, ne, ni), (
            f"mlm_mask shape {tuple(self.mlm_mask.shape)} != ({batch}, {ne}, {ni})"
        )
        assert self.targets.shape == (batch, ne, ni), (
            f"targets shape {tuple(self.targets.shape)} != ({batch}, {ne}, {ni})"
        )

        # ADR 002 key invariant — the guard against the global-ID-as-target bug.
        # Only inspect positions that are NOT ignored.
        valid_mask = self.targets != self.IGNORE_INDEX
        if valid_mask.any():
            valid_targets = self.targets[valid_mask]
            assert valid_targets.min() >= 0, (
                f"ADR 002: targets must be >= 0 at non-ignore positions. "
                f"Found min = {valid_targets.min().item()}. "
                f"Only IGNORE_INDEX ({self.IGNORE_INDEX}) is permitted as a negative target."
            )
            assert valid_targets.max() < config.value_vocab_size, (
                f"ADR 002: targets must be value-vocab-local IDs in [0, value_vocab_size={config.value_vocab_size}). "  # noqa: E501
                f"Found max = {valid_targets.max().item()}. "
                f"Pass value_vocab_id (global_token_id - value_start), not global_token_id."
            )
