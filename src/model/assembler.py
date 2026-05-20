"""EmbeddingAssembler — converts global token IDs to embedded float tensors.

Implements Equation 1 from PRAGMA paper Section 2.3.1:

    x_ij = PosEmb_j( E(k_i) + E(v_ij) )

where:
    k_i      — global key token ID for field i
    v_ij     — global value token ID at position j within field i
               (may be MASK_ID=1 if MaskingStrategy has already corrupted it)
    E        — shared nn.Embedding table (both key and value IDs look up from E)
    PosEmb_j — sinusoidal positional embedding for within-field position j
               NOT learnable: Vaswani et al. (2017) fixed sine/cosine encoding

Responsibility boundary (ADR 002):
    MaskingStrategy:    CHOOSES which positions to mask; replaces selected
                        value token IDs with MASK_ID=1 or UNK_ID=0.
    EmbeddingAssembler: APPLIES Equation 1 to whatever token IDs it receives;
                        CONVERTS global value token IDs to value-vocab-local
                        target IDs for the MLM loss.

Caller flow (pre-training):
    1. TokenizerPipeline  → symbolic token IDs (key_ids, val_ids, pos_ids)
    2. MaskingStrategy    → masked_val_ids, target_ids, mask
    3. EmbeddingAssembler → AssembledBatch (xa, ta, xe, xt, te, mlm_mask, targets)
    4. PRAGMA.forward()   ← receives xa, ta, xe, xt, te, mask, token_ids

Dependency constraints (ADR 002, DEVELOPMENT_PROCESS.md):
    EmbeddingAssembler does NOT import TokenizerPipeline.
    EmbeddingAssembler does NOT import PRAGMA or any encoder.
    EmbeddingAssembler does NOT import MaskingStrategy.
    Imports allowed: VocabularySpec, VocabularyMap (from tokenizer.vocabulary),
                     PRAGMAConfig, AssembledBatch (both from model).

Technical debt:
    TD-003 (docs/tech-debt.md): ProfileTokenizerPipeline not yet implemented.
    Callers must supply xa_key_ids, xa_val_ids, xa_pos_ids manually
    until ProfileTokenizerPipeline is built.

Reference: Ostroukhov et al. (2026), Section 2.3.1, Equation 1; ADR 002
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.model.assembled_batch import AssembledBatch
from src.model.config import PRAGMAConfig
from src.tokenizer.vocabulary import VocabularyMap, VocabularySpec


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding for within-field positions.

    Implements PosEmb_j from Equation 1 (§2.3.1).
    Uses the standard sine/cosine formulation from Vaswani et al. (2017).

    NOT a learnable component — positions are stored as a fixed buffer.
    sum(p.numel() for p in self.parameters()) == 0.

    Args:
        max_len:  Maximum within-field position index (exclusive).
                  Set to config.max_event_tokens to cover all event tokens.
        d_model:  Embedding dimension. Must match the shared embedding table E.
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        # Build fixed sinusoidal table: shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        # register_buffer: part of model state (saved/loaded with state_dict)
        # but NOT a trainable parameter.
        self.register_buffer("pe", pe)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Return sinusoidal embeddings for the given position IDs.

        Args:
            position_ids: (...) long — within-field positions in [0, max_len).

        Returns:
            (..., d_model) — sinusoidal embeddings, same leading dims as position_ids.
        """
        return self.pe[position_ids]  # type: ignore[index]


class EmbeddingAssembler(nn.Module):
    """Converts global token IDs to embedded float tensors via Equation 1.

    Implements the bridge between TokenizerPipeline (symbolic token IDs) and
    PRAGMA.forward() (embedded float tensors). This is the only component that
    performs global ↔ local ID arithmetic; all other components see only
    embedded float tensors or value-vocab-local IDs (ADR 002).

    Submodules:
        E:       nn.Embedding(spec.total_embedding_vocab_size, config.d_model)
                 Shared lookup table for both key and value token IDs (Eq 1).
        pos_emb: SinusoidalPositionEmbedding(config.max_event_tokens, config.d_model)
                 Fixed within-field positional encoding (not a Parameter).
        vocab:   VocabularyMap(spec) — single source of truth for ID arithmetic.

    Args:
        spec:   VocabularySpec from TokenizerPipeline.vocabulary_spec().
                Defines vocabulary boundaries — the only tokenizer surface that
                crosses into the model layer (ADR 002).
        config: PRAGMAConfig — provides d_model and max_event_tokens.
    """

    def __init__(self, spec: VocabularySpec, config: PRAGMAConfig) -> None:
        super().__init__()
        self.spec   = spec
        self.config = config

        # Shared embedding table — both key and value IDs look up from E (Eq 1)
        self.E = nn.Embedding(spec.total_embedding_vocab_size, config.d_model)

        # Fixed sinusoidal positional encoding — NOT a trainable parameter
        # Use the larger of max_event_tokens and max_profile_tokens so that
        # both event (max 24) and profile (max 200) position IDs are safely
        # within the sinusoidal table.
        # key-numbers.md: max_event_tokens=24, max_profile_tokens=200, §2.4
        _max_positions = max(config.max_event_tokens, config.max_profile_tokens)
        self.pos_emb = SinusoidalPositionEmbedding(_max_positions, config.d_model)

        # ID arithmetic — converts global IDs to value-vocab-local IDs for MLM targets
        self.vocab = VocabularyMap(spec)

    def _embed(
        self,
        key_ids: torch.Tensor,  # (...) long — global key token IDs
        val_ids: torch.Tensor,  # (...) long — global value token IDs
        pos_ids: torch.Tensor,  # (...) long — within-field positions [0, max_event_tokens)
    ) -> torch.Tensor:          # (..., d_model)
        """Equation 1: x = E(k) + E(v) + PosEmb(j).

        Works for any leading shape — both (batch, na) and (batch, ne, ni).
        """
        return self.E(key_ids) + self.E(val_ids) + self.pos_emb(pos_ids)  # type: ignore[no-any-return]

    def forward(
        self,
        xa_key_ids: torch.Tensor,            # (batch, na)       long — profile key IDs
        xa_val_ids: torch.Tensor,            # (batch, na)       long — profile value IDs
        xa_pos_ids: torch.Tensor,            # (batch, na)       long — profile positions
        ta:         torch.Tensor,            # (batch, na)       float — profile temporal
        xe_key_ids: torch.Tensor,            # (batch, ne, ni)   long — event key IDs
        xe_val_ids: torch.Tensor,            # (batch, ne, ni)   long — event value IDs
        xe_pos_ids: torch.Tensor,            # (batch, ne, ni)   long — event positions
        xt:         torch.Tensor,            # (batch, ne, 3)    long — calendar features
        te:         torch.Tensor,            # (batch, 1+ne)     float — history temporal
        target_ids: Optional[torch.Tensor] = None,  # (batch, ne, ni) long — original global IDs
        mask:       Optional[torch.Tensor] = None,  # (batch, ne, ni) bool — [MASK] positions
    ) -> AssembledBatch:
        """Embed token IDs via Equation 1 and build an AssembledBatch.

        Profile path (xa): x_a_j = E(xa_key_ids_j) + E(xa_val_ids_j) + PosEmb(xa_pos_ids_j)
        Event path   (xe): x_e_ij = E(xe_key_ids_ij) + E(xe_val_ids_ij) + PosEmb(xe_pos_ids_ij)

        Both paths use the shared embedding table E. Changing xa inputs does not
        affect xe and vice versa — they are independent lookups.

        When mask is provided (pre-training mode):
            - xe_val_ids must already be corrupted by MaskingStrategy
              (MASK_ID=1 at masked positions, original IDs elsewhere)
            - target_ids must be the ORIGINAL global value IDs (pre-corruption)
            - targets are localised: local_id = global_id - value_start
            - Positions where mask=False receive IGNORE_INDEX=-100 in targets

        When mask is None (inference / embedding-extraction mode):
            - No masking is applied or assumed
            - All targets are set to IGNORE_INDEX=-100
            - mlm_mask is all-False

        Args:
            xa_key_ids: Global key token IDs for profile tokens. Shape: (batch, na).
            xa_val_ids: Global value token IDs for profile tokens. Shape: (batch, na).
            xa_pos_ids: Within-field positions for profile tokens. Shape: (batch, na).
            ta:         Profile temporal coordinates (log-seconds). Shape: (batch, na).
                        Passed through unchanged to AssembledBatch for ProfileStateEncoder.
            xe_key_ids: Global key token IDs for event tokens. Shape: (batch, ne, ni).
            xe_val_ids: Global value token IDs for event tokens. Shape: (batch, ne, ni).
                        Must be pre-corrupted by MaskingStrategy if mask is provided.
            xe_pos_ids: Within-field positions for event tokens. Shape: (batch, ne, ni).
            xt:         Calendar features [hour, dow, dom]. Shape: (batch, ne, 3).
                        Passed through unchanged for EventEncoder.
            te:         History temporal coordinates (log-seconds). Shape: (batch, 1+ne).
                        Passed through unchanged for HistoryEncoder.
            target_ids: Original global value IDs (pre-corruption). Shape: (batch, ne, ni).
                        Required when mask is provided. Ignored when mask is None.
            mask:       True at [MASK] positions (included in MLM loss). Shape: (batch, ne, ni).
                        When None, operates in inference mode.

        Returns:
            AssembledBatch ready for PRAGMA.forward(). validate() is called
            before returning to enforce the ADR 002 key invariant.
        """
        # ------------------------------------------------------------------
        # Equation 1: embed both paths through shared table E
        # ------------------------------------------------------------------
        xa = self._embed(xa_key_ids, xa_val_ids, xa_pos_ids)  # (batch, na, d_model)
        xe = self._embed(xe_key_ids, xe_val_ids, xe_pos_ids)  # (batch, ne, ni, d_model)

        # ------------------------------------------------------------------
        # Build MLM targets (ADR 002 invariant: local_id in [0, value_vocab_size))
        # ------------------------------------------------------------------
        batch_size, ne, ni = xe_val_ids.shape

        if mask is not None and target_ids is not None:
            # Training mode: localise global IDs at [MASK] positions
            targets = torch.full(
                (batch_size, ne, ni), AssembledBatch.IGNORE_INDEX,
                dtype=torch.long, device=xe_val_ids.device,
            )
            if mask.any():
                targets[mask] = self.vocab.global_to_local_value_id(target_ids[mask])
            mlm_mask = mask
        else:
            # Inference mode: no masking, all targets are IGNORE_INDEX
            targets = torch.full(
                (batch_size, ne, ni), AssembledBatch.IGNORE_INDEX,
                dtype=torch.long, device=xe_val_ids.device,
            )
            mlm_mask = torch.zeros(
                batch_size, ne, ni, dtype=torch.bool, device=xe_val_ids.device,
            )

        # ------------------------------------------------------------------
        # Assemble and validate
        # ------------------------------------------------------------------
        out = AssembledBatch(
            xa=xa, ta=ta, xe=xe, xt=xt, te=te,
            mlm_mask=mlm_mask, targets=targets,
        )
        out.validate(self.config)
        return out
