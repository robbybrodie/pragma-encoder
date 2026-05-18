"""VocabularySpec and VocabularyMap — vocabulary layout for the embedding table.

Implements the vocabulary boundary objects from ADR 002
(docs/decisions/002-embedding-assembler.md).

These two classes are the only public surface that crosses the tokenizer /
model boundary.  EmbeddingAssembler imports VocabularySpec and VocabularyMap
but does NOT import TokenizerPipeline.

Vocabulary layout (mirroring TokenizerPipeline._build_vocabulary_layout):
    [0, N_SPECIAL_TOKENS)            — special tokens (PAD=0, MASK=1, CLS=2, SEP=3)
    [key_start, key_start+key_size)  — key tokens (one per field, sorted alphabetically)
    [value_start, value_start+value_size) — value tokens (contiguous per field)

ID naming convention (ADR 002):
    global_token_id  — index into the embedding table; range [0, total_embedding_vocab_size)
    value_vocab_id   — MLM target; range [0, value_vocab_size); equals global_token_id - value_start
    field_value_id   — internal to each field tokenizer; range [0, field.vocab_size)

Note on temporal tokenizer:
    The temporal tokenizer is NOT included in total_embedding_vocab_size.
    Temporal encoding is handled by RoPE (continuous float t' = 8·ln(1+t/8))
    and calendar features ([hour, day_of_week, day_of_month] — 3 floats),
    not by discrete token IDs in the embedding table.

Reference: Ostroukhov et al. (2026), Section 2.2; ADR 002
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class VocabularySpec:
    """Frozen snapshot of the vocabulary layout exported by TokenizerPipeline.

    All fields are plain Python scalars or dicts — no tokenizer internals.
    This is the only object that crosses the tokenizer / model boundary
    (ADR 002: EmbeddingAssembler depends on VocabularySpec, not on
    TokenizerPipeline).

    Attributes:
        special_tokens:            Dict of special token name → global ID.
                                   e.g. {"PAD": 0, "MASK": 1, "CLS": 2, "SEP": 3}
        key_start:                 First global ID in the key token range.
        key_size:                  Number of key tokens (one per registered field).
        value_start:               First global ID in the value token range.
                                   Always equals key_start + key_size.
        value_size:                Total number of value tokens across all fields.
        total_embedding_vocab_size: Full size of the embedding table.
                                   = key_start + key_size + value_size
        field_key_ids:             Dict of field name → global key token ID.
        field_value_ranges:        Dict of field name → (lo, hi) global ID range
                                   (half-open: value IDs in [lo, hi)).
    """

    special_tokens: Dict[str, int]
    key_start: int
    key_size: int
    value_start: int
    value_size: int
    total_embedding_vocab_size: int
    field_key_ids: Dict[str, int]
    field_value_ranges: Dict[str, Tuple[int, int]]


class VocabularyMap:
    """Single source of truth for global ↔ local ID arithmetic.

    Constructed from a VocabularySpec only — no dependency on TokenizerPipeline.
    All methods operate element-wise on torch.Tensor inputs and return tensors
    of the same shape.

    ADR 002 ID arithmetic:
        value_vocab_id = global_token_id - value_start

    Args:
        spec: VocabularySpec from TokenizerPipeline.vocabulary_spec().
    """

    def __init__(self, spec: VocabularySpec) -> None:
        self.spec = spec

    def global_to_local_value_id(self, ids: torch.Tensor) -> torch.Tensor:
        """Convert global value token IDs to value-vocab-local IDs.

        ADR 002: value_vocab_id = global_token_id - value_start

        Callers must ensure ids are in [value_start, value_start + value_size)
        before calling this method.  Out-of-range inputs produce negative or
        oversized local IDs that will fail the MLM loss computation.

        Args:
            ids: Tensor of global token IDs (any shape).

        Returns:
            Tensor of value-vocab-local IDs, same shape as ids.
        """
        return ids - self.spec.value_start

    def is_key_id(self, ids: torch.Tensor) -> torch.Tensor:
        """Return True for IDs in the key token range [key_start, key_start+key_size).

        Args:
            ids: Tensor of global token IDs (any shape, any integer dtype).

        Returns:
            Bool tensor, same shape as ids.
        """
        return (ids >= self.spec.key_start) & (ids < self.spec.key_start + self.spec.key_size)

    def is_special_id(self, ids: torch.Tensor) -> torch.Tensor:
        """Return True for IDs in the special token range [0, key_start).

        Args:
            ids: Tensor of global token IDs (any shape, any integer dtype).

        Returns:
            Bool tensor, same shape as ids.
        """
        return ids < self.spec.key_start

    def is_global_value_id(self, ids: torch.Tensor) -> torch.Tensor:
        """Return True for IDs in the value token range [value_start, value_start+value_size).

        Args:
            ids: Tensor of global token IDs (any shape, any integer dtype).

        Returns:
            Bool tensor, same shape as ids.
        """
        return (ids >= self.spec.value_start) & (ids < self.spec.value_start + self.spec.value_size)
