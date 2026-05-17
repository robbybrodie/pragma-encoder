"""TokenizerPipeline — orchestrates tokenisation across all field types.

Implements the unified tokenisation pipeline from PRAGMA paper Section 2.2.

The TokenizerPipeline manages a collection of per-field tokenisers and
assembles the final token sequence for each (key, value, time) tuple.

Each event in the PRAGMA representation is a sequence of fields:
    [(field_name, raw_value, timestamp), ...]

The pipeline:
    1. Dispatches each field to its registered tokeniser.
    2. Applies vocabulary offsets so all tokenisers share a global namespace.
    3. Prepends the [field_name] key token before each value token sequence.
    4. Appends the temporal tokens from the TemporalTokenizer.

The resulting flat token sequence is consumed by the encoders.

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseTokenizer
from .temporal import TemporalTokenizer


class TokenizerPipeline:
    """Orchestrates tokenisation of structured (key, value, time) events.

    Maintains a registry of field-name → BaseTokenizer mappings, manages
    global vocabulary offsets, and produces flat token sequences.

    Args:
        field_tokenizers: Dict mapping field name → BaseTokenizer instance.
        temporal_tokenizer: Optional TemporalTokenizer for timestamps.
                            If None, a default TemporalTokenizer is created.
    """

    # Special token IDs (reserved at start of global vocabulary)
    PAD_ID = 0
    MASK_ID = 1
    CLS_ID = 2  # [EVT] in event sequences
    SEP_ID = 3

    N_SPECIAL_TOKENS = 4

    def __init__(
        self,
        field_tokenizers: Dict[str, BaseTokenizer],
        temporal_tokenizer: Optional[TemporalTokenizer] = None,
    ):
        self.field_tokenizers = field_tokenizers
        self.temporal_tokenizer = temporal_tokenizer or TemporalTokenizer()

        # Build vocabulary offsets for each field tokeniser
        self._offsets: Dict[str, int] = {}
        self._key_token_ids: Dict[str, int] = {}
        self._build_vocabulary_layout()

    def _build_vocabulary_layout(self) -> None:
        """Assign non-overlapping vocabulary ranges to each field tokeniser.

        Layout:
            [0, N_SPECIAL_TOKENS)         — special tokens
            [N_SPECIAL_TOKENS, ...)       — key tokens (one per field name)
            [N_SPECIAL_TOKENS + n_keys, ...) — value tokens per field
        """
        offset = self.N_SPECIAL_TOKENS

        # Key tokens — one per field name
        for i, field_name in enumerate(sorted(self.field_tokenizers.keys())):
            self._key_token_ids[field_name] = offset + i
        offset += len(self.field_tokenizers)

        # Value token ranges — one segment per field tokeniser
        for field_name in sorted(self.field_tokenizers.keys()):
            self._offsets[field_name] = offset
            offset += self.field_tokenizers[field_name].vocab_size

        self._temporal_offset = offset

    def encode_event(
        self,
        fields: List[Tuple[str, Any, Any]],
    ) -> List[int]:
        """Encode a single event as a flat token sequence.

        Args:
            fields: List of (field_name, raw_value, timestamp) tuples.
                    All fields in a single event share the same timestamp.

        Returns:
            Flat list of token IDs representing the event.
        """
        tokens: List[int] = [self.CLS_ID]  # [EVT] token

        for field_name, raw_value, timestamp in fields:
            if field_name not in self.field_tokenizers:
                continue

            # Key token
            tokens.append(self._key_token_ids[field_name])

            # Value token(s) with offset applied
            tokenizer = self.field_tokenizers[field_name]
            value_ids = tokenizer.encode(raw_value)
            if isinstance(value_ids, int):
                value_ids = [value_ids]
            tokens.extend(v + self._offsets[field_name] for v in value_ids)

            # Temporal tokens
            time_ids = self.temporal_tokenizer.encode(timestamp)
            tokens.extend(t + self._temporal_offset for t in time_ids)

        return tokens

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary size across all tokenisers and special tokens."""
        return (
            self.N_SPECIAL_TOKENS
            + len(self.field_tokenizers)  # key tokens
            + sum(t.vocab_size for t in self.field_tokenizers.values())
            + self.temporal_tokenizer.vocab_size
        )
