"""TokenizerPipeline — orchestrates tokenisation across all field types.

Implements the unified tokenisation pipeline from PRAGMA paper Section 2.2.

The TokenizerPipeline manages a collection of per-field tokenisers and
produces the structured token representation for each event:

    - key_ids      : one key token per value token (replicated for multi-token fields)
    - value_ids    : value token IDs (one or more per field)
    - position_ids : within-field positions — resets to 0 at each new field (Eq 1)
    - temporal_coord: t' = 8·ln(1+t/8) — continuous float for RoPE (Eq 2)
    - calendar_features: [hour, day_of_week, day_of_month] — 3 floats (§2.2)

Position indexing rule (Equation 1, §2.2):
    Positions index values WITHIN a field, not across fields.
    'Currency: eur'         → positions [0]
    'Description: metal plan' → positions [0, 1]
    The NEXT field restarts at position 0.

Key replication rule (Equation 1, §2.2):
    The key token is replicated to match each of its value tokens.
    'Description: metal plan' → 2 key tokens + 2 value tokens.

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from .base import BaseTokenizer
from .temporal import TemporalTokenizer
from .vocabulary import VocabularySpec


class TokenizerOutput(NamedTuple):
    """Structured output from TokenizerPipeline.encode_event().

    All lists are parallel — key_ids[i], value_ids[i], position_ids[i]
    describe the same token.

    Attributes:
        key_ids:           Key token IDs — one per value token. (§2.2, Eq 1)
        value_ids:         Value token IDs — one or more per field.
        position_ids:      Within-field position indices — resets to 0 per field.
                           (§2.2, Eq 1: positions index values within a field)
        temporal_coord:    t' = 8·ln(1+t/8) — continuous float for RoPE. (Eq 2)
        calendar_features: [hour, day_of_week, day_of_month] — exactly 3 floats.
                           (§2.2, key-numbers.md: calendar_feature_dims = 3)
    """

    key_ids: List[int]           # one key token per value token
    value_ids: List[int]         # value token IDs
    position_ids: List[int]      # within-field positions (reset per field)
    temporal_coord: float        # t' = 8·ln(1+t/8), log-seconds (Eq 2)
    calendar_features: List[float]  # [hour, dow, dom] — exactly 3 (§2.2)


class TokenizerPipeline:
    """Orchestrates tokenisation of structured (key, value, time) events.

    Maintains a registry of field-name → BaseTokenizer mappings, manages
    global vocabulary offsets, and produces structured TokenizerOutput.

    Args:
        field_tokenizers: Dict mapping field name → BaseTokenizer instance.
        temporal_tokenizer: Optional TemporalTokenizer for temporal encoding.
                            If None, a default TemporalTokenizer is created.
    """

    # Special token IDs (reserved at start of global vocabulary)
    # Layout: PAD=0, MASK=1, USR=2, EVT=3, UNK=4
    # USR (ID=2) — profile sentinel, placed at xa[:, 0] by PragmaDataset.
    # EVT (ID=3) — event sentinel, placed at xe[:, :, 0] by PragmaDataset.
    # UNK (ID=4) — unknown / out-of-vocabulary token.
    PAD_ID = 0
    MASK_ID = 1
    USR_ID = 2   # [USR] profile sentinel — formerly CLS_ID
    EVT_ID = 3   # [EVT] event sentinel
    UNK_ID = 4   # [UNK] unknown token

    N_SPECIAL_TOKENS = 5

    def __init__(
        self,
        field_tokenizers: Dict[str, BaseTokenizer],
        temporal_tokenizer: Optional[TemporalTokenizer] = None,
    ) -> None:
        self.field_tokenizers = field_tokenizers
        self.temporal_tokenizer = temporal_tokenizer or TemporalTokenizer()

        # Build vocabulary offsets for each field tokeniser
        self._offsets: Dict[str, int] = {}
        self._key_token_ids: Dict[str, int] = {}
        self._build_vocabulary_layout()

    def _build_vocabulary_layout(self) -> None:
        """Assign non-overlapping vocabulary ranges to each field tokeniser.

        Layout:
            [0, N_SPECIAL_TOKENS)              — special tokens (PAD, MASK, USR, EVT, UNK)
            [N_SPECIAL_TOKENS, ...)            — key tokens (one per field name)
            [N_SPECIAL_TOKENS + n_keys, ...)   — value tokens per field
        """
        offset = self.N_SPECIAL_TOKENS

        # Key tokens — one per field name (sorted for determinism)
        for i, field_name in enumerate(sorted(self.field_tokenizers.keys())):
            self._key_token_ids[field_name] = offset + i
        offset += len(self.field_tokenizers)

        # Value token ranges — one contiguous segment per field tokeniser
        for field_name in sorted(self.field_tokenizers.keys()):
            self._offsets[field_name] = offset
            offset += self.field_tokenizers[field_name].vocab_size

    def encode_event(
        self,
        fields: List[Tuple[str, Any, Any]],
        t_seconds: float = 0.0,
    ) -> TokenizerOutput:
        """Encode a single event as a structured TokenizerOutput.

        Applies position indexing (Equation 1, §2.2):
            - Positions index values WITHIN a field, not across fields.
            - Position counter resets to 0 at each new field.
            - Key token is replicated once per value token.

        Args:
            fields: List of (field_name, raw_value, timestamp) tuples.
                    All fields in a single event share the same timestamp.
            t_seconds: Elapsed seconds since most recent event (for temporal
                       coordinate — Equation 2). Default: 0.0.

        Returns:
            TokenizerOutput with parallel key_ids, value_ids, position_ids,
            plus temporal_coord (float) and calendar_features (3 floats).
        """
        key_ids: List[int] = []
        value_ids: List[int] = []
        position_ids: List[int] = []

        for field_name, raw_value, _timestamp in fields:
            if field_name not in self.field_tokenizers:
                continue

            tokenizer = self.field_tokenizers[field_name]
            key_id = self._key_token_ids[field_name]
            field_offset = self._offsets[field_name]

            # Encode the value (may return int or list of ints)
            raw_ids: Union[int, List[int]] = tokenizer.encode(raw_value)
            if isinstance(raw_ids, int):
                raw_ids = [raw_ids]

            n = len(raw_ids)

            # Equation 1 / §2.2: key replicated once per value token;
            # positions index within this field only (reset to 0 each field).
            key_ids.extend([key_id] * n)
            value_ids.extend(v + field_offset for v in raw_ids)
            position_ids.extend(range(n))  # 0, 1, ..., n-1 — within-field

        # Temporal coordinate — Equation 2: t' = 8·ln(1+t/8)
        temporal_coord = self.temporal_tokenizer.compute_temporal_coordinate(t_seconds)

        # Calendar features — §2.2: [hour, day_of_week, day_of_month], 3 floats
        # Use timestamp from first field (all fields in an event share it).
        calendar_features: List[float] = [0.0, 0.0, 0.0]
        for _field_name, _raw_value, timestamp in fields:
            if timestamp is not None:
                calendar_features = self.temporal_tokenizer.extract_calendar_features(
                    timestamp
                )
                break

        return TokenizerOutput(
            key_ids=key_ids,
            value_ids=value_ids,
            position_ids=position_ids,
            temporal_coord=temporal_coord,
            calendar_features=calendar_features,
        )

    def compute_temporal_coordinate(self, t_seconds: float) -> float:
        """Equation 2: t' = 8·ln(1+t/8). Delegates to TemporalTokenizer."""
        return self.temporal_tokenizer.compute_temporal_coordinate(t_seconds)

    def extract_calendar_features(self, timestamp: Any) -> List[float]:
        """§2.2: [hour, day_of_week, day_of_month]. Delegates to TemporalTokenizer."""
        return self.temporal_tokenizer.extract_calendar_features(timestamp)

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary size across all tokenisers and special tokens."""
        return (
            self.N_SPECIAL_TOKENS
            + len(self.field_tokenizers)  # key tokens
            + sum(t.vocab_size for t in self.field_tokenizers.values())
            + self.temporal_tokenizer.vocab_size
        )

    def vocabulary_spec(self) -> VocabularySpec:
        """Return a frozen snapshot of the vocabulary layout.

        The VocabularySpec is the only object that crosses the tokenizer /
        model boundary (ADR 002).  EmbeddingAssembler depends on VocabularySpec
        but NOT on TokenizerPipeline.

        Layout (mirrors _build_vocabulary_layout):
            [0, N_SPECIAL_TOKENS)            — special tokens (PAD, MASK, USR, EVT, UNK)
            [key_start, key_start+key_size)  — key tokens (sorted alphabetically)
            [value_start, value_start+value_size) — value tokens (per field)

        Note: temporal tokenizer vocab is excluded — temporal encoding is
        handled by RoPE (continuous float) and calendar features (3 floats),
        not by discrete token IDs in the embedding table.

        Returns:
            VocabularySpec — frozen dataclass with all vocabulary boundaries.
        """
        key_start  = self.N_SPECIAL_TOKENS
        key_size   = len(self.field_tokenizers)
        value_start = key_start + key_size
        value_size  = sum(t.vocab_size for t in self.field_tokenizers.values())

        # Build field value ranges in sorted field order (matches _build_vocabulary_layout)
        field_value_ranges: Dict[str, Tuple[int, int]] = {}
        offset = value_start
        for field_name in sorted(self.field_tokenizers.keys()):
            vsz = self.field_tokenizers[field_name].vocab_size
            field_value_ranges[field_name] = (offset, offset + vsz)
            offset += vsz

        return VocabularySpec(
            special_tokens={
                "PAD":  self.PAD_ID,
                "MASK": self.MASK_ID,
                "USR":  self.USR_ID,
                "EVT":  self.EVT_ID,
                "UNK":  self.UNK_ID,
            },
            key_start=key_start,
            key_size=key_size,
            value_start=value_start,
            value_size=value_size,
            total_embedding_vocab_size=value_start + value_size,
            field_key_ids=dict(self._key_token_ids),
            field_value_ranges=field_value_ranges,
        )
