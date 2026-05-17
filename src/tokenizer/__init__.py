"""Tokeniser package for the PRAGMA key-value-time tokenisation scheme.

Implements the tokenisation scheme described in PRAGMA paper Section 2.2.

PRAGMA represents each financial event as a structured sequence of
(key, value, time) tuples. Unlike text serialisation approaches used
by tabular-GPT-style models, PRAGMA tokenises each field type with a
dedicated strategy that preserves the statistical properties of the
field.

Key design decisions (§2.2):
    - Position indices are WITHIN a field, not across fields (Equation 1)
    - Key token is replicated once per value token (Equation 1)
    - Temporal coordinate is a continuous float: 8·ln(1+t/8) (Equation 2)
    - Calendar features are exactly 3: [hour, day_of_week, day_of_month]
    - key_vocab_size ≈ 60,  value_vocab_size ≈ 28,000

Interface contract defined in PRAGMATokenizerProtocol below.
Every component that consumes tokeniser output can type-check against it.

Exported components:
    BaseTokenizer               — Abstract base for all tokenisers
    NumericalTokenizer          — Percentile-bucket tokeniser for continuous values
    CategoricalTokenizer        — Single-token tokeniser for categorical fields
    TextualTokenizer            — BPE subword tokeniser for free-text fields
    TemporalTokenizer           — Log-seconds + calendar feature extractor
    TokenizerOutput             — NamedTuple: structured encode_event() output
    TokenizerPipeline           — Orchestrates tokenisation across all field types
    FinancialTokenizerPipeline  — Financial-domain-specific pipeline
    PRAGMATokenizerProtocol     — Interface contract (typing.Protocol)

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from typing import Any, List, Protocol, Tuple

from .base import BaseTokenizer
from .categorical import CategoricalTokenizer
from .financial_pipeline import FinancialTokenizerPipeline
from .numerical import NumericalTokenizer
from .pipeline import TokenizerOutput, TokenizerPipeline
from .temporal import TemporalTokenizer
from .textual import TextualTokenizer


class PRAGMATokenizerProtocol(Protocol):
    """Interface contract for the PRAGMA tokeniser (§2.2).

    Any class implementing this protocol can be used wherever a
    PRAGMATokenizer is expected, and will satisfy the encoder interfaces
    for profile state, event, and history encoding.

    Method shapes follow DEVELOPMENT_PROCESS.md naming conventions.
    """

    def encode_event(
        self,
        fields: List[Tuple[str, Any, Any]],  # [(field_name, value, timestamp)]
        t_seconds: float,                     # elapsed seconds since most recent event
    ) -> TokenizerOutput:
        """Encode one event to structured TokenizerOutput (§2.2, Eq 1)."""
        ...

    def compute_temporal_coordinate(
        self,
        t_seconds: float,  # elapsed seconds since most recent event
    ) -> float:            # t' = 8·ln(1+t/8) — Equation 2
        """Temporal log transform for RoPE input (§2.2, Equation 2)."""
        ...

    def extract_calendar_features(
        self,
        timestamp: Any,   # datetime or unix timestamp
    ) -> List[float]:     # [hour, day_of_week, day_of_month] — exactly 3 floats
        """Calendar features for EventEncoder MLP (§2.2, calendar_feature_dims=3)."""
        ...


__all__ = [
    "BaseTokenizer",
    "NumericalTokenizer",
    "CategoricalTokenizer",
    "TextualTokenizer",
    "TemporalTokenizer",
    "TokenizerOutput",
    "TokenizerPipeline",
    "FinancialTokenizerPipeline",
    "PRAGMATokenizerProtocol",
]
