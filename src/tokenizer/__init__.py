"""Tokeniser package for the PRAGMA key-value-time tokenisation scheme.

Implements the tokenisation scheme described in PRAGMA paper Section 2.2.

PRAGMA represents each financial event as a structured sequence of
(key, value, time) tuples. Unlike text serialisation approaches used
by tabular-GPT-style models, PRAGMA tokenises each field type with a
dedicated strategy that preserves the statistical properties of the
field.

Exported components:
    BaseTokenizer           — Abstract base for all tokenisers
    NumericalTokenizer      — Percentile-bucket tokeniser for continuous values
    CategoricalTokenizer    — Single-token tokeniser for categorical fields
    TextualTokenizer        — BPE subword tokeniser for free-text fields
    TemporalTokenizer       — Log-seconds + calendar tokeniser for timestamps
    TokenizerPipeline       — Orchestrates tokenisation across all field types
    FinancialTokenizerPipeline — Financial-domain-specific pipeline

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from .base import BaseTokenizer
from .numerical import NumericalTokenizer
from .categorical import CategoricalTokenizer
from .textual import TextualTokenizer
from .temporal import TemporalTokenizer
from .pipeline import TokenizerPipeline
from .financial_pipeline import FinancialTokenizerPipeline

__all__ = [
    "BaseTokenizer",
    "NumericalTokenizer",
    "CategoricalTokenizer",
    "TextualTokenizer",
    "TemporalTokenizer",
    "TokenizerPipeline",
    "FinancialTokenizerPipeline",
]
