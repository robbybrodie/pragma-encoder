"""Masking package — three-strategy MLM masking objective.

Implements the three masking strategies described in PRAGMA paper
Section 2.3.5. PRAGMA extends the standard BERT masked language
modelling objective with domain-specific masking strategies that
reflect the structure of financial event sequences.

Three masking strategies:
    1. Token masking:   Mask individual tokens within events (standard MLM).
    2. Field masking:   Mask all tokens belonging to a specific field type
                        across the entire sequence (e.g. mask all amounts).
    3. Event masking:   Mask all tokens belonging to entire events
                        (forces the model to impute missing transactions).

The three strategies are applied with tunable probabilities during
pretraining. The paper uses a mixture to ensure the model learns at
multiple granularities: token-level semantics, field-level statistics,
and event-level temporal patterns.

Reference: Ostroukhov et al. (2026), Section 2.3.5
"""

from .strategy import MaskingStrategy, TokenMasker, FieldMasker, EventMasker

__all__ = [
    "MaskingStrategy",
    "TokenMasker",
    "FieldMasker",
    "EventMasker",
]
