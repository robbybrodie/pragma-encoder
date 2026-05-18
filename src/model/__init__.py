"""Model package — full PRAGMA model assembly.

Composes the three encoders and MLM head into the complete PRAGMA
foundation model as described in PRAGMA paper Section 2.3.

    PRAGMAConfig       — dataclass of architecture hyperparameters (Section 2.3.1)
    PRAGMA             — full model: Profile State + Event + History encoders
    MLMHead            — masked event modelling prediction head (Section 2.3.5)
    AssembledBatch     — typed batch container for EmbeddingAssembler output (ADR 002)
    EmbeddingAssembler — converts global token IDs to embedded tensors (Section 2.3.1)

Reference: Ostroukhov et al. (2026), Section 2.3
"""

from .config import PRAGMAConfig

__all__ = [
    "PRAGMAConfig",
    "PRAGMA",
    "MLMHead",
    "AssembledBatch",
    "EmbeddingAssembler",
]


def __getattr__(name: str) -> object:
    """Lazy-import torch-dependent symbols to keep PRAGMAConfig importable
    without torch installed (e.g. during config-only tests).
    """
    if name == "MLMHead":
        from .mlm_head import MLMHead
        return MLMHead
    if name == "PRAGMA":
        from .pragma import PRAGMA
        return PRAGMA
    if name == "AssembledBatch":
        from .assembled_batch import AssembledBatch
        return AssembledBatch
    if name == "EmbeddingAssembler":
        from .assembler import EmbeddingAssembler
        return EmbeddingAssembler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
