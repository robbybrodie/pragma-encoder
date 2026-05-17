"""Model package — full PRAGMA model assembly.

Composes the three encoders and MLM head into the complete PRAGMA
foundation model as described in PRAGMA paper Section 2.3.

    PRAGMAConfig — dataclass of architecture hyperparameters (Section 2.3.1)
    PRAGMA       — full model: Profile State + Event + History encoders
    MLMHead      — masked event modelling prediction head (Section 2.3.5)

Reference: Ostroukhov et al. (2026), Section 2.3
"""

from .config import PRAGMAConfig
from .mlm_head import MLMHead
from .pragma import PRAGMA

__all__ = [
    "PRAGMAConfig",
    "PRAGMA",
    "MLMHead",
]
