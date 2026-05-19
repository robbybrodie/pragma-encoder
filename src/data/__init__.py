"""PRAGMA data pipeline — real transaction data loading and tokenisation.

Provides:
    TabFormerAdapter  — maps IBM TabFormer CSV columns to pipeline field names
    PragmaDataset     — torch.utils.data.Dataset over tokenised customer sequences
"""

from .tabformer_adapter import TabFormerAdapter
from .pragma_dataset import PragmaDataset

__all__ = ["TabFormerAdapter", "PragmaDataset"]
