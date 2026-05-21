"""PRAGMA data pipeline — real transaction data loading and tokenisation.

Provides:
    TabFormerAdapter  — maps IBM TabFormer CSV columns to pipeline field names
    PragmaDataset     — torch.utils.data.Dataset over tokenised customer sequences
"""

from .pragma_dataset import PragmaDataset
from .tabformer_adapter import TabFormerAdapter

__all__ = ["TabFormerAdapter", "PragmaDataset"]
