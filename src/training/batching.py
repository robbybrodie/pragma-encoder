"""Dynamic batching by sequence length.

Provides a batch sampler that groups sequences of similar length into
batches to minimise within-batch padding, improving training throughput.

Dynamic batching motivation (from Section 2.4):
    PRAGMA training processes variable-length customer histories. Static
    batching (fixed batch size B) wastes GPU memory for short histories
    and runs out of memory for long histories. Dynamic batching groups
    similar-length sequences together and adjusts batch size to keep
    total tokens per batch approximately constant.

This implementation provides a token-budget-based batch sampler
compatible with PyTorch DataLoader.

Reference: Ostroukhov et al. (2026), Section 2.4
"""

import random
from typing import Iterator, List

from torch.utils.data import Sampler


class DynamicBatchSampler(Sampler):
    """Groups sequences by length and batches by token budget.

    Sorts the dataset by sequence length, then groups sequences
    into batches where the total number of tokens ≤ max_tokens_per_batch.
    Batches are shuffled at epoch boundaries.

    Args:
        lengths: List of sequence lengths for each dataset sample.
        max_tokens_per_batch: Maximum total tokens per batch. Default: 16_384.
        shuffle: Whether to shuffle batches between epochs. Default: True.
        bucket_size_multiplier: Group sequences into length buckets of this
                                 multiple for approximate sorting. Default: 100.
    """

    def __init__(
        self,
        lengths: List[int],
        max_tokens_per_batch: int = 16_384,
        shuffle: bool = True,
        bucket_size_multiplier: int = 100,
    ):
        self.lengths = lengths
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.bucket_size_multiplier = bucket_size_multiplier

        self._batches: List[List[int]] = self._build_batches()

    def _build_batches(self) -> List[List[int]]:
        """Pre-compute batches sorted by length."""
        # Sort indices by sequence length
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len = 0

        for idx in indices:
            seq_len = self.lengths[idx]
            new_max = max(current_max_len, seq_len)

            if current_batch and new_max * (len(current_batch) + 1) > self.max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = []
                current_max_len = 0

            current_batch.append(idx)
            current_max_len = max(current_max_len, seq_len)

        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = list(self._batches)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)
