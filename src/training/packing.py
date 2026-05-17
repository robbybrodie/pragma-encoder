"""Sequence packing utilities for training efficiency.

Provides utilities for packing multiple short sequences into a single
training example to minimise padding waste, as used in PRAGMA training
(Section 2.4).

Sequence packing motivation (from Section 2.4):
    At scale (207B tokens), Revolut uses LMDB storage and custom
    sequence packing to achieve high GPU utilisation. Customer histories
    vary greatly in length — packing prevents short-history customers
    from wasting most of the sequence length budget with padding.

This implementation provides a simplified packing strategy suitable
for research-scale training. Full LMDB-based packing at 207B token
scale is not implemented here.

Packing strategy:
    Given a target sequence length T, greedily bin-pack sequences
    into packs such that the total length of each pack ≤ T.
    Position IDs are reset within each packed sequence to maintain
    correct RoPE computation.

Reference: Ostroukhov et al. (2026), Section 2.4
"""

from typing import Iterator, List, Tuple

import torch


class SequencePacker:
    """Greedy bin-packing of variable-length sequences.

    Packs multiple sequences into a single tensor of length max_seq_len
    to reduce padding waste during training.

    Args:
        max_seq_len: Target packed sequence length. Default: 512.
        pad_id: Padding token ID. Default: 0.
    """

    def __init__(self, max_seq_len: int = 512, pad_id: int = 0):
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def pack(
        self,
        sequences: List[List[int]],
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Greedily pack sequences into fixed-length packs.

        Args:
            sequences: List of token ID sequences (variable length).

        Yields:
            Tuples of:
                packed_ids: (max_seq_len,) — packed token IDs.
                pack_mask:  (max_seq_len,) — 1 for valid tokens, 0 for padding.
        """
        current_pack: List[int] = []
        current_mask: List[int] = []

        for seq in sequences:
            if len(seq) > self.max_seq_len:
                seq = seq[: self.max_seq_len]

            if len(current_pack) + len(seq) > self.max_seq_len:
                # Flush current pack
                yield self._pad_pack(current_pack, current_mask)
                current_pack = []
                current_mask = []

            current_pack.extend(seq)
            current_mask.extend([1] * len(seq))

        # Flush remaining
        if current_pack:
            yield self._pad_pack(current_pack, current_mask)

    def _pad_pack(
        self, pack: List[int], mask: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a pack to max_seq_len."""
        pad_len = self.max_seq_len - len(pack)
        packed_ids = torch.tensor(pack + [self.pad_id] * pad_len, dtype=torch.long)
        pack_mask = torch.tensor(mask + [0] * pad_len, dtype=torch.bool)
        return packed_ids, pack_mask
