"""Categorical field tokeniser — single-token encoding.

Implements the categorical tokenisation strategy from PRAGMA paper Section 2.2.

Categorical fields (e.g. merchant category code, transaction type, currency,
country code) are mapped to a single token per value. The vocabulary is built
from the training corpus. An [UNK] token handles unseen values at inference.

PRAGMA uses a unified vocabulary across all field types. Each tokeniser
manages its own segment of the global vocabulary; offsets are applied by
the TokenizerPipeline.

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from typing import Any, Dict, List, Union

from .base import BaseTokenizer


class CategoricalTokenizer(BaseTokenizer):
    """Single-token tokeniser for categorical fields.

    Builds a vocabulary from training data, mapping each unique
    category to an integer ID. Unseen values at inference receive
    the [UNK] token.

    Args:
        min_freq: Minimum occurrence count for a category to receive
                  its own token. Categories below this threshold are
                  mapped to [UNK]. Default: 1.
        max_vocab: Maximum vocabulary size (excluding special tokens).
                   Rare categories beyond this limit map to [UNK].
                   Default: 10_000.
    """

    UNK = "<UNK>"
    MISSING = "<MISSING>"

    def __init__(self, min_freq: int = 1, max_vocab: int = 10_000):
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self._token2id: Dict[str, int] = {}
        self._id2token: Dict[int, str] = {}
        self._fitted = False

    def fit(self, data: List[Any]) -> "CategoricalTokenizer":
        """Build vocabulary from training data.

        Args:
            data: List of categorical values (will be cast to str).

        Returns:
            Self.
        """
        from collections import Counter

        counts = Counter(str(v) for v in data if v is not None)

        # Reserve 0 for [UNK], 1 for [MISSING]
        self._token2id = {self.UNK: 0, self.MISSING: 1}
        self._id2token = {0: self.UNK, 1: self.MISSING}

        next_id = 2
        for token, freq in counts.most_common(self.max_vocab):
            if freq < self.min_freq:
                break
            self._token2id[token] = next_id
            self._id2token[next_id] = token
            next_id += 1

        self._fitted = True
        return self

    def encode(self, value: Any) -> int:
        """Map a categorical value to its token ID.

        Args:
            value: Raw categorical value.

        Returns:
            Token ID, or UNK ID for unseen values.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")
        if value is None:
            return self._token2id[self.MISSING]
        return self._token2id.get(str(value), self._token2id[self.UNK])

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        """Return the category string for a token ID.

        Args:
            token_ids: A single token ID.

        Returns:
            Category string.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before decode().")
        token_id = token_ids if isinstance(token_ids, int) else token_ids[0]
        return self._id2token.get(token_id, self.UNK)

    @property
    def vocab_size(self) -> int:
        """Size of the built vocabulary including special tokens."""
        return len(self._token2id)
