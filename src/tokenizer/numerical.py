"""Numerical field tokeniser — percentile-bucket encoding.

Implements the numerical tokenisation strategy from PRAGMA paper Section 2.2.

Continuous numerical fields (e.g. transaction amount, account balance) are
discretised into percentile buckets computed from the training corpus. This
preserves the ordinal structure of the field while producing a fixed-size
discrete vocabulary, and is robust to outliers and heavy-tailed distributions
common in financial data.

Approach:
    1. At fit() time, compute N equally-spaced percentile boundaries from
       the training distribution.
    2. At encode() time, assign a raw value to a bucket ID via binary search.
    3. Special tokens handle missing values and out-of-range inputs.

The number of buckets (n_buckets) is a hyperparameter. The PRAGMA paper
uses field-specific bucket counts tuned to the vocabulary budget.

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from typing import Any, List, Union

import numpy as np

from .base import BaseTokenizer


class NumericalTokenizer(BaseTokenizer):
    """Percentile-bucket tokeniser for continuous numerical fields.

    Maps continuous values to discrete bucket IDs by computing
    percentile boundaries from the training distribution.

    Args:
        n_buckets: Number of percentile buckets. Default: 100.
        handle_missing: How to handle NaN/None values. One of:
            'special_token' (default) — assign a dedicated [MISSING] token.
            'zero' — map to bucket 0.
    """

    def __init__(self, n_buckets: int = 100, handle_missing: str = "special_token"):
        self.n_buckets = n_buckets
        self.handle_missing = handle_missing
        self._boundaries: np.ndarray = np.array([])
        self._fitted = False

    def fit(self, data: List[Any]) -> "NumericalTokenizer":
        """Compute percentile boundaries from training data.

        Args:
            data: List of numerical values. NaN/None values are excluded
                  from boundary computation.

        Returns:
            Self.
        """
        values = np.array([v for v in data if v is not None and not np.isnan(float(v))], dtype=float)
        percentiles = np.linspace(0, 100, self.n_buckets + 1)
        self._boundaries = np.percentile(values, percentiles)
        self._fitted = True
        return self

    def encode(self, value: Any) -> int:
        """Assign value to a percentile bucket.

        Returns:
            Bucket ID in [0, n_buckets - 1], or n_buckets for missing values
            when handle_missing='special_token'.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if self.handle_missing == "special_token":
                return self.n_buckets  # [MISSING] token
            return 0
        bucket = int(np.searchsorted(self._boundaries[1:-1], float(value)))
        return min(bucket, self.n_buckets - 1)

    def decode(self, token_ids: Union[int, List[int]]) -> float:
        """Return the bucket midpoint for interpretability.

        Args:
            token_ids: A single bucket ID.

        Returns:
            Midpoint of the corresponding percentile bucket.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before decode().")
        bucket_id = token_ids if isinstance(token_ids, int) else token_ids[0]
        if bucket_id == self.n_buckets:
            return float("nan")  # missing token
        lo = self._boundaries[bucket_id]
        hi = self._boundaries[min(bucket_id + 1, len(self._boundaries) - 1)]
        return float((lo + hi) / 2)

    @property
    def vocab_size(self) -> int:
        """n_buckets regular tokens + 1 [MISSING] token."""
        return self.n_buckets + 1
