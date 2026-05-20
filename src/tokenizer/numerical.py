"""Numerical field tokeniser — percentile-bucket encoding.

Implements the numerical tokenisation strategy from PRAGMA paper Section 2.2.

Continuous numerical fields (e.g. transaction amount, account balance) are
discretised into percentile buckets computed from the training corpus. This
preserves the ordinal structure of the field while producing a fixed-size
discrete vocabulary, and is robust to outliers and heavy-tailed distributions
common in financial data.

Approach:
    1. At fit() time, compute N equally-spaced percentile boundaries from
       the non-zero training distribution.
    2. At encode() time:
         - Exactly zero → dedicated zero bucket (bucket ID n_buckets).
         - Non-zero, non-missing → percentile bucket in [0, n_buckets - 1].
         - Missing/NaN → missing token (bucket ID n_buckets + 1).

§2.2: "Extra bucket for zero" — exact zeros are separated from small
non-zero values. This is critical for financial fields where zero is a
semantically distinct value (e.g. zero fee, zero interest).

The number of buckets (n_buckets) is a hyperparameter. The PRAGMA paper
uses field-specific bucket counts tuned to the vocabulary budget.

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from typing import Any, List, Union

import numpy as np

from .base import BaseTokenizer


class NumericalTokenizer(BaseTokenizer):
    """Percentile-bucket tokeniser for continuous numerical fields.

    Token ID layout:
        0  to  n_buckets - 1   : percentile buckets (non-zero values)
        n_buckets               : ZERO bucket — dedicated to exactly 0.0 (§2.2)
        n_buckets + 1           : MISSING token — NaN / None values

    Args:
        n_buckets: Number of percentile buckets for non-zero values. Default: 100.
        handle_missing: How to handle NaN/None values. One of:
            'special_token' (default) — assign a dedicated MISSING token.
            'zero' — map to zero bucket.
    """

    def __init__(self, n_buckets: int = 100, handle_missing: str = "special_token"):
        self.n_buckets = n_buckets
        self.handle_missing = handle_missing
        self._boundaries: np.ndarray = np.array([])
        self._fitted = False

    @property
    def ZERO_ID(self) -> int:  # noqa: N802 — uppercase is intentional (constant-like property)
        """Token ID for exactly-zero values. §2.2: extra bucket for zero."""
        return self.n_buckets

    @property
    def MISSING_ID(self) -> int:  # noqa: N802 — uppercase is intentional (constant-like property)
        """Token ID for NaN / None values."""
        return self.n_buckets + 1

    def fit(self, data: List[Any]) -> "NumericalTokenizer":
        """Compute percentile boundaries from non-zero, non-missing training data.

        Zero values are excluded from the percentile computation because they
        receive a dedicated bucket at encode() time (§2.2: extra bucket for zero).

        Args:
            data: List of numerical values. NaN/None and exactly-zero values are
                  excluded from boundary computation.

        Returns:
            Self.
        """
        values = np.array(
            [
                v
                for v in data
                if v is not None
                and not np.isnan(float(v))
                and float(v) != 0.0
            ],
            dtype=float,
        )
        percentiles = np.linspace(0, 100, self.n_buckets + 1)
        self._boundaries = np.percentile(values, percentiles)
        self._fitted = True
        return self

    def encode(self, value: Any) -> int:
        """Assign value to a bucket.

        Token ID assignment:
            None / NaN        → MISSING_ID  (= n_buckets + 1)
            exactly 0.0       → ZERO_ID     (= n_buckets)     [§2.2: extra bucket]
            non-zero, non-NaN → percentile bucket in [0, n_buckets - 1]

        Args:
            value: Raw numerical value.

        Returns:
            Bucket ID (int).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")

        if value is None or (isinstance(value, float) and np.isnan(value)):
            if self.handle_missing == "special_token":
                return self.MISSING_ID
            return self.ZERO_ID

        fval = float(value)

        # §2.2: extra bucket for zero — exactly zero gets its own token
        if fval == 0.0:
            return self.ZERO_ID

        bucket = int(np.searchsorted(self._boundaries[1:-1], fval))
        return min(bucket, self.n_buckets - 1)

    def decode(self, token_ids: Union[int, List[int]]) -> float:
        """Return the bucket midpoint for interpretability.

        Args:
            token_ids: A single bucket ID.

        Returns:
            Midpoint of the corresponding percentile bucket, or special
            sentinel values for ZERO (0.0) and MISSING (NaN).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before decode().")
        bucket_id = token_ids if isinstance(token_ids, int) else token_ids[0]
        if bucket_id == self.MISSING_ID:
            return float("nan")
        if bucket_id == self.ZERO_ID:
            return 0.0
        lo = self._boundaries[bucket_id]
        hi = self._boundaries[min(bucket_id + 1, len(self._boundaries) - 1)]
        return float((lo + hi) / 2)

    @property
    def vocab_size(self) -> int:
        """Total token count: n_buckets percentile + 1 zero + 1 missing."""
        return self.n_buckets + 2
