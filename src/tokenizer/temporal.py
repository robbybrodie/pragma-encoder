"""Temporal field tokeniser — log-seconds coordinate + calendar features.

Implements the temporal encoding scheme from PRAGMA paper Section 2.2.

PRAGMA uses two separate temporal encodings:

1. Temporal coordinate (Equation 2) — a continuous float for RoPE:
       t' = 8 · ln(1 + t/8)
   where t = elapsed seconds since most recent event.
   Scale constant 8 controls the linear-to-log crossover.
   key-numbers.md: temporal_transform = 8·ln(1+t/8), §2.2

2. Calendar features (§2.2) — three discrete values per event for the
   EventEncoder calendar MLP:
       [hour_of_day, day_of_week, day_of_month]
   Exactly 3 features. Month and quarter are NOT included.
   key-numbers.md: calendar_feature_dims = 3, §2.2

These are separate outputs. The temporal coordinate feeds RoPE in the
Profile State Encoder and History Encoder. The calendar features feed
the 2-layer calendar MLP in the Event Encoder (Equation 3).

Reference: Ostroukhov et al. (2026), Section 2.2 and Section 2.3.3
"""

import math
from datetime import datetime, timezone
from typing import Any, List, Union

from .base import BaseTokenizer

# Paper constant — Equation 2, key-numbers.md: temporal_transform_scale = 8
_TEMPORAL_SCALE: float = 8.0

# Calendar feature indices (§2.2: exactly 3)
_CALENDAR_DIMS = 3  # key-numbers.md: calendar_feature_dims = 3, §2.2


class TemporalTokenizer(BaseTokenizer):
    """Log-seconds temporal coordinate + calendar feature extractor.

    Two public methods beyond the BaseTokenizer interface:

      compute_temporal_coordinate(t_seconds) -> float
          Equation 2: t' = 8·ln(1 + t/8)

      extract_calendar_features(timestamp) -> List[float]
          §2.2: [hour_of_day, day_of_week, day_of_month] — exactly 3 values.

    The encode() method returns the 3 calendar feature integers as discrete
    token IDs for embedding table lookup (0-indexed).

    Args:
        n_log_buckets: Retained for backward compatibility. Not used in the
                       paper-derived temporal coordinate (which is continuous).
    """

    # key-numbers.md: calendar_feature_dims = 3, §2.2
    N_CALENDAR_DIMS: int = _CALENDAR_DIMS

    def __init__(self, n_log_buckets: int = 128) -> None:
        # n_log_buckets kept for backward compatibility with existing
        # callers (e.g. build_financial_pipeline). Not used in vocab_size
        # or the paper-derived temporal coordinate.
        self.n_log_buckets = n_log_buckets
        self._fitted = True  # no training data required

    # ------------------------------------------------------------------
    # BaseTokenizer interface
    # ------------------------------------------------------------------

    def fit(self, data: List[Any]) -> "TemporalTokenizer":
        """No-op — temporal boundaries are fixed, not learned.

        Args:
            data: Unused. Included for API compatibility with BaseTokenizer.

        Returns:
            Self.
        """
        return self

    def encode(self, value: Any) -> List[int]:
        """Encode a timestamp to calendar feature token IDs.

        Returns exactly 3 integer token IDs:
            [hour (0-23), day_of_week (0-6), day_of_month (0-30)]

        §2.2: calendar_feature_dims = 3 (hour, day_of_week, day_of_month).
        Month and quarter are NOT included — not in the paper.

        Args:
            value: datetime, Unix timestamp (float/int), ISO 8601 string,
                   or None (yields [0, 0, 0]).

        Returns:
            List of exactly 3 integer token IDs (0-indexed).
        """
        if value is None:
            return [0, 0, 0]

        dt = self._to_datetime(value)
        return [
            dt.hour,           # 0–23
            dt.weekday(),      # 0 Mon … 6 Sun
            dt.day - 1,        # 0–30 (0-indexed for embedding table)
        ]

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        """Return a human-readable description of calendar token IDs.

        Args:
            token_ids: List of 3 token IDs from encode().

        Returns:
            Descriptive string (not a reconstructed datetime).
        """
        ids = [token_ids] if isinstance(token_ids, int) else list(token_ids)
        return f"hour={ids[0]}, dow={ids[1]}, dom={ids[2] + 1}"

    @property
    def vocab_size(self) -> int:
        """Token vocabulary size: 3 calendar ranges.

        §2.2: calendar_feature_dims = 3.
        hour (24) + day_of_week (7) + day_of_month (31) = 62 tokens.
        """
        return 24 + 7 + 31  # = 62

    # ------------------------------------------------------------------
    # Paper-derived methods (beyond BaseTokenizer interface)
    # ------------------------------------------------------------------

    def compute_temporal_coordinate(self, t_seconds: float) -> float:
        """Equation 2: t' = 8·ln(1 + t/8).

        Transforms elapsed seconds into a log-compressed coordinate for
        RoPE positional encoding. The scale constant 8 preserves linear
        granularity for recent events (t < 8 s) while compressing large
        temporal gaps logarithmically.

        key-numbers.md: temporal_transform = 8·ln(1+t/8), §2.2
        key-numbers.md: temporal_transform_scale = 8, §2.2

        Args:
            t_seconds: Elapsed time since most recent event, in seconds.
                       Must be non-negative. Negative values are clamped to 0.

        Returns:
            t' — log-seconds temporal coordinate (continuous float).
        """
        t = max(0.0, float(t_seconds))
        return _TEMPORAL_SCALE * math.log(1.0 + t / _TEMPORAL_SCALE)

    def extract_calendar_features(self, timestamp: Any) -> List[float]:
        """Extract calendar features: [hour, day_of_week, day_of_month].

        §2.2: exactly 3 calendar features used by EventEncoder calendar MLP.
        Month and quarter are NOT included — not in the paper.
        key-numbers.md: calendar_feature_dims = 3, §2.2

        Features:
            hour        ∈ [0, 23]
            day_of_week ∈ [0, 6]   (0=Monday, 6=Sunday)
            day_of_month ∈ [1, 31]

        Args:
            timestamp: datetime, Unix timestamp (float/int), or ISO 8601 string.

        Returns:
            [hour, day_of_week, day_of_month] — exactly 3 floats.
        """
        dt = self._to_datetime(timestamp)
        return [
            float(dt.hour),       # 0–23
            float(dt.weekday()),  # 0 Mon … 6 Sun
            float(dt.day),        # 1–31
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_datetime(value: Any) -> datetime:
        """Coerce value to a timezone-aware datetime."""
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        raise TypeError(f"Cannot convert {type(value)} to datetime.")
