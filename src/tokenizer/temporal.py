"""Temporal field tokeniser — log-seconds + calendar encoding.

Implements the temporal tokenisation strategy from PRAGMA paper Section 2.2.

PRAGMA uses a two-part encoding for timestamp fields:
    1. Log-seconds offset: the log of elapsed seconds since a reference
       epoch, discretised into buckets. Captures relative timing at
       multiple time scales (minutes to years) without arithmetic blowup.
    2. Calendar tokens: discrete tokens for hour-of-day, day-of-week,
       day-of-month, month, and quarter. These capture periodic patterns
       (lunch-hour spending, weekend behaviour, month-end salary credits).

The calendar tokens are passed through the Event Encoder's calendar
embedding table (Section 2.3.3) where they receive dedicated embeddings.

Reference: Ostroukhov et al. (2026), Section 2.2 and Section 2.3.3
"""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

from .base import BaseTokenizer

# Calendar field ranges (for discrete token offset computation)
_CALENDAR_RANGES: Dict[str, int] = {
    "hour": 24,       # 0–23
    "dow": 7,         # 0 Monday … 6 Sunday
    "dom": 31,        # 1–31
    "month": 12,      # 1–12
    "quarter": 4,     # 1–4
}


class TemporalTokenizer(BaseTokenizer):
    """Log-seconds + calendar tokeniser for timestamp fields.

    Encodes a datetime as:
        [log_seconds_bucket, hour, day_of_week, day_of_month, month, quarter]

    The log-seconds bucket captures time delta from a reference epoch.
    Calendar tokens capture periodic behavioural patterns.

    Args:
        n_log_buckets: Number of log-seconds buckets. Default: 128.
        reference_epoch: Unix timestamp (seconds) of the reference epoch.
                         Defaults to 2020-01-01 00:00:00 UTC.
        max_log_seconds: Upper bound for log-seconds bucketing (seconds).
                         Default: 3 years in seconds.
    """

    def __init__(
        self,
        n_log_buckets: int = 128,
        reference_epoch: float = 1577836800.0,  # 2020-01-01 UTC
        max_log_seconds: float = 3 * 365 * 24 * 3600,
    ):
        self.n_log_buckets = n_log_buckets
        self.reference_epoch = reference_epoch
        self.max_log_seconds = max_log_seconds
        self._log_max = math.log1p(max_log_seconds)
        self._fitted = True  # no training data required

    def fit(self, data: List[Any]) -> "TemporalTokenizer":
        """No-op for temporal tokeniser (boundaries are fixed, not learned).

        Args:
            data: Unused. Included for API compatibility.

        Returns:
            Self.
        """
        return self

    def encode(self, value: Any) -> List[int]:
        """Encode a datetime to [log_bucket, hour, dow, dom, month, quarter].

        Args:
            value: A datetime object, Unix timestamp (float/int), or
                   ISO 8601 string. None/NaN yields all-zero tokens.

        Returns:
            List of 6 token IDs.
        """
        if value is None:
            return [0] * 6

        dt = self._to_datetime(value)

        # Log-seconds bucket
        delta = max(0.0, dt.timestamp() - self.reference_epoch)
        log_val = math.log1p(delta)
        log_bucket = min(
            int(log_val / self._log_max * (self.n_log_buckets - 1)),
            self.n_log_buckets - 1,
        )

        # Calendar tokens (1-indexed fields adjusted to 0-indexed tokens)
        hour = dt.hour                         # 0–23
        dow = dt.weekday()                     # 0 Mon … 6 Sun
        dom = dt.day - 1                       # 0–30
        month = dt.month - 1                   # 0–11
        quarter = (dt.month - 1) // 3         # 0–3

        return [log_bucket, hour, dow, dom, month, quarter]

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        """Return a human-readable description of the encoded time.

        Args:
            token_ids: List of 6 token IDs from encode().

        Returns:
            Descriptive string (not a reconstructed datetime).
        """
        ids = list(token_ids) if not isinstance(token_ids, list) else token_ids
        return (
            f"log_bucket={ids[0]}, hour={ids[1]}, "
            f"dow={ids[2]}, dom={ids[3]+1}, month={ids[4]+1}, quarter={ids[5]+1}"
        )

    @property
    def vocab_size(self) -> int:
        """Total tokens: log buckets + sum of calendar ranges."""
        return self.n_log_buckets + sum(_CALENDAR_RANGES.values())

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
