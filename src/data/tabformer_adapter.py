"""TabFormer CSV adapter for PRAGMA tokenisation.

Maps IBM TabFormer synthetic credit-card transaction columns to the field
names used by FinancialTokenizerPipeline, groups by customer (User), and
returns per-customer transaction sequences sorted by timestamp.

TabFormer dataset: https://github.com/IBM/TabFormer
CSV columns: User, Card, Year, Month, Day, Time, Amount, Use Chip,
             Merchant Name, Merchant City, Merchant State, MCC,
             Errors?, Is Fraud?

Field mapping to FinancialTokenizerPipeline:
    amount_local      ← Amount  (strip '$', cast to float)
    amount_usd        ← Amount  (same — currency is always USD in TabFormer)
    merchant_category ← MCC     (str)
    currency          ← "USD"   (constant)
    country           ← "US"    (constant)
    transaction_type  ← Use Chip
    channel           ← derived from Use Chip
    merchant_name     ← Merchant Name
    description       ← Merchant City + " " + Merchant State
    timestamp         ← Year/Month/Day/Time  (pandas Timestamp)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# Mapping from "Use Chip" values to a normalised channel label
_CHIP_TO_CHANNEL: Dict[str, str] = {
    "Chip Transaction": "chip",
    "Online Transaction": "online",
    "Swipe Transaction": "swipe",
}


def _parse_amount(raw: str) -> float:
    """Strip leading '$' and commas, return float."""
    return float(raw.replace("$", "").replace(",", ""))


def _parse_timestamp(year: int, month: int, day: int, time_str: str) -> pd.Timestamp:
    """Build a pandas Timestamp from TabFormer Year/Month/Day/Time columns."""
    hour, minute = (int(x) for x in time_str.split(":"))
    return pd.Timestamp(year=int(year), month=int(month), day=int(day),
                        hour=hour, minute=minute)


def _row_to_fields(row: pd.Series) -> Dict:
    """Convert a single TabFormer row to a dict of pipeline field values."""
    ts = _parse_timestamp(row["Year"], row["Month"], row["Day"], row["Time"])
    amount = _parse_amount(row["Amount"])
    use_chip = str(row["Use Chip"]).strip()
    channel = _CHIP_TO_CHANNEL.get(use_chip, "other")
    city = str(row["Merchant City"]).strip()
    state = str(row["Merchant State"]).strip()
    return {
        "timestamp": ts,
        "amount_local": amount,
        "amount_usd": amount,
        "merchant_category": str(row["MCC"]).strip(),
        "currency": "USD",
        "country": "US",
        "transaction_type": use_chip,
        "channel": channel,
        "merchant_name": str(row["Merchant Name"]).strip(),
        "description": f"{city} {state}".strip(),
    }


class TabFormerAdapter:
    """Load and adapt the TabFormer CSV for use with FinancialTokenizerPipeline.

    Args:
        csv_path: Path to card_transaction.v1.csv (or equivalent).

    Example::

        adapter = TabFormerAdapter("data/tabformer/card_transaction.v1.csv")
        for customer_id, transactions in adapter.iter_customers():
            # transactions: List[dict], sorted by timestamp
            ...
    """

    def __init__(self, csv_path: Path | str) -> None:
        self._csv_path = Path(csv_path)
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self._csv_path, low_memory=False)
        return self._df

    def iter_customers(
        self,
    ) -> Iterable[Tuple[str, List[Dict]]]:
        """Yield (customer_id, transactions) for each unique User.

        Customers are yielded in ascending User-ID order.
        Transactions within each customer are sorted by timestamp.

        Yields:
            customer_id: str form of the User column value.
            transactions: list of field dicts, each containing:
                'timestamp', 'amount_local', 'amount_usd',
                'merchant_category', 'currency', 'country',
                'transaction_type', 'channel', 'merchant_name', 'description'
        """
        df = self._load()
        for user_id in sorted(df["User"].unique()):
            subset = df[df["User"] == user_id].copy()
            rows = [_row_to_fields(row) for _, row in subset.iterrows()]
            rows.sort(key=lambda r: r["timestamp"])
            yield str(user_id), rows
