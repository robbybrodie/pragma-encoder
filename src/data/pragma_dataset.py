"""PragmaDataset — torch.utils.data.Dataset over IBM TabFormer transactions.

Loads a fitted FinancialTokenizerPipeline from disk, reads all customers
from the TabFormer CSV via TabFormerAdapter, and returns per-customer
tokenised sequences as tensor dicts ready for EmbeddingAssembler.

Each sample covers one customer's full transaction history up to ne_max
events. Shorter histories are padded; longer ones are truncated.

Tensor shapes per sample (no batch dimension — DataLoader adds it):
    xe_key_ids  : (ne_max, ni_max)  int64  — event key token IDs
    xe_val_ids  : (ne_max, ni_max)  int64  — event value token IDs (pre-mask)
    xe_pos_ids  : (ne_max, ni_max)  int64  — within-field position IDs
    xe_valid    : (ne_max, ni_max)  bool   — True = real token, False = padding
    xt          : (ne_max, 3)       float32 — [hour, day_of_week, day_of_month]
    te          : (1 + ne_max,)     float32 — [USR]=0.0, then log-sec coords
    xa_key_ids  : (1,)              int64  — TD-003: single placeholder profile token
    xa_val_ids  : (1,)              int64
    xa_pos_ids  : (1,)              int64
    ta          : (1,)              float32
    n_events    : ()                int64  — number of real (non-padded) events

Training loop contract:
    After calling MaskingStrategy.forward(xe_val_ids, xe_key_ids):
        mlm_mask &= batch["xe_valid"]   # zero out padding token positions
    This ensures padded positions never contribute to the MLM loss.

TD-003 (docs/tech-debt.md):
    The profile path (xa_*) uses a single placeholder token (key_start /
    value_start) until ProfileTokenizerPipeline is implemented.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.data.tabformer_adapter import TabFormerAdapter
from src.tokenizer.pipeline import TokenizerPipeline


# Fields passed to encode_event in the order they appear in each transaction
_EVENT_FIELDS = [
    "amount_local",
    "amount_usd",
    "merchant_category",
    "currency",
    "country",
    "transaction_type",
    "channel",
    "merchant_name",
    "description",
]


class PragmaDataset(Dataset):
    """torch.utils.data.Dataset over tokenised TabFormer customer sequences.

    Args:
        csv_path:   Path to card_transaction.v1.csv.
        vocab_path: Path to fitted pipeline pickle produced by fit_tokenizer.py.
        split:      "train" (first 80% of customers) or "val" (remaining 20%).
        ne_max:     Maximum number of events per sample. Longer histories are
                    truncated; shorter ones are zero-padded.
        ni_max:     Maximum number of tokens per event (= config.max_event_tokens).
                    Matches PRAGMA-S default of 24.
    """

    def __init__(
        self,
        csv_path: Path | str,
        vocab_path: Path | str,
        split: str = "train",
        ne_max: int = 50,
        ni_max: int = 24,
    ) -> None:
        super().__init__()
        self.ne_max = ne_max
        self.ni_max = ni_max

        # Load fitted tokeniser pipeline
        with open(vocab_path, "rb") as f:
            self.pipeline: TokenizerPipeline = pickle.load(f)

        vocab_spec = self.pipeline.vocabulary_spec()
        self._key_pad = vocab_spec.key_start
        self._val_pad = vocab_spec.value_start

        # Load all customers and apply train/val split
        adapter = TabFormerAdapter(csv_path)
        all_customers = list(adapter.iter_customers())
        n_train = int(len(all_customers) * 0.8)
        if split == "train":
            self._customers = all_customers[:n_train]
        elif split == "val":
            self._customers = all_customers[n_train:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    def __len__(self) -> int:
        return len(self._customers)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        _cid, transactions = self._customers[idx]
        ne_max = self.ne_max
        ni_max = self.ni_max

        # Allocate output tensors (padding values for empty slots)
        xe_key_ids = torch.full((ne_max, ni_max), self._key_pad, dtype=torch.int64)
        xe_val_ids = torch.full((ne_max, ni_max), self._val_pad, dtype=torch.int64)
        xe_pos_ids = torch.zeros((ne_max, ni_max), dtype=torch.int64)
        xe_valid   = torch.zeros((ne_max, ni_max), dtype=torch.bool)
        xt         = torch.zeros((ne_max, 3), dtype=torch.float32)
        te_events  = torch.zeros(ne_max, dtype=torch.float32)

        # Profile path — TD-003: single placeholder token
        xa_key_ids = torch.tensor([self._key_pad], dtype=torch.int64)
        xa_val_ids = torch.tensor([self._val_pad], dtype=torch.int64)
        xa_pos_ids = torch.zeros(1, dtype=torch.int64)
        ta         = torch.zeros(1, dtype=torch.float32)

        # Encode events
        events = transactions[:ne_max]
        n_events = len(events)
        t_prev: Optional[object] = None  # pd.Timestamp

        for i, txn in enumerate(events):
            ts = txn["timestamp"]

            # Compute elapsed seconds from the previous event (Equation 2)
            t_seconds = 0.0
            if t_prev is not None:
                delta = ts - t_prev
                t_seconds = max(0.0, delta.total_seconds())
            t_prev = ts

            # Build the field list for this event
            fields = [
                (field, txn[field], ts)
                for field in _EVENT_FIELDS
                if field in txn
            ]

            enc = self.pipeline.encode_event(fields, t_seconds=t_seconds)

            # Pack key/value/pos token IDs, truncating to ni_max
            n_tok = min(len(enc.key_ids), ni_max)
            if n_tok > 0:
                xe_key_ids[i, :n_tok] = torch.tensor(enc.key_ids[:n_tok], dtype=torch.int64)
                xe_val_ids[i, :n_tok] = torch.tensor(enc.value_ids[:n_tok], dtype=torch.int64)
                xe_pos_ids[i, :n_tok] = torch.tensor(enc.position_ids[:n_tok], dtype=torch.int64)
                xe_valid[i, :n_tok]   = True

            # Calendar features [hour, day_of_week, day_of_month]
            if len(enc.calendar_features) == 3:
                xt[i] = torch.tensor(enc.calendar_features, dtype=torch.float32)

            # Temporal coordinate for this event (log-seconds)
            te_events[i] = enc.temporal_coord

        # te: [USR]=0.0 at position 0, then event temporal coords
        te = torch.cat([torch.zeros(1, dtype=torch.float32), te_events])  # (1+ne_max,)

        return {
            "xe_key_ids": xe_key_ids,   # (ne_max, ni_max) int64
            "xe_val_ids": xe_val_ids,   # (ne_max, ni_max) int64
            "xe_pos_ids": xe_pos_ids,   # (ne_max, ni_max) int64
            "xe_valid":   xe_valid,     # (ne_max, ni_max) bool
            "xt":         xt,           # (ne_max, 3) float32
            "te":         te,           # (1+ne_max,) float32
            "xa_key_ids": xa_key_ids,   # (1,) int64
            "xa_val_ids": xa_val_ids,   # (1,) int64
            "xa_pos_ids": xa_pos_ids,   # (1,) int64
            "ta":         ta,           # (1,) float32
            "n_events":   torch.tensor(n_events, dtype=torch.int64),
        }
