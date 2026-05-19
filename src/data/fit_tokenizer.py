"""Fit FinancialTokenizerPipeline on TabFormer data and persist the vocabulary.

Usage::

    python src/data/fit_tokenizer.py

Reads  : data/tabformer/card_transaction.v1.csv
Writes : data/tabformer/vocab.pkl   (fitted TokenizerPipeline, ~50 MB)

The first 80% of customers (by ascending User ID) are used for fitting.
The remaining 20% form the validation split used by PragmaDataset.

After fitting all individual field tokenisers the pipeline vocabulary layout
is rebuilt via _build_vocabulary_layout() — this is required because
CategoricalTokenizer.vocab_size starts at 0 and grows only after fit(),
so the offsets computed in __init__ are stale until explicitly refreshed.

TD-003 note: only FinancialTokenizerPipeline fields are fitted here.
The ProfileTokenizerPipeline (static customer attributes) is not yet
implemented — see docs/tech-debt.md.
"""

from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tokenizer import FinancialTokenizerPipeline
from src.data.tabformer_adapter import TabFormerAdapter

CSV_PATH = Path("data/tabformer/card_transaction.v1.csv")
VOCAB_PATH = Path("data/tabformer/vocab.pkl")

# Fields present in TabFormer after adapter mapping.
# 'timestamp' is excluded — it drives temporal encoding, not discrete fitting.
_FITTABLE_FIELDS = [
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


def main() -> None:
    if not CSV_PATH.exists():
        print(
            f"TabFormer CSV not found at {CSV_PATH}\n"
            "Download from https://github.com/IBM/TabFormer/releases\n"
            f"and place it at that path before running."
        )
        sys.exit(1)

    print(f"Loading {CSV_PATH} ...")
    adapter = TabFormerAdapter(CSV_PATH)
    customers = list(adapter.iter_customers())
    print(f"  {len(customers)} unique customers found")

    n_train = int(len(customers) * 0.8)
    train_customers = customers[:n_train]
    print(f"  Using first {n_train} customers for fitting ({n_train / len(customers):.0%})")

    # Accumulate per-field raw values from training customers
    field_data: dict[str, list] = defaultdict(list)
    n_txns = 0
    for _cid, transactions in train_customers:
        for txn in transactions:
            for field in _FITTABLE_FIELDS:
                if field in txn:
                    field_data[field].append(txn[field])
            n_txns += 1
    print(f"  {n_txns:,} training transactions accumulated\n")

    pipeline = FinancialTokenizerPipeline()

    print("Fitting tokenisers:")
    for field in _FITTABLE_FIELDS:
        data = field_data.get(field, [])
        if not data:
            print(f"  {field}: no data — skipping")
            continue
        pipeline.field_tokenizers[field].fit(data)
        print(f"  {field}: vocab_size={pipeline.field_tokenizers[field].vocab_size}")

    # CRITICAL: rebuild vocabulary layout after fitting.
    # CategoricalTokenizer.vocab_size starts at 0 before fit(); the layout
    # computed in TokenizerPipeline.__init__ therefore has incorrect offsets
    # for all categorical fields. Rebuilding here propagates the correct sizes.
    pipeline._build_vocabulary_layout()

    vocab_spec = pipeline.vocabulary_spec()
    print(f"\nVocabulary layout:")
    print(f"  special_tokens : 0..{pipeline.N_SPECIAL_TOKENS - 1}")
    print(f"  key tokens     : {vocab_spec.key_start}..{vocab_spec.key_start + vocab_spec.key_size - 1}  ({vocab_spec.key_size} keys)")
    print(f"  value tokens   : {vocab_spec.value_start}..{vocab_spec.value_start + vocab_spec.value_size - 1}  ({vocab_spec.value_size} values)")
    print(f"  total_embedding_vocab_size = {vocab_spec.total_embedding_vocab_size}")

    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nPipeline serialised → {VOCAB_PATH}")
    print("Run scripts/train_pragma.py --csv-path data/tabformer/card_transaction.v1.csv "
          "--vocab-path data/tabformer/vocab.pkl to begin training.")


if __name__ == "__main__":
    main()
