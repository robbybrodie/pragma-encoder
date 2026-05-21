"""IBMTabFormerSmokeAdapter — synthetic smoke dataset for KFP component testing.

Generates a minimal 15-row IBM TabFormer-compatible CSV in a local tempdir,
fits the FinancialTokenizerPipeline on it, and returns a DatasetManifest.

Purpose:
    Prove that the prepare_dataset KFP component executes correctly inside
    a component pod, without requiring:
    - Real IBM TabFormer data
    - S3 credentials
    - Network access

This adapter exists solely to make the Level 3 KFP pipeline smoke test pass
end-to-end. Real IBM TabFormer ingestion uses IBMTabFormerAdapter ("ibm-tabformer").

Safety rules:
    - NEVER used as a substitute for real data in training.
    - The registry key "ibm-tabformer-smoke" is explicitly different from
      "ibm-tabformer" so callers cannot accidentally use smoke data in production.
    - upload parameter is silently ignored — this adapter never writes to S3.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

from __future__ import annotations

import pickle
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from pragma_encoder.data.dataset_manifest import DatasetManifest, DatasetShard
from pragma_encoder.data.tabformer_adapter import TabFormerAdapter
from pragma_encoder.model.config import PRAGMAConfig
from pragma_encoder.tokenizer import FinancialTokenizerPipeline

# ---------------------------------------------------------------------------
# Smoke dataset identity
# ---------------------------------------------------------------------------

_DATASET_NAME = "ibm-tabformer-smoke"
_DATASET_VERSION = "smoke-v1"

# ---------------------------------------------------------------------------
# Synthetic CSV — same 15-row fixture used by the Level 3b training job smoke.
# Five customers × three transactions each; all required IBM TabFormer columns.
# ---------------------------------------------------------------------------

_SMOKE_CSV = """\
User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,MCC,Errors?,Is Fraud?
0,0,2020,1,1,12:00,$10.00,Chip Transaction,Store A,New York,NY,5411,,No
0,0,2020,1,2,14:30,$25.50,Swipe Transaction,Store B,Los Angeles,CA,5912,,No
0,0,2020,1,3,09:15,$7.99,Online Transaction,Store C,Houston,TX,5734,,No
1,0,2020,1,4,11:00,$50.00,Chip Transaction,Store D,Seattle,WA,5411,,No
1,0,2020,1,5,16:45,$12.00,Swipe Transaction,Store E,Miami,FL,5912,,No
1,0,2020,1,6,08:30,$99.99,Online Transaction,Store A,New York,NY,5734,,No
2,0,2020,1,7,13:00,$35.00,Chip Transaction,Store B,Los Angeles,CA,5411,,No
2,0,2020,1,8,15:30,$8.50,Swipe Transaction,Store C,Houston,TX,5912,,No
2,0,2020,1,9,10:00,$45.00,Online Transaction,Store D,Seattle,WA,5734,,No
3,0,2020,1,10,12:00,$20.00,Chip Transaction,Store E,Miami,FL,5411,,No
3,0,2020,1,11,14:00,$15.00,Swipe Transaction,Store A,New York,NY,5912,,No
3,0,2020,1,12,16:00,$30.00,Online Transaction,Store B,Los Angeles,CA,5734,,No
4,0,2020,1,13,11:00,$55.00,Chip Transaction,Store C,Houston,TX,5411,,No
4,0,2020,1,14,13:00,$22.00,Swipe Transaction,Store D,Seattle,WA,5912,,No
4,0,2020,1,15,15:00,$18.00,Online Transaction,Store E,Miami,FL,5734,,No
"""

# Fields fitted by FinancialTokenizerPipeline (timestamp excluded — not discretely fitted)
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

_TRAIN_SPLIT = 0.8


class IBMTabFormerSmokeAdapter:
    """Smoke adapter: synthetic IBM TabFormer data, no S3, for KFP smoke testing.

    Writes a 15-row CSV to a tempdir, fits the tokenizer, returns a manifest.
    The upload parameter is ignored — this adapter never writes to S3.

    Example::

        adapter = IBMTabFormerSmokeAdapter()
        manifest = adapter.prepare(PRAGMAConfig.pragma_s(), upload=False)
        # manifest.dataset_name == "ibm-tabformer-smoke"
        # manifest.shards[0].uri is a local /tmp/... path
    """

    @property
    def dataset_name(self) -> str:
        """Registry key for this adapter."""
        return _DATASET_NAME

    def prepare(
        self,
        config: PRAGMAConfig,
        upload: bool = False,
    ) -> DatasetManifest:
        """Generate synthetic CSV, fit tokenizer, return DatasetManifest.

        The upload parameter is silently ignored — no S3 credentials are needed
        or accessed. This adapter is self-contained and works in any environment.

        Args:
            config: PRAGMAConfig carrying §2.4 truncation limits.
            upload: Ignored. Retained for DatasetAdapterProtocol compatibility.

        Returns:
            DatasetManifest with one local CSV shard and local vocab_uri.
        """
        tmpdir = Path(tempfile.mkdtemp(prefix="pragma-smoke-"))

        # Step 1 — write synthetic CSV to tempdir
        csv_path = tmpdir / "card_transaction.smoke.csv"
        csv_path.write_text(_SMOKE_CSV)

        # Step 2 — load customers via TabFormerAdapter
        tab_adapter = TabFormerAdapter(csv_path)
        customers = list(tab_adapter.iter_customers())
        row_count = len(customers)

        # Step 3 — fit tokenizer on training split (80%)
        n_train = max(1, int(row_count * _TRAIN_SPLIT))
        train_customers = customers[:n_train]

        field_data: dict[str, list[Any]] = defaultdict(list)
        for _cid, transactions in train_customers:
            for txn in transactions:
                for field in _FITTABLE_FIELDS:
                    if field in txn:
                        field_data[field].append(txn[field])

        pipeline = FinancialTokenizerPipeline()
        for field in _FITTABLE_FIELDS:
            data = field_data.get(field, [])
            if data:
                pipeline.field_tokenizers[field].fit(data)

        # Step 4 — rebuild vocabulary layout
        pipeline._build_vocabulary_layout()

        # Step 5 — serialise fitted pipeline locally
        vocab_path = tmpdir / "vocab.pkl"
        with open(vocab_path, "wb") as f:
            pickle.dump(pipeline, f)

        # Step 6 — return manifest with local URIs (no S3)
        shard = DatasetShard(
            uri=str(csv_path),
            format="csv",
            rows=row_count,
        )

        return DatasetManifest(
            dataset_name=_DATASET_NAME,
            dataset_version=_DATASET_VERSION,
            prepared_prefix_uri=str(tmpdir),
            shards=(shard,),
            vocab_uri=str(vocab_path),
            schema_uri=None,
            manifest_uri=None,
            row_count=row_count,
            source={
                "origin": "ibm-tabformer-smoke",
                "format": "csv",
                "note": "synthetic 15-row dataset for KFP smoke testing only",
            },
            config=config,
        )
