"""IBMTabFormerAdapter — prepare the IBM TabFormer dataset for PRAGMA training.

Transforms the IBM TabFormer synthetic credit-card CSV into a DatasetManifest:
  1. Validates that the CSV file exists (FileNotFoundError if not).
  2. Loads all customers via TabFormerAdapter and counts rows.
  3. Fits FinancialTokenizerPipeline on the first 80% of customers (training split).
  4. Rebuilds the vocabulary layout after fitting (required — see fit_tokenizer.py).
  5. Serialises the fitted pipeline to a local vocab file (pickle).
  6. If upload=True: uploads CSV and vocab to S3 via pragma-workbench-env credentials.
  7. Returns a DatasetManifest with one CSV shard, vocab_uri, source metadata, and
     the exact PRAGMAConfig instance passed in.

IBM-specific S3 path conventions are contained entirely within this file.
DatasetManifest remains format-agnostic.

TabFormer dataset reference:
  https://github.com/IBM/TabFormer
  IBM, 2021. Tabular Transformers for Modeling Multivariate Time Series.
  arXiv:2011.01843

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.dataset_manifest import DatasetManifest, DatasetShard
from src.data.tabformer_adapter import TabFormerAdapter
from src.model.config import PRAGMAConfig
from src.tokenizer import FinancialTokenizerPipeline

# ---------------------------------------------------------------------------
# IBM TabFormer S3 path conventions — contained here only, not in the manifest
# ---------------------------------------------------------------------------

_DATASET_NAME = "ibm-tabformer"
_DATASET_VERSION = "v1"
_S3_PREFIX = "pragma-encoder/data/tabformer"
_S3_PREPARED_PREFIX = f"{_S3_PREFIX}/"
_S3_CSV_KEY = f"{_S3_PREFIX}/card_transaction.v1.csv"
_S3_VOCAB_KEY = f"{_S3_PREFIX}/vocab.pkl"

# Fields fitted by FinancialTokenizerPipeline (timestamp drives temporal encoding,
# not discrete fitting — excluded per fit_tokenizer.py)
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

# Fraction of customers used for tokeniser fitting (remainder = validation split)
_TRAIN_SPLIT = 0.8


class IBMTabFormerAdapter:
    """Prepare the IBM TabFormer synthetic credit-card dataset for PRAGMA training.

    Args:
        csv_path:   Path to the TabFormer CSV file (card_transaction.v1.csv).
        vocab_path: Local path where the fitted vocab will be written.
                    Defaults to csv_path.parent / "vocab.pkl".

    Example (local mode, no S3)::

        adapter = IBMTabFormerAdapter("data/tabformer/card_transaction.v1.csv")
        manifest = adapter.prepare(PRAGMAConfig.pragma_s(), upload=False)
        # manifest.shards[0].format == "csv"
        # manifest.vocab_uri == "pragma-encoder/data/tabformer/vocab.pkl"
        # manifest.config.max_event_tokens == 24  (§2.4)

    Example (cluster mode, uploads to S3)::

        adapter = IBMTabFormerAdapter("data/tabformer/card_transaction.v1.csv")
        manifest = adapter.prepare(PRAGMAConfig.pragma_s(), upload=True)
    """

    #: Environment variable that overrides the default csv_path.
    #: Set this before instantiating IBMTabFormerAdapter (e.g. in a KFP pipeline
    #: component pod) to control which CSV file is used without changing call sites.
    ENV_DATA_PATH = "IBM_TABFORMER_DATA_PATH"

    #: Default local path used when neither csv_path nor IBM_TABFORMER_DATA_PATH
    #: is provided.  Matches the conventional working-directory layout used by
    #: scripts/upload_training_data.py and the Level 3b batch/v1 Job smoke.
    DEFAULT_CSV_PATH = "data/tabformer/card_transaction.v1.csv"

    def __init__(
        self,
        csv_path: Path | str | None = None,
        vocab_path: Path | str | None = None,
    ) -> None:
        if csv_path is None:
            csv_path = os.environ.get(self.ENV_DATA_PATH, self.DEFAULT_CSV_PATH)
        self._csv_path = Path(csv_path)
        self._vocab_path = (
            Path(vocab_path) if vocab_path is not None
            else self._csv_path.parent / "vocab.pkl"
        )

    @property
    def dataset_name(self) -> str:
        """Registry key for this adapter."""
        return _DATASET_NAME

    def prepare(
        self,
        config: PRAGMAConfig,
        upload: bool = True,
    ) -> DatasetManifest:
        """Prepare the TabFormer dataset and return a DatasetManifest.

        Steps:
          1. Validate CSV exists — FileNotFoundError if not.
          2. Load all customers, count rows.
          3. Fit FinancialTokenizerPipeline on the training split.
          4. Rebuild vocabulary layout (required after fitting categorical fields).
          5. Serialise fitted pipeline to self._vocab_path.
          6. If upload=True: upload CSV and vocab to S3.
          7. Return DatasetManifest.

        Args:
            config: PRAGMAConfig carrying §2.4 truncation limits. The returned
                    manifest carries this exact instance (not a copy).
            upload: If True, upload CSV and vocab to S3. False for local dev/tests
                    — no S3 credentials are accessed.

        Returns:
            DatasetManifest with one CSV shard, vocab_uri, source metadata, and
            the given config.

        Raises:
            FileNotFoundError: If self._csv_path does not exist.
        """
        # Step 1 — fail early if source data is missing
        if not self._csv_path.exists():
            raise FileNotFoundError(
                f"IBM TabFormer CSV not found: {self._csv_path}\n"
                "Download from https://github.com/IBM/TabFormer and place it "
                "at that path, or upload it to S3 first."
            )

        # Step 2 — load all customers and count rows
        tab_adapter = TabFormerAdapter(self._csv_path)
        customers = list(tab_adapter.iter_customers())
        row_count = len(customers)

        # Step 3 — fit tokeniser on training split
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

        # Step 4 — rebuild vocabulary layout after categorical fields are fitted
        # (CategoricalTokenizer.vocab_size starts at 0 before fit(); offsets
        # computed in __init__ are stale until this is called)
        pipeline._build_vocabulary_layout()

        # Step 5 — serialise fitted pipeline locally
        self._vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._vocab_path, "wb") as f:
            pickle.dump(pipeline, f)

        # Step 6 — upload to S3 (skipped when upload=False)
        if upload:
            self._upload_to_s3()

        # Step 7 — return the manifest
        shard = DatasetShard(
            uri=_S3_CSV_KEY,
            format="csv",
            rows=row_count,
        )

        return DatasetManifest(
            dataset_name=_DATASET_NAME,
            dataset_version=_DATASET_VERSION,
            prepared_prefix_uri=_S3_PREPARED_PREFIX,
            shards=(shard,),
            vocab_uri=_S3_VOCAB_KEY,
            schema_uri=None,
            manifest_uri=None,
            row_count=row_count,
            source={
                "origin": "ibm-tabformer",
                "format": "csv",
                "csv_path": str(self._csv_path),
            },
            config=config,
        )

    def _upload_to_s3(self) -> None:
        """Upload CSV and vocab to S3 using pragma-workbench-env credentials.

        Reads credentials from environment variables:
            MODEL_REGISTRY_BUCKET, MODEL_REGISTRY_ENDPOINT,
            MODEL_REGISTRY_ACCESS_KEY, MODEL_REGISTRY_SECRET_KEY

        Raises:
            EnvironmentError: If any required credential env var is missing.
            Exception: boto3 raises on S3 connection or upload failure.
        """
        import boto3  # optional import — not required when upload=False

        required = [
            "MODEL_REGISTRY_BUCKET",
            "MODEL_REGISTRY_ENDPOINT",
            "MODEL_REGISTRY_ACCESS_KEY",
            "MODEL_REGISTRY_SECRET_KEY",
        ]
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            raise EnvironmentError(
                f"S3 upload requires these env vars (from pragma-workbench-env): "
                f"{missing}"
            )

        bucket = os.environ["MODEL_REGISTRY_BUCKET"]
        endpoint = os.environ["MODEL_REGISTRY_ENDPOINT"]
        access_key = os.environ["MODEL_REGISTRY_ACCESS_KEY"]
        secret_key = os.environ["MODEL_REGISTRY_SECRET_KEY"]

        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{endpoint}",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        s3.upload_file(str(self._csv_path), bucket, _S3_CSV_KEY)
        s3.upload_file(str(self._vocab_path), bucket, _S3_VOCAB_KEY)
