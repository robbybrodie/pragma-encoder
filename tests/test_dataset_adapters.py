"""Tests for DatasetAdapterProtocol and IBMTabFormerAdapter.

Derived from PRAGMA paper Section 2.4:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

The adapter contract is format-agnostic (ADR 003). These tests verify:
  - IBMTabFormerAdapter satisfies DatasetAdapterProtocol
  - prepare(upload=False) returns a valid DatasetManifest without S3 access
  - The produced manifest carries §2.4 truncation limits
  - File-not-found is raised early (before any fitting or S3 access)
  - The adapter registry maps "ibm-tabformer" to IBMTabFormerAdapter

Test types:
    Contract tests:  verify Protocol compliance and registry
    Property tests:  verify §2.4 adapter contract invariants
    Spec tests:      verify exact values from docs/paper/key-numbers.md
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pragma_encoder.data.adapters import DatasetAdapterProtocol, IBMTabFormerAdapter, get_adapter
from pragma_encoder.data.dataset_manifest import DatasetManifest, DatasetManifestProtocol
from pragma_encoder.model.config import PRAGMAConfig

# ---------------------------------------------------------------------------
# Fixtures — minimal TabFormer CSV for local (upload=False) testing
# ---------------------------------------------------------------------------

# Column headers matching TabFormerAdapter's expected schema
_TABFORMER_HEADER = (
    "User,Card,Year,Month,Day,Time,Amount,Use Chip,"
    "Merchant Name,Merchant City,Merchant State,MCC,Errors?,Is Fraud?"
)


def _write_minimal_csv(path: Path) -> Path:
    """Write a minimal valid TabFormer CSV with 3 customers and several transactions.

    Provides enough variety for all field tokenisers to fit:
      - Numerical: Amount has multiple distinct values
      - Categorical: Use Chip has all three canonical values
      - Text: Merchant Name and description have distinct strings
    """
    rows = [
        # User 0 — 4 transactions
        "0,0,2019,1,1,00:00,$10.00,Chip Transaction,Coffee Shop,New York,NY,5411,,0",
        "0,0,2019,1,2,08:30,$25.50,Online Transaction,Book Store,Los Angeles,CA,5412,,0",
        "0,0,2019,2,1,12:00,$5.99,Swipe Transaction,Fast Food,Chicago,IL,5812,,0",
        "0,0,2019,2,14,18:45,$150.00,Chip Transaction,Electronics,Seattle,WA,5734,,0",
        # User 1 — 3 transactions
        "1,1,2019,1,3,09:00,$75.00,Online Transaction,Pharmacy,Boston,MA,5912,,0",
        "1,1,2019,3,5,14:30,$300.00,Chip Transaction,Jewellery Store,Miami,FL,5944,,0",
        "1,1,2019,4,1,11:15,$12.50,Swipe Transaction,Grocery,Denver,CO,5411,,0",
        # User 2 — 3 transactions (used for validation split, not fitting)
        "2,2,2019,1,10,10:00,$88.00,Chip Transaction,Hardware,Phoenix,AZ,5251,,0",
        "2,2,2019,2,2,16:00,$44.50,Online Transaction,Books,Portland,OR,5942,,0",
        "2,2,2019,3,15,08:00,$7.25,Swipe Transaction,Bakery,Austin,TX,5462,,1",
    ]
    csv_file = path / "card_transaction.v1.csv"
    with open(csv_file, "w") as f:
        f.write(_TABFORMER_HEADER + "\n")
        for row in rows:
            f.write(row + "\n")
    return csv_file


@pytest.fixture()
def minimal_csv(tmp_path: Path) -> Path:
    """Return a path to a minimal valid TabFormer CSV in a temp directory."""
    return _write_minimal_csv(tmp_path)


@pytest.fixture()
def adapter_with_csv(minimal_csv: Path) -> IBMTabFormerAdapter:
    """Return an IBMTabFormerAdapter pointing at the minimal fixture CSV."""
    return IBMTabFormerAdapter(
        csv_path=minimal_csv,
        vocab_path=minimal_csv.parent / "vocab.pkl",
    )


# ---------------------------------------------------------------------------
# Contract tests — Protocol compliance and registry
# ---------------------------------------------------------------------------

class TestContract:
    """Verify IBMTabFormerAdapter satisfies DatasetAdapterProtocol."""

    def test_adapter_implements_protocol(self, tmp_path: Path) -> None:
        """ADR 003: IBMTabFormerAdapter must satisfy DatasetAdapterProtocol.

        Protocol compliance is structural (duck-typing). The adapter must
        expose dataset_name and prepare() with the correct signatures.
        """
        adapter = IBMTabFormerAdapter(csv_path=tmp_path / "data.csv")
        assert isinstance(adapter, DatasetAdapterProtocol), (
            "IBMTabFormerAdapter does not satisfy DatasetAdapterProtocol — "
            "dataset_name or prepare() is missing or has wrong signature."
        )

    def test_dataset_name(self, tmp_path: Path) -> None:
        """ADR 003: IBMTabFormerAdapter.dataset_name must be 'ibm-tabformer'.

        The registry key is the canonical identifier shared between the adapter,
        the manifest, and the train_pragma() API.
        """
        adapter = IBMTabFormerAdapter(csv_path=tmp_path / "data.csv")
        assert adapter.dataset_name == "ibm-tabformer", (
            f"dataset_name must be 'ibm-tabformer', got '{adapter.dataset_name}'"
        )

    def test_adapter_registered_in_registry(self) -> None:
        """ADR 003: get_adapter('ibm-tabformer') must return IBMTabFormerAdapter."""
        adapter_cls = get_adapter("ibm-tabformer")
        assert adapter_cls is IBMTabFormerAdapter, (
            f"Registry must map 'ibm-tabformer' → IBMTabFormerAdapter, "
            f"got {adapter_cls}"
        )

    def test_get_adapter_raises_for_unknown_name(self) -> None:
        """ADR 003: get_adapter() must raise KeyError for unregistered dataset names."""
        with pytest.raises(KeyError, match="not-a-real-dataset"):
            get_adapter("not-a-real-dataset")


# ---------------------------------------------------------------------------
# Property tests — §2.4 adapter contract invariants
# ---------------------------------------------------------------------------

class TestProperties:
    """Verify prepare() contract: early validation, no S3 on upload=False."""

    def test_raises_file_not_found_on_missing_csv(self, tmp_path: Path) -> None:
        """ADR 003: prepare() must raise FileNotFoundError if CSV does not exist.

        Validation must happen before any fitting or S3 access. This prevents
        a training pod from starting, running for minutes, then discovering
        the data is missing.
        """
        adapter = IBMTabFormerAdapter(
            csv_path=tmp_path / "does_not_exist.csv",
        )
        with pytest.raises(FileNotFoundError):
            adapter.prepare(PRAGMAConfig.pragma_s(), upload=False)

    def test_prepare_returns_dataset_manifest(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """ADR 003: prepare(upload=False) must return a DatasetManifest."""
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert isinstance(manifest, DatasetManifest), (
            f"prepare() must return a DatasetManifest, got {type(manifest)}"
        )

    def test_prepare_returns_manifest_satisfying_protocol(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """ADR 003: the returned manifest must satisfy DatasetManifestProtocol."""
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert isinstance(manifest, DatasetManifestProtocol)

    def test_prepare_upload_false_does_not_raise_on_missing_s3_env(
        self, adapter_with_csv: IBMTabFormerAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR 003: upload=False must not attempt S3 access.

        Unset all S3 env vars (native AWS_* and legacy MODEL_REGISTRY_*).
        If prepare(upload=False) tries to connect to S3, it will fail with a
        missing-credentials error. With upload=False, it must succeed silently.
        """
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("MODEL_REGISTRY_BUCKET", raising=False)
        monkeypatch.delenv("MODEL_REGISTRY_ENDPOINT", raising=False)
        monkeypatch.delenv("MODEL_REGISTRY_ACCESS_KEY", raising=False)
        monkeypatch.delenv("MODEL_REGISTRY_SECRET_KEY", raising=False)
        # Must not raise — no S3 credentials needed when upload=False
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert manifest is not None

    def test_manifest_config_matches_passed_config(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """ADR 003: manifest.config must be the exact config passed to prepare()."""
        config = PRAGMAConfig.pragma_s()
        manifest = adapter_with_csv.prepare(config, upload=False)
        assert manifest.config is config, (
            "manifest.config must be the PRAGMAConfig passed to prepare(), "
            "not a copy or a different instance."
        )

    def test_manifest_has_at_least_one_shard(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """ADR 003: the returned manifest must have at least one prepared shard."""
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert len(manifest.shards) >= 1, (
            "IBMTabFormerAdapter must produce a manifest with at least one shard"
        )

    def test_manifest_dataset_name_matches_adapter(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """ADR 003: manifest.dataset_name must match adapter.dataset_name."""
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert manifest.dataset_name == adapter_with_csv.dataset_name


# ---------------------------------------------------------------------------
# Specification tests — exact §2.4 values from docs/paper/key-numbers.md
# ---------------------------------------------------------------------------

class TestPaperSpecifications:
    """Verify the IBM TabFormer manifest carries exact §2.4 truncation limits."""

    def test_manifest_has_csv_shard(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """§2.4 / ADR 003: IBM TabFormer adapter produces a CSV shard."""
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert "csv" in manifest.formats, (
            "IBMTabFormerAdapter must produce a manifest with at least one CSV shard"
        )

    def test_manifest_has_vocab_uri(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """ADR 003: IBM TabFormer manifest must have a non-None vocab_uri.

        The fitted FinancialTokenizerPipeline is required for training.
        It must be referenced in the manifest so training can download it.
        """
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert manifest.vocab_uri is not None, (
            "IBM TabFormer manifest must set vocab_uri — "
            "the fitted tokeniser is required for training."
        )

    def test_manifest_max_event_tokens(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """§2.4 / key-numbers.md: manifest carries max_event_tokens = 24."""
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert manifest.config.max_event_tokens == 24  # key-numbers.md: §2.4

    def test_manifest_max_profile_tokens(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """§2.4 / key-numbers.md: manifest carries max_profile_tokens = 200."""
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert manifest.config.max_profile_tokens == 200  # key-numbers.md: §2.4

    def test_manifest_max_events(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """§2.4 / key-numbers.md: manifest carries max_events = 6500."""
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert manifest.config.max_events == 6_500  # key-numbers.md: §2.4

    def test_manifest_has_source_metadata(
        self, adapter_with_csv: IBMTabFormerAdapter
    ) -> None:
        """ADR 003: manifest source dict must identify the IBM TabFormer origin."""
        manifest = adapter_with_csv.prepare(PRAGMAConfig.pragma_s(), upload=False)
        assert "origin" in manifest.source, (
            "manifest.source must include 'origin' key"
        )
        assert "ibm-tabformer" in manifest.source["origin"].lower(), (
            "manifest.source['origin'] must identify 'ibm-tabformer'"
        )
