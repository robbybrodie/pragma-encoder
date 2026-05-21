"""Tests for DatasetManifest and DatasetShard.

Derived from PRAGMA paper Section 2.4:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

The manifest is format-agnostic. Tests prove the generic shard contract,
not IBM TabFormer CSV specifics. IBM TabFormer is tested only where the
registry key and adapter-produced manifest shape need anchoring.

Test types:
    Contract tests:  verify interface fields present and correctly typed
    Property tests:  verify §2.4 truncation limits, shard invariants, S3 model
    Spec tests:      verify exact values from docs/paper/key-numbers.md
"""

from __future__ import annotations

import pytest

from pragma_encoder.data.dataset_manifest import (
    DatasetManifest,
    DatasetManifestProtocol,
    DatasetShard,
    DatasetShardProtocol,
)
from pragma_encoder.model.config import PRAGMAConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _csv_shard() -> DatasetShard:
    return DatasetShard(
        uri="pragma-encoder/data/tabformer/card_transaction.v1.csv",
        format="csv",
        rows=24_000,
    )


def _parquet_shard() -> DatasetShard:
    return DatasetShard(
        uri="pragma-encoder/data/bank-txn/transactions.parquet",
        format="parquet",
        rows=500_000,
    )


def _make_manifest(
    shards: tuple[DatasetShard, ...] | None = None,
    config: PRAGMAConfig | None = None,
    vocab_uri: str | None = "pragma-encoder/data/tabformer/vocab.pkl",
    schema_uri: str | None = None,
    dataset_name: str = "ibm-tabformer",
) -> DatasetManifest:
    """Return a valid DatasetManifest for testing."""
    return DatasetManifest(
        dataset_name=dataset_name,
        dataset_version="v1",
        prepared_prefix_uri="pragma-encoder/data/tabformer/",
        shards=shards if shards is not None else (_csv_shard(),),
        vocab_uri=vocab_uri,
        schema_uri=schema_uri,
        manifest_uri=None,
        row_count=24_000,
        source={"origin": "ibm-tabformer", "format": "csv"},
        config=config if config is not None else PRAGMAConfig.pragma_s(),
    )


# ---------------------------------------------------------------------------
# Contract tests — equivalent to shape tests for non-tensor components
# ---------------------------------------------------------------------------

class TestContract:
    """Verify DatasetManifest and DatasetShard satisfy their Protocol interfaces."""

    def test_manifest_implements_protocol(self) -> None:
        """§2.4 / ADR 003: DatasetManifest must satisfy DatasetManifestProtocol."""
        manifest = _make_manifest()
        assert isinstance(manifest, DatasetManifestProtocol), (
            "DatasetManifest does not satisfy DatasetManifestProtocol — "
            "one or more required fields is missing."
        )

    def test_shard_implements_protocol(self) -> None:
        """§2.4 / ADR 003: DatasetShard must satisfy DatasetShardProtocol."""
        shard = _csv_shard()
        assert isinstance(shard, DatasetShardProtocol), (
            "DatasetShard does not satisfy DatasetShardProtocol."
        )

    def test_manifest_has_at_least_one_shard(self) -> None:
        """§2.4 / ADR 003: manifest must reference at least one prepared data shard.

        Training cannot proceed without data. An empty manifest is an error state,
        not a valid deferred-data manifest.
        """
        manifest = _make_manifest()
        assert len(manifest.shards) >= 1, (
            "DatasetManifest must contain at least one shard"
        )

    def test_each_shard_has_uri_and_format(self) -> None:
        """§2.4 / ADR 003: every shard must have a non-empty uri and format."""
        manifest = _make_manifest()
        for shard in manifest.shards:
            assert shard.uri, f"Shard uri must be non-empty, got {shard.uri!r}"
            assert shard.format, f"Shard format must be non-empty, got {shard.format!r}"

    def test_manifest_is_immutable(self) -> None:
        """ADR 003: DatasetManifest is immutable once created."""
        manifest = _make_manifest()
        with pytest.raises((AttributeError, TypeError)):
            manifest.dataset_name = "something-else"  # type: ignore[misc]

    def test_shard_is_immutable(self) -> None:
        """ADR 003: DatasetShard is immutable once created."""
        shard = _csv_shard()
        with pytest.raises((AttributeError, TypeError)):
            shard.uri = "other/path"  # type: ignore[misc]

    def test_manifest_has_source_metadata(self) -> None:
        """ADR 003: manifest must carry provenance metadata in source dict."""
        manifest = _make_manifest()
        assert isinstance(manifest.source, dict), "source must be a dict"
        assert len(manifest.source) >= 1, "source must be non-empty"

    def test_vocab_uri_is_optional(self) -> None:
        """ADR 003: vocab_uri is optional — not every dataset has one."""
        manifest_with = _make_manifest(vocab_uri="pragma-encoder/data/tabformer/vocab.pkl")
        manifest_without = _make_manifest(vocab_uri=None)
        assert manifest_with.vocab_uri is not None
        assert manifest_without.vocab_uri is None

    def test_schema_uri_is_optional(self) -> None:
        """ADR 003: schema_uri is optional — not every dataset has one."""
        manifest_with = _make_manifest(schema_uri="pragma-encoder/data/bank-txn/schema.json")
        manifest_without = _make_manifest(schema_uri=None)
        assert manifest_with.schema_uri is not None
        assert manifest_without.schema_uri is None

    def test_row_count_is_optional(self) -> None:
        """ADR 003: row_count may be None if not yet computed."""
        manifest = DatasetManifest(
            dataset_name="ibm-tabformer",
            dataset_version="v1",
            prepared_prefix_uri="pragma-encoder/data/tabformer/",
            shards=(_csv_shard(),),
            vocab_uri=None,
            schema_uri=None,
            manifest_uri=None,
            row_count=None,   # explicitly unknown
            source={"origin": "ibm-tabformer"},
            config=PRAGMAConfig.pragma_s(),
        )
        assert manifest.row_count is None


# ---------------------------------------------------------------------------
# Property tests — format-agnostic shard and §2.4 S3 storage model
# ---------------------------------------------------------------------------

class TestProperties:
    """Verify manifest invariants and §2.4 truncation limits."""

    def test_manifest_can_describe_csv_shards(self) -> None:
        """§2.4 / ADR 003: manifest describes CSV shards (IBM TabFormer today)."""
        manifest = _make_manifest(shards=(_csv_shard(),))
        assert "csv" in manifest.formats, (
            "Manifest with a CSV shard must report 'csv' in formats"
        )

    def test_manifest_can_describe_parquet_shards(self) -> None:
        """§2.4 / ADR 003: manifest describes Parquet shards (future bank data)."""
        manifest = _make_manifest(shards=(_parquet_shard(),))
        assert "parquet" in manifest.formats, (
            "Manifest with a Parquet shard must report 'parquet' in formats"
        )

    def test_manifest_can_have_multiple_shards_of_mixed_format(self) -> None:
        """ADR 003: manifest supports multiple shards, potentially mixed format."""
        manifest = _make_manifest(shards=(_csv_shard(), _parquet_shard()))
        assert len(manifest.shards) == 2
        assert "csv" in manifest.formats
        assert "parquet" in manifest.formats

    def test_max_event_tokens_from_config(self) -> None:
        """§2.4 / key-numbers.md: max_event_tokens = 24 travels with manifest."""
        manifest = _make_manifest()
        assert manifest.config.max_event_tokens == 24, (  # key-numbers.md: §2.4
            f"§2.4 truncation limit max_event_tokens must be 24, "
            f"got {manifest.config.max_event_tokens}"
        )

    def test_max_profile_tokens_from_config(self) -> None:
        """§2.4 / key-numbers.md: max_profile_tokens = 200 travels with manifest."""
        manifest = _make_manifest()
        assert manifest.config.max_profile_tokens == 200, (  # key-numbers.md: §2.4
            f"§2.4 truncation limit max_profile_tokens must be 200, "
            f"got {manifest.config.max_profile_tokens}"
        )

    def test_max_events_from_config(self) -> None:
        """§2.4 / key-numbers.md: max_events = 6500 travels with manifest."""
        manifest = _make_manifest()
        assert manifest.config.max_events == 6_500, (  # key-numbers.md: §2.4
            f"§2.4 truncation limit max_events must be 6500, "
            f"got {manifest.config.max_events}"
        )

    def test_rejects_empty_shards(self) -> None:
        """ADR 003: manifest with zero shards must raise ValueError."""
        with pytest.raises(ValueError, match="at least one shard"):
            DatasetManifest(
                dataset_name="ibm-tabformer",
                dataset_version="v1",
                prepared_prefix_uri="pragma-encoder/data/tabformer/",
                shards=(),  # empty — must be rejected
                vocab_uri=None,
                schema_uri=None,
                manifest_uri=None,
                row_count=None,
                source={},
                config=PRAGMAConfig.pragma_s(),
            )

    def test_rejects_shard_with_empty_uri(self) -> None:
        """ADR 003: a shard with an empty URI must raise ValueError."""
        with pytest.raises(ValueError, match="uri"):
            DatasetShard(uri="", format="csv")

    def test_rejects_shard_with_empty_format(self) -> None:
        """ADR 003: a shard with an empty format string must raise ValueError."""
        with pytest.raises(ValueError, match="format"):
            DatasetShard(uri="pragma-encoder/data/tabformer/data.csv", format="")

    def test_rejects_negative_row_count(self) -> None:
        """ADR 003: row_count must be >= 0 if provided."""
        with pytest.raises(ValueError, match="row_count"):
            DatasetManifest(
                dataset_name="ibm-tabformer",
                dataset_version="v1",
                prepared_prefix_uri="pragma-encoder/data/tabformer/",
                shards=(_csv_shard(),),
                vocab_uri=None,
                schema_uri=None,
                manifest_uri=None,
                row_count=-1,  # must be rejected
                source={},
                config=PRAGMAConfig.pragma_s(),
            )

    def test_shard_rows_optional(self) -> None:
        """ADR 003: shard row count may be None if not yet counted."""
        shard = DatasetShard(
            uri="pragma-encoder/data/bank-txn/transactions.parquet",
            format="parquet",
            rows=None,  # not yet counted — valid
        )
        assert shard.rows is None

    def test_config_is_single_source_of_truncation_limits(self) -> None:
        """§2.4 / ADR 003: no truncation values are hardcoded in DatasetManifest.

        Truncation limits come from config only. Switching from PRAGMA-S to
        PRAGMA-M requires changing config — nothing else in the manifest changes.
        """
        config_s = PRAGMAConfig.pragma_s()
        config_m = PRAGMAConfig.pragma_m()
        manifest_s = _make_manifest(config=config_s)
        manifest_m = _make_manifest(config=config_m)
        # Both carry the same §2.4 truncation limits (data-driven, not model-size)
        assert manifest_s.config.max_event_tokens == manifest_m.config.max_event_tokens
        assert manifest_s.config.max_profile_tokens == manifest_m.config.max_profile_tokens
        assert manifest_s.config.max_events == manifest_m.config.max_events


# ---------------------------------------------------------------------------
# Specification tests — exact values from docs/paper/key-numbers.md
# ---------------------------------------------------------------------------

class TestPaperSpecifications:
    """Verify manifest carries exact §2.4 truncation limits for all model variants."""

    def test_pragma_s_truncation_limits(self) -> None:
        """§2.4 / key-numbers.md: PRAGMA-S — all three truncation limits correct."""
        manifest = _make_manifest(config=PRAGMAConfig.pragma_s())
        assert manifest.config.max_event_tokens == 24    # key-numbers.md: §2.4
        assert manifest.config.max_profile_tokens == 200  # key-numbers.md: §2.4
        assert manifest.config.max_events == 6_500        # key-numbers.md: §2.4

    def test_pragma_m_truncation_limits(self) -> None:
        """§2.4 / key-numbers.md: PRAGMA-M — truncation limits identical to PRAGMA-S.

        Truncation limits are fixed by the data distribution (§2.4 Table),
        not by model size. All three variants use the same values.
        """
        manifest = _make_manifest(config=PRAGMAConfig.pragma_m())
        assert manifest.config.max_event_tokens == 24    # key-numbers.md: §2.4
        assert manifest.config.max_profile_tokens == 200  # key-numbers.md: §2.4
        assert manifest.config.max_events == 6_500        # key-numbers.md: §2.4

    def test_ibm_tabformer_registry_key(self) -> None:
        """ADR 003: IBM TabFormer dataset is registered as 'ibm-tabformer'.

        This is the canonical registry key used in S3 paths and examples.
        The adapter and manifest must agree on this key.
        """
        manifest = _make_manifest(dataset_name="ibm-tabformer")
        assert manifest.dataset_name == "ibm-tabformer"

    def test_ibm_tabformer_uses_csv_shards(self) -> None:
        """§2.4 / ADR 003: IBM TabFormer adapter produces CSV shards."""
        manifest = _make_manifest(shards=(_csv_shard(),))
        assert "csv" in manifest.formats, (
            "IBM TabFormer manifest must contain at least one CSV shard"
        )

    def test_future_bank_dataset_uses_parquet_shards(self) -> None:
        """§2.4 / ADR 003: future bank dataset adapters will produce Parquet shards.

        This test validates that the manifest contract does not assume CSV.
        A Parquet-only manifest must pass all the same validations.
        """
        bank_manifest = DatasetManifest(
            dataset_name="bank-txn",
            dataset_version="v1",
            prepared_prefix_uri="pragma-encoder/data/bank-txn/",
            shards=(_parquet_shard(),),
            vocab_uri="pragma-encoder/data/bank-txn/vocab.pkl",
            schema_uri="pragma-encoder/data/bank-txn/schema.json",
            manifest_uri=None,
            row_count=500_000,
            source={"origin": "bank-internal", "format": "parquet"},
            config=PRAGMAConfig.pragma_s(),
        )
        assert isinstance(bank_manifest, DatasetManifestProtocol)
        assert "parquet" in bank_manifest.formats
        assert bank_manifest.config.max_event_tokens == 24  # key-numbers.md: §2.4
