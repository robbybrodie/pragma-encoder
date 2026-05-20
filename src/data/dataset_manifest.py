"""DatasetManifest — generic prepared-dataset reference for PRAGMA training.

Implements the data storage contract described in Section 2.4 of the paper.
A DatasetManifest describes a *prepared* dataset — data that has been
tokeniser-fitted, validated, and uploaded to S3-compatible object storage.
Training consumes only the manifest; it never knows the original source format.

Design (ADR 003):
- Format-agnostic: shards may be CSV, Parquet, Arrow, or any future format.
- IBM TabFormer uses CSV shards today. Future bank datasets may use Parquet.
- vocab_uri and schema_uri are optional: not every dataset requires them.
- S3-compatible storage is the canonical store; local paths are dev/test only.
- config carries §2.4 truncation limits so they travel with the dataset reference.
- Immutable once created — training must not modify the manifest.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
ADR: docs/decisions/003-workbench-training-api.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from src.model.config import PRAGMAConfig

# ---------------------------------------------------------------------------
# DatasetShard — one prepared data file within a manifest
# ---------------------------------------------------------------------------

@runtime_checkable
class DatasetShardProtocol(Protocol):
    """Interface contract for a single data shard within a DatasetManifest.

    A shard is one prepared file (CSV, Parquet, Arrow, …) in S3.
    A manifest may contain one or many shards.
    """

    uri: str              # S3 path or URI to the shard file (non-empty)
    format: str           # file format: "csv", "parquet", "arrow", etc.
    rows: int | None      # row count for this shard, or None if unknown
    sha256: str | None    # hex SHA-256 checksum for integrity, or None


@dataclass(frozen=True)
class DatasetShard:
    """One prepared data shard: a single file (CSV, Parquet, Arrow…) in S3.

    Args:
        uri:    S3 path to the shard file. Non-empty. No specific prefix enforced
                here — the adapter sets the correct project prefix.
        format: File format string. Canonical values: "csv", "parquet", "arrow".
                Adapters may use other values for future formats.
        rows:   Number of data rows in this shard. None if not yet counted.
        sha256: Hex SHA-256 checksum for integrity verification. None if not computed.

    Reference: Ostroukhov et al. (2026), Section 2.4 (event shards / Parquet files)
    """

    uri: str
    format: str
    rows: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("DatasetShard.uri must not be empty")
        if not self.format:
            raise ValueError("DatasetShard.format must not be empty")
        if self.rows is not None and self.rows < 0:
            raise ValueError(f"DatasetShard.rows must be >= 0, got {self.rows}")


# ---------------------------------------------------------------------------
# DatasetManifestProtocol — the interface contract
# ---------------------------------------------------------------------------

@runtime_checkable
class DatasetManifestProtocol(Protocol):
    """Interface contract for a prepared PRAGMA dataset manifest.

    A manifest is a lightweight, immutable reference to a dataset that has
    been prepared (tokeniser fitted, data staged to S3). Training consumes
    only the manifest — it never needs to know the original data source or
    format.

    Section 2.4: data must survive pod restarts via S3 (not emptyDir or PVC).
    All shard URIs point to S3-compatible object storage.
    """

    dataset_name: str           # registry key, e.g. "ibm-tabformer"
    dataset_version: str        # version tag, e.g. "v1"
    prepared_prefix_uri: str    # S3 prefix where all prepared artifacts live
    shards: Sequence[DatasetShard]  # one or more prepared data shards
    vocab_uri: str | None       # S3 URI for fitted tokeniser vocab — optional
    schema_uri: str | None      # S3 URI for field schema definition — optional
    manifest_uri: str | None    # S3 URI for the serialised manifest JSON — optional
    row_count: int | None       # total rows across all shards — optional
    source: Mapping[str, str]   # provenance metadata dict (origin, format, …)
    config: PRAGMAConfig        # carries §2.4 truncation limits from PRAGMAConfig


# ---------------------------------------------------------------------------
# DatasetManifest — concrete implementation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetManifest:
    """Immutable reference to a prepared PRAGMA training dataset.

    Format-agnostic: shards may be CSV (IBM TabFormer today), Parquet
    (future bank data), Arrow, or any other format. The manifest contract
    is the same regardless of shard format.

    Args:
        dataset_name:        Registry key, e.g. "ibm-tabformer".
        dataset_version:     Version string, e.g. "v1".
        prepared_prefix_uri: S3 prefix under which all prepared artifacts live.
                             e.g. "pragma-encoder/data/tabformer/"
        shards:              One or more DatasetShard objects. Must be non-empty.
        vocab_uri:           S3 URI to fitted tokeniser vocab (vocab.pkl). Optional.
        schema_uri:          S3 URI to field schema definition. Optional.
        manifest_uri:        S3 URI where this manifest is serialised. Optional.
        row_count:           Total rows across all shards. Optional (None = unknown).
        source:              Provenance metadata. Required keys are adapter-defined.
                             Example: {"origin": "ibm-tabformer", "format": "csv"}
        config:              PRAGMAConfig carrying §2.4 truncation limits:
                             max_event_tokens=24, max_profile_tokens=200,
                             max_events=6500.

    Reference: Ostroukhov et al. (2026), Section 2.4
    ADR: docs/decisions/003-workbench-training-api.md
    """

    dataset_name: str
    dataset_version: str
    prepared_prefix_uri: str
    shards: tuple[DatasetShard, ...]      # tuple for hashability (frozen dataclass)
    vocab_uri: str | None
    schema_uri: str | None
    manifest_uri: str | None
    row_count: int | None
    source: dict[str, str]               # dict (Mapping) of provenance metadata
    config: PRAGMAConfig

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty")
        if not self.dataset_version:
            raise ValueError("dataset_version must not be empty")
        if not self.prepared_prefix_uri:
            raise ValueError("prepared_prefix_uri must not be empty")
        if not self.shards:
            raise ValueError(
                "DatasetManifest must contain at least one shard. "
                "Training cannot proceed without data."
            )
        if self.row_count is not None and self.row_count < 0:
            raise ValueError(f"row_count must be >= 0, got {self.row_count}")

    @property
    def formats(self) -> set[str]:
        """Return the set of shard formats present in this manifest."""
        return {shard.format for shard in self.shards}
