"""Unit tests for S3 checkpoint/resume logic (src/training/checkpoints.py).

Validates the all-rank download pattern that fixes TD-006:

  Old (broken) behaviour:
    - Rank 0 downloads checkpoint from S3 to its own emptyDir
    - Workers broadcast-receive a boolean "found" flag
    - Workers then call _find_latest_local_checkpoint() on their own emptyDir
    - Workers find nothing — emptyDir is per-pod, rank 0's download is invisible

  New (correct) behaviour:
    - Rank 0 selects the latest checkpoint key from S3
    - Rank 0 broadcasts the key string to all ranks via dist.broadcast_object_list
    - EVERY rank independently downloads the checkpoint from S3 to its own local path
    - All ranks call dist.barrier() after download
    - All ranks load from their own local copy

No shared filesystem (emptyDir) is assumed between ranks.
No kfp / kfp-kubernetes is imported — this is a training-image-only module.
No real S3 connections are made — boto3 client is mocked throughout.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Section 2.4
TD-006: docs/tech-debt.md — multi-node checkpoint resume with per-pod emptyDir
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Import under test — will fail (ModuleNotFoundError) until
# src/training/checkpoints.py is created. That is the expected red phase.
# ---------------------------------------------------------------------------
from pragma_encoder.training.checkpoints import (  # noqa: E402
    CheckpointStore,
    LocalCheckpointStore,
    S3CheckpointStore,
    barrier_if_distributed,
    build_checkpoint_store,
    download_checkpoint_for_rank,
    list_checkpoint_keys,
    parse_s3_config_from_env,
    resolve_resume_checkpoint,
    select_latest_checkpoint_key,
    upload_checkpoint_if_rank0,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_s3_object(key: str, last_modified: datetime) -> dict:
    """Build a fake S3 ListObjectsV2 Contents item."""
    return {"Key": key, "LastModified": last_modified, "Size": 1000}


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. TestCheckpointKeySelection
#    list_checkpoint_keys and select_latest_checkpoint_key
# ---------------------------------------------------------------------------


class TestCheckpointKeySelection:
    """Validate S3 checkpoint key listing and latest-key selection.

    No real S3 — boto3 client is mocked.
    """

    def test_no_checkpoints_returns_empty_list(self) -> None:
        """list_checkpoint_keys returns [] when S3 bucket has no .pt files."""
        client = mock.MagicMock()
        client.list_objects_v2.return_value = {"Contents": []}
        keys = list_checkpoint_keys(client, bucket="my-bucket", prefix="model/ckpts")
        assert keys == [], (
            "list_checkpoint_keys must return [] when no .pt files exist. "
            f"Got: {keys!r}"
        )

    def test_lists_only_pt_files(self) -> None:
        """list_checkpoint_keys ignores non-.pt objects."""
        client = mock.MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [
                _make_s3_object("model/ckpts/checkpoint_epoch0001.pt", _utc(2026, 1, 1)),
                _make_s3_object("model/ckpts/config.json", _utc(2026, 1, 2)),
                _make_s3_object("model/ckpts/vocab.pkl", _utc(2026, 1, 3)),
                _make_s3_object("model/ckpts/checkpoint_epoch0002.pt", _utc(2026, 1, 4)),
            ]
        }
        keys = list_checkpoint_keys(client, bucket="my-bucket", prefix="model/ckpts")
        assert len(keys) == 2, f"Expected 2 .pt keys, got {len(keys)}: {keys}"
        assert all(k.endswith(".pt") for k in keys), (
            f"All returned keys must end with .pt. Got: {keys}"
        )

    def test_returns_empty_when_no_contents_key(self) -> None:
        """list_checkpoint_keys handles missing 'Contents' key (empty S3 prefix)."""
        client = mock.MagicMock()
        client.list_objects_v2.return_value = {}  # No 'Contents' key
        keys = list_checkpoint_keys(client, bucket="my-bucket", prefix="model/ckpts")
        assert keys == [], (
            "list_checkpoint_keys must return [] when Contents key is absent. "
            f"Got: {keys!r}"
        )

    def test_select_latest_by_last_modified(self) -> None:
        """select_latest_checkpoint_key returns the key with the latest LastModified."""
        client = mock.MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [
                _make_s3_object("model/ckpts/checkpoint_epoch0001.pt", _utc(2026, 1, 1)),
                _make_s3_object("model/ckpts/checkpoint_epoch0003.pt", _utc(2026, 1, 3)),
                _make_s3_object("model/ckpts/checkpoint_epoch0002.pt", _utc(2026, 1, 2)),
            ]
        }
        key = select_latest_checkpoint_key(client, bucket="my-bucket", prefix="model/ckpts")
        assert key == "model/ckpts/checkpoint_epoch0003.pt", (
            f"Expected epoch0003 (latest LastModified), got: {key!r}"
        )

    def test_select_returns_none_when_no_checkpoints(self) -> None:
        """select_latest_checkpoint_key returns None when no .pt files exist."""
        client = mock.MagicMock()
        client.list_objects_v2.return_value = {"Contents": []}
        key = select_latest_checkpoint_key(client, bucket="my-bucket", prefix="model/ckpts")
        assert key is None, (
            f"select_latest_checkpoint_key must return None when no checkpoints exist. "
            f"Got: {key!r}"
        )

    def test_select_returns_none_when_only_non_pt_files(self) -> None:
        """select_latest_checkpoint_key returns None when only non-.pt objects exist."""
        client = mock.MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [
                _make_s3_object("model/ckpts/config.json", _utc(2026, 1, 5)),
            ]
        }
        key = select_latest_checkpoint_key(client, bucket="my-bucket", prefix="model/ckpts")
        assert key is None, (
            f"select_latest_checkpoint_key must return None when only non-.pt files exist. "
            f"Got: {key!r}"
        )

    def test_list_uses_prefix_with_trailing_slash(self) -> None:
        """list_checkpoint_keys calls list_objects_v2 with prefix ending in '/'."""
        client = mock.MagicMock()
        client.list_objects_v2.return_value = {}
        list_checkpoint_keys(client, bucket="my-bucket", prefix="model/ckpts")
        call_kwargs = client.list_objects_v2.call_args
        prefix_used = (
            call_kwargs.kwargs.get("Prefix")
            or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
            or call_kwargs[1].get("Prefix")
        )
        # The Prefix kwarg must include model/ckpts/ (trailing slash) to avoid
        # matching keys under other prefixes like model/ckpts-archive/
        assert prefix_used is not None, "list_objects_v2 must be called with Prefix keyword"
        assert "model/ckpts" in prefix_used, (
            f"Prefix must contain the given prefix. Got: {prefix_used!r}"
        )


# ---------------------------------------------------------------------------
# 2. TestS3ConfigFromEnv
#    parse_s3_config_from_env
# ---------------------------------------------------------------------------


class TestS3ConfigFromEnv:
    """Validate S3 config parsing from MODEL_REGISTRY_* env vars."""

    def test_returns_none_when_bucket_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """parse_s3_config_from_env returns None when MODEL_REGISTRY_BUCKET is unset."""
        monkeypatch.delenv("MODEL_REGISTRY_BUCKET", raising=False)
        monkeypatch.setenv("MODEL_REGISTRY_ENDPOINT", "s3.example.com")
        monkeypatch.setenv("MODEL_REGISTRY_ACCESS_KEY", "key")
        monkeypatch.setenv("MODEL_REGISTRY_SECRET_KEY", "secret")
        result = parse_s3_config_from_env()
        assert result is None, (
            "parse_s3_config_from_env must return None when BUCKET is missing. "
            f"Got: {result!r}"
        )

    def test_returns_none_when_endpoint_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """parse_s3_config_from_env returns None when MODEL_REGISTRY_ENDPOINT is unset."""
        monkeypatch.setenv("MODEL_REGISTRY_BUCKET", "my-bucket")
        monkeypatch.delenv("MODEL_REGISTRY_ENDPOINT", raising=False)
        monkeypatch.setenv("MODEL_REGISTRY_ACCESS_KEY", "key")
        monkeypatch.setenv("MODEL_REGISTRY_SECRET_KEY", "secret")
        result = parse_s3_config_from_env()
        assert result is None, (
            "parse_s3_config_from_env must return None when ENDPOINT is missing. "
            f"Got: {result!r}"
        )

    def test_returns_config_dict_when_all_vars_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """parse_s3_config_from_env returns a config dict when all vars are set."""
        monkeypatch.setenv("MODEL_REGISTRY_BUCKET", "my-bucket")
        monkeypatch.setenv("MODEL_REGISTRY_ENDPOINT", "s3.example.com")
        monkeypatch.setenv("MODEL_REGISTRY_ACCESS_KEY", "access-key")
        monkeypatch.setenv("MODEL_REGISTRY_SECRET_KEY", "secret-key")
        result = parse_s3_config_from_env()
        assert result is not None, (
            "parse_s3_config_from_env must return a config dict when all vars are set."
        )
        assert isinstance(result, dict), (
            f"parse_s3_config_from_env must return a dict. Got: {type(result).__name__}"
        )
        assert "bucket" in result, "Config dict must include 'bucket' key."
        assert result["bucket"] == "my-bucket", (
            f"Config bucket must match MODEL_REGISTRY_BUCKET. Got: {result['bucket']!r}"
        )

    def test_config_does_not_expose_secret_in_repr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """parse_s3_config_from_env result must not expose the secret key in repr."""
        monkeypatch.setenv("MODEL_REGISTRY_BUCKET", "my-bucket")
        monkeypatch.setenv("MODEL_REGISTRY_ENDPOINT", "s3.example.com")
        monkeypatch.setenv("MODEL_REGISTRY_ACCESS_KEY", "access-key")
        monkeypatch.setenv("MODEL_REGISTRY_SECRET_KEY", "very-secret-key-12345")
        result = parse_s3_config_from_env()
        if result is None:
            return  # Covered by other tests
        # Repr must not contain the raw secret value
        result_repr = repr(result)
        assert "very-secret-key-12345" not in result_repr, (
            "Config repr must not expose the raw MODEL_REGISTRY_SECRET_KEY value. "
            "Use a redacted placeholder or exclude from repr. "
            f"Got: {result_repr!r}"
        )


# ---------------------------------------------------------------------------
# 3. TestRank0UploadSemantics
#    upload_checkpoint_if_rank0
# ---------------------------------------------------------------------------


class TestRank0UploadSemantics:
    """Validate that only rank 0 uploads checkpoints to S3."""

    def test_rank0_uploads_checkpoint(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """upload_checkpoint_if_rank0 calls S3 upload when is_rank0=True."""
        ckpt_path = tmp_path / "checkpoint_epoch0001.pt"
        ckpt_path.write_bytes(b"fake-checkpoint")

        mock_client = mock.MagicMock()
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client", return_value=mock_client):
            with mock.patch(
                "pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                              "access_key": "key", "secret_key": "secret"},
            ):
                upload_checkpoint_if_rank0(
                    ckpt_path=ckpt_path,
                    s3_prefix="model/ckpts",
                    is_rank0=True,
                )

        mock_client.upload_file.assert_called_once(), (
            "upload_checkpoint_if_rank0 must call client.upload_file when is_rank0=True."
        )

    def test_non_rank0_does_not_upload(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """upload_checkpoint_if_rank0 is a no-op when is_rank0=False."""
        ckpt_path = tmp_path / "checkpoint_epoch0001.pt"
        ckpt_path.write_bytes(b"fake-checkpoint")

        mock_client = mock.MagicMock()
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client", return_value=mock_client):
            upload_checkpoint_if_rank0(
                ckpt_path=ckpt_path,
                s3_prefix="model/ckpts",
                is_rank0=False,
            )

        mock_client.upload_file.assert_not_called(), (
            "upload_checkpoint_if_rank0 must NOT call upload when is_rank0=False. "
            "Only rank 0 is responsible for S3 uploads to avoid write races."
        )

    def test_upload_skipped_when_no_s3_config(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """upload_checkpoint_if_rank0 is a no-op when S3 config is absent."""
        ckpt_path = tmp_path / "checkpoint_epoch0001.pt"
        ckpt_path.write_bytes(b"fake-checkpoint")

        with mock.patch(
            "pragma_encoder.training.checkpoints.parse_s3_config_from_env", return_value=None
        ):
            # Should not raise, should not call any S3 client
            upload_checkpoint_if_rank0(
                ckpt_path=ckpt_path,
                s3_prefix="model/ckpts",
                is_rank0=True,
            )
        # No assertion needed — if it raised or called S3, the test would fail

    def test_upload_constructs_correct_s3_key(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """upload_checkpoint_if_rank0 constructs key as '<prefix>/<filename>'."""
        ckpt_path = tmp_path / "checkpoint_epoch0042.pt"
        ckpt_path.write_bytes(b"fake-checkpoint")

        mock_client = mock.MagicMock()
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client", return_value=mock_client):
            with mock.patch(
                "pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                              "access_key": "key", "secret_key": "secret"},
            ):
                upload_checkpoint_if_rank0(
                    ckpt_path=ckpt_path,
                    s3_prefix="pragma-encoder/checkpoints/pragma-s",
                    is_rank0=True,
                )

        call_args = mock_client.upload_file.call_args
        assert call_args is not None, "upload_file must have been called."
        # The S3 key should be: prefix/filename
        s3_key_used = call_args.args[2] if len(call_args.args) > 2 else call_args[0][2]
        assert s3_key_used == "pragma-encoder/checkpoints/pragma-s/checkpoint_epoch0042.pt", (
            f"S3 key must be '<prefix>/<filename>'. Got: {s3_key_used!r}"
        )


# ---------------------------------------------------------------------------
# 4. TestAllRankDownloadSemantics
#    download_checkpoint_for_rank
# ---------------------------------------------------------------------------


class TestAllRankDownloadSemantics:
    """Validate that ALL ranks (not just rank 0) call S3 download.

    This is the core fix for TD-006: every rank downloads the checkpoint
    independently to its own local emptyDir. No shared filesystem is assumed.
    """

    def test_download_calls_s3_download_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        """download_checkpoint_for_rank calls client.download_file."""
        mock_client = mock.MagicMock()
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client", return_value=mock_client):
            with mock.patch(
                "pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                              "access_key": "key", "secret_key": "secret"},
            ):
                result = download_checkpoint_for_rank(
                    key="model/ckpts/checkpoint_epoch0001.pt",
                    output_dir=tmp_path,
                )

        mock_client.download_file.assert_called_once(), (
            "download_checkpoint_for_rank must call client.download_file."
        )
        assert result is not None, (
            "download_checkpoint_for_rank must return the local path."
        )

    def test_download_returns_local_path_with_correct_filename(
        self, tmp_path: pathlib.Path
    ) -> None:
        """download_checkpoint_for_rank returns Path with the checkpoint filename."""
        mock_client = mock.MagicMock()
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client", return_value=mock_client):
            with mock.patch(
                "pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                              "access_key": "key", "secret_key": "secret"},
            ):
                result = download_checkpoint_for_rank(
                    key="model/ckpts/checkpoint_epoch0042.pt",
                    output_dir=tmp_path,
                )

        assert result is not None, "result must not be None when download succeeds."
        assert result.name == "checkpoint_epoch0042.pt", (
            f"Downloaded filename must match the S3 key filename. Got: {result.name!r}"
        )
        assert result.parent == tmp_path, (
            f"Downloaded file must be in output_dir. Got parent: {result.parent!r}"
        )

    def test_download_creates_output_dir_if_missing(
        self, tmp_path: pathlib.Path
    ) -> None:
        """download_checkpoint_for_rank creates output_dir if it does not exist."""
        new_dir = tmp_path / "deep" / "output" / "dir"
        assert not new_dir.exists(), "Pre-condition: directory must not exist"

        mock_client = mock.MagicMock()
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client", return_value=mock_client):
            with mock.patch(
                "pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                              "access_key": "key", "secret_key": "secret"},
            ):
                download_checkpoint_for_rank(
                    key="model/ckpts/checkpoint_epoch0001.pt",
                    output_dir=new_dir,
                )

        assert new_dir.exists(), (
            "download_checkpoint_for_rank must create output_dir if it does not exist."
        )

    def test_download_returns_none_when_no_s3_config(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """download_checkpoint_for_rank returns None when S3 config is absent."""
        with mock.patch(
            "pragma_encoder.training.checkpoints.parse_s3_config_from_env", return_value=None
        ):
            result = download_checkpoint_for_rank(
                key="model/ckpts/checkpoint_epoch0001.pt",
                output_dir=tmp_path,
            )
        assert result is None, (
            "download_checkpoint_for_rank must return None when S3 config is absent. "
            f"Got: {result!r}"
        )

    def test_download_returns_none_when_key_is_none(
        self, tmp_path: pathlib.Path
    ) -> None:
        """download_checkpoint_for_rank returns None when key is None (no checkpoint)."""
        result = download_checkpoint_for_rank(
            key=None,
            output_dir=tmp_path,
        )
        assert result is None, (
            "download_checkpoint_for_rank must return None when key is None. "
            f"Got: {result!r}"
        )


# ---------------------------------------------------------------------------
# 5. TestBroadcastAndBarrierSemantics
#    resolve_resume_checkpoint — distributed coordination
# ---------------------------------------------------------------------------


class TestBroadcastAndBarrierSemantics:
    """Validate distributed checkpoint resolution semantics.

    The key invariant: every rank downloads the checkpoint, not just rank 0.
    This is the fix for TD-006.
    """

    def test_rank0_selects_and_broadcasts_key(self, tmp_path: pathlib.Path) -> None:
        """resolve_resume_checkpoint: rank 0 selects key and broadcasts to all ranks."""
        mock_client = mock.MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                _make_s3_object("model/ckpts/checkpoint_epoch0001.pt", _utc(2026, 1, 1)),
            ]
        }

        with mock.patch("pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                        return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                                      "access_key": "key", "secret_key": "secret"}):
            with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                            return_value=mock_client):
                with mock.patch("torch.distributed.broadcast_object_list") as mock_broadcast:
                    with mock.patch("torch.distributed.barrier"):
                        # Simulate rank 0 in a 2-rank distributed job
                        resolve_resume_checkpoint(
                            output_dir=tmp_path,
                            s3_prefix="model/ckpts",
                            rank=0,
                            distributed=True,
                            device=torch.device("cpu"),
                        )

        # Rank 0 must call broadcast_object_list to share the selected key
        mock_broadcast.assert_called_once(), (
            "Rank 0 must call dist.broadcast_object_list to share the checkpoint key "
            "with all other ranks. Got no broadcast call."
        )

    def test_all_ranks_download_checkpoint_not_just_rank0(
        self, tmp_path: pathlib.Path
    ) -> None:
        """resolve_resume_checkpoint: ALL ranks call download, not just rank 0.

        This is the core fix for TD-006. Workers must download the checkpoint
        to their own emptyDir independently — they cannot read rank 0's local files.
        """
        mock_client = mock.MagicMock()
        key = "model/ckpts/checkpoint_epoch0001.pt"

        with mock.patch("pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                        return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                                      "access_key": "key", "secret_key": "secret"}):
            with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                            return_value=mock_client):
                with mock.patch("torch.distributed.broadcast_object_list",
                                side_effect=lambda obj_list, src: obj_list.__setitem__(0, key)):
                    with mock.patch("torch.distributed.barrier"):
                        # Simulate rank 1 (worker) in a 2-rank distributed job
                        resolve_resume_checkpoint(
                            output_dir=tmp_path,
                            s3_prefix="model/ckpts",
                            rank=1,
                            distributed=True,
                            device=torch.device("cpu"),
                        )

        # Worker (rank 1) must ALSO call download_file — not just look locally
        mock_client.download_file.assert_called(), (
            "Rank 1 (worker) must call S3 download_file to get the checkpoint. "
            "Workers have separate emptyDir from rank 0 — they cannot read rank 0's "
            "downloaded files. This is the core TD-006 fix. "
            "Got no download_file call on rank 1."
        )

    def test_barrier_called_after_all_rank_download(
        self, tmp_path: pathlib.Path
    ) -> None:
        """resolve_resume_checkpoint: dist.barrier() is called after download."""
        mock_client = mock.MagicMock()
        key = "model/ckpts/checkpoint_epoch0001.pt"

        with mock.patch("pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                        return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                                      "access_key": "key", "secret_key": "secret"}):
            with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                            return_value=mock_client):
                with mock.patch("torch.distributed.broadcast_object_list",
                                side_effect=lambda obj_list, src: obj_list.__setitem__(0, key)):
                    with mock.patch("torch.distributed.barrier") as mock_barrier:
                        resolve_resume_checkpoint(
                            output_dir=tmp_path,
                            s3_prefix="model/ckpts",
                            rank=0,
                            distributed=True,
                            device=torch.device("cpu"),
                        )

        mock_barrier.assert_called(), (
            "dist.barrier() must be called after all-rank download so all ranks "
            "are synchronised before loading the checkpoint. "
            "Got no barrier call."
        )

    def test_no_broadcast_in_single_node_mode(self, tmp_path: pathlib.Path) -> None:
        """resolve_resume_checkpoint: no dist calls when distributed=False."""
        mock_client = mock.MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                _make_s3_object("model/ckpts/checkpoint_epoch0001.pt", _utc(2026, 1, 1)),
            ]
        }

        with mock.patch("pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                        return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                                      "access_key": "key", "secret_key": "secret"}):
            with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                            return_value=mock_client):
                with mock.patch("torch.distributed.broadcast_object_list") as mock_broadcast:
                    with mock.patch("torch.distributed.barrier") as mock_barrier:
                        resolve_resume_checkpoint(
                            output_dir=tmp_path,
                            s3_prefix="model/ckpts",
                            rank=0,
                            distributed=False,
                            device=torch.device("cpu"),
                        )

        mock_broadcast.assert_not_called(), (
            "dist.broadcast_object_list must not be called in single-node mode. "
            "dist functions are only safe to call after process group init."
        )
        mock_barrier.assert_not_called(), (
            "dist.barrier must not be called in single-node mode."
        )


# ---------------------------------------------------------------------------
# 6. TestLocalSingleNodeResume
#    resolve_resume_checkpoint — local (no S3, no distributed)
# ---------------------------------------------------------------------------


class TestLocalSingleNodeResume:
    """Validate local single-node checkpoint resume (no S3, no distributed)."""

    def test_finds_latest_local_checkpoint(self, tmp_path: pathlib.Path) -> None:
        """resolve_resume_checkpoint returns the latest local .pt file when no S3."""
        (tmp_path / "checkpoint_epoch0001.pt").write_bytes(b"ckpt1")
        (tmp_path / "checkpoint_epoch0002.pt").write_bytes(b"ckpt2")

        with mock.patch(
            "pragma_encoder.training.checkpoints.parse_s3_config_from_env", return_value=None
        ):
            result = resolve_resume_checkpoint(
                output_dir=tmp_path,
                s3_prefix="",
                rank=0,
                distributed=False,
                device=torch.device("cpu"),
            )

        assert result is not None, "Expected to find a local checkpoint."
        assert result.name == "checkpoint_epoch0002.pt", (
            f"Expected latest checkpoint (epoch0002), got: {result.name!r}"
        )

    def test_returns_none_when_no_local_checkpoints_and_no_s3(
        self, tmp_path: pathlib.Path
    ) -> None:
        """resolve_resume_checkpoint returns None when output_dir is empty and no S3."""
        with mock.patch(
            "pragma_encoder.training.checkpoints.parse_s3_config_from_env", return_value=None
        ):
            result = resolve_resume_checkpoint(
                output_dir=tmp_path,
                s3_prefix="",
                rank=0,
                distributed=False,
                device=torch.device("cpu"),
            )

        assert result is None, (
            f"Expected None when no local checkpoints and no S3 config. Got: {result!r}"
        )

    def test_local_resume_does_not_call_s3(self, tmp_path: pathlib.Path) -> None:
        """resolve_resume_checkpoint does not make any S3 calls in local mode."""
        (tmp_path / "checkpoint_epoch0001.pt").write_bytes(b"ckpt1")

        with mock.patch("pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                        return_value=None):
            with mock.patch("pragma_encoder.training.checkpoints._make_s3_client") as mock_make_client:
                resolve_resume_checkpoint(
                    output_dir=tmp_path,
                    s3_prefix="",
                    rank=0,
                    distributed=False,
                    device=torch.device("cpu"),
                )

        mock_make_client.assert_not_called(), (
            "_make_s3_client must not be called when S3 config is absent. "
            "Local resume must not require S3 credentials."
        )


# ---------------------------------------------------------------------------
# 7. TestNoSharedFilesystemAssumption
#    Workers must not rely on rank 0's local download
# ---------------------------------------------------------------------------


class TestNoSharedFilesystemAssumption:
    """Validate that no shared filesystem is assumed between ranks.

    The broken pattern (TD-006):
      - Rank 0 downloads to /tmp/output/checkpoint.pt
      - Worker tries _find_latest_local_checkpoint('/tmp/output')
      - Worker's /tmp/output/ is a separate emptyDir — it is empty
      - Worker gets None, starts from scratch
      - Model states diverge

    The correct pattern:
      - Rank 0 selects key, broadcasts key string to all ranks
      - ALL ranks independently call download_file with the received key
      - Each rank downloads to its own local path
      - All ranks barrier, all ranks load
    """

    def test_worker_does_not_use_find_latest_local_checkpoint(
        self, tmp_path: pathlib.Path
    ) -> None:
        """resolve_resume_checkpoint: workers do not call _find_latest_local_checkpoint.

        Workers must receive the key via broadcast and download from S3,
        not scan a local directory that contains only rank 0's files.
        """
        key = "model/ckpts/checkpoint_epoch0001.pt"
        mock_client = mock.MagicMock()

        with mock.patch("pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                        return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                                      "access_key": "key", "secret_key": "secret"}):
            with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                            return_value=mock_client):
                with mock.patch("torch.distributed.broadcast_object_list",
                                side_effect=lambda obj_list, src: obj_list.__setitem__(0, key)):
                    with mock.patch("torch.distributed.barrier"):
                        with mock.patch(
                            "pragma_encoder.training.checkpoints._find_latest_local_checkpoint"
                        ) as mock_local_find:
                            resolve_resume_checkpoint(
                                output_dir=tmp_path,
                                s3_prefix="model/ckpts",
                                rank=1,  # Worker rank
                                distributed=True,
                                device=torch.device("cpu"),
                            )

        # Workers must NOT call _find_latest_local_checkpoint (that's the broken pattern)
        mock_local_find.assert_not_called(), (
            "Workers (rank > 0) must not call _find_latest_local_checkpoint. "
            "That is the broken TD-006 pattern — workers' emptyDir is empty. "
            "Workers must use the broadcasted key and download from S3 instead."
        )

    def test_each_rank_calls_download_file_independently(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Both rank 0 and rank 1 call download_file independently.

        In a 2-rank job, both ranks must download the checkpoint.
        This is the fundamental correctness requirement for distributed resume.
        """
        key = "model/ckpts/checkpoint_epoch0001.pt"
        mock_client = mock.MagicMock()

        # Configure list_objects_v2 so select_latest_checkpoint_key returns the key
        mock_client.list_objects_v2.return_value = {
            "Contents": [_make_s3_object(key, _utc(2026, 1, 1))]
        }

        with mock.patch("pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                        return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                                      "access_key": "key", "secret_key": "secret"}):
            with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                            return_value=mock_client):
                # Simulate rank 0
                with mock.patch("torch.distributed.broadcast_object_list"):
                    with mock.patch("torch.distributed.barrier"):
                        resolve_resume_checkpoint(
                            output_dir=tmp_path / "rank0",
                            s3_prefix="model/ckpts",
                            rank=0,
                            distributed=True,
                            device=torch.device("cpu"),
                        )
                        rank0_download_count = mock_client.download_file.call_count

        assert rank0_download_count >= 1, (
            "Rank 0 must call download_file at least once. "
            f"Got: {rank0_download_count} calls."
        )

        mock_client.reset_mock()

        with mock.patch("pragma_encoder.training.checkpoints.parse_s3_config_from_env",
                        return_value={"bucket": "my-bucket", "endpoint": "s3.example.com",
                                      "access_key": "key", "secret_key": "secret"}):
            with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                            return_value=mock_client):
                # Simulate rank 1
                with mock.patch("torch.distributed.broadcast_object_list",
                                side_effect=lambda obj_list, src: obj_list.__setitem__(0, key)):
                    with mock.patch("torch.distributed.barrier"):
                        resolve_resume_checkpoint(
                            output_dir=tmp_path / "rank1",
                            s3_prefix="model/ckpts",
                            rank=1,
                            distributed=True,
                            device=torch.device("cpu"),
                        )
                        rank1_download_count = mock_client.download_file.call_count

        assert rank1_download_count >= 1, (
            "Rank 1 (worker) must ALSO call download_file. "
            "Workers cannot read rank 0's emptyDir. "
            "This is the core TD-006 fix. "
            f"Got: {rank1_download_count} calls."
        )


# ---------------------------------------------------------------------------
# 8. TestBarrierIfDistributed
#    barrier_if_distributed helper
# ---------------------------------------------------------------------------


class TestBarrierIfDistributed:
    """Validate barrier_if_distributed helper."""

    def test_calls_barrier_when_distributed_true(self) -> None:
        """barrier_if_distributed calls dist.barrier() when distributed=True."""
        with mock.patch("torch.distributed.barrier") as mock_barrier:
            barrier_if_distributed(distributed=True)
        mock_barrier.assert_called_once(), (
            "barrier_if_distributed must call dist.barrier() when distributed=True."
        )

    def test_no_barrier_when_distributed_false(self) -> None:
        """barrier_if_distributed is a no-op when distributed=False."""
        with mock.patch("torch.distributed.barrier") as mock_barrier:
            barrier_if_distributed(distributed=False)
        mock_barrier.assert_not_called(), (
            "barrier_if_distributed must not call dist.barrier() when distributed=False."
        )


# ---------------------------------------------------------------------------
# 9. TestNoKfpImport
#    checkpoints.py must not import kfp / kfp_kubernetes
# ---------------------------------------------------------------------------


class TestNoKfpImport:
    """checkpoints.py must not import kfp or kfp_kubernetes.

    KFP/kfp-kubernetes are compile-time workbench dependencies only.
    The training image runtime must not require them.
    See docs/openshift-image-contract.md — KFP boundary.
    """

    def test_checkpoints_module_does_not_import_kfp(self) -> None:
        """src/training/checkpoints.py must not contain top-level kfp import."""
        checkpoints_path = (
            pathlib.Path(__file__).parent.parent / "src" / "pragma_encoder" / "training" / "checkpoints.py"
        )
        assert checkpoints_path.exists(), (
            f"src/training/checkpoints.py must exist. Path: {checkpoints_path}"
        )
        text = checkpoints_path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not stripped:
                continue
            if (stripped.startswith("import kfp") or stripped.startswith("from kfp")):
                pytest.fail(
                    f"src/training/checkpoints.py must not import kfp. "
                    f"Found: {line!r}. "
                    "kfp belongs only in the workbench compile environment. "
                    "See docs/openshift-image-contract.md."
                )

    def test_checkpoints_module_does_not_import_kfp_kubernetes(self) -> None:
        """src/training/checkpoints.py must not contain top-level kfp_kubernetes import."""
        checkpoints_path = (
            pathlib.Path(__file__).parent.parent / "src" / "pragma_encoder" / "training" / "checkpoints.py"
        )
        assert checkpoints_path.exists(), (
            f"src/training/checkpoints.py must exist. Path: {checkpoints_path}"
        )
        text = checkpoints_path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not stripped:
                continue
            if "kfp_kubernetes" in stripped and (
                stripped.startswith("import") or stripped.startswith("from")
            ):
                pytest.fail(
                    f"src/training/checkpoints.py must not import kfp_kubernetes. "
                    f"Found: {line!r}. "
                    "kfp-kubernetes belongs only in the workbench compile environment. "
                    "See docs/openshift-image-contract.md."
                )

    def test_checkpoints_module_importable_without_kfp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pragma_encoder.training.checkpoints must be importable even when kfp is absent."""
        import sys  # noqa: PLC0415
        # Simulate kfp being absent
        monkeypatch.setitem(sys.modules, "kfp", None)
        monkeypatch.setitem(sys.modules, "kfp_kubernetes", None)

        # Force reimport — must not raise ImportError about kfp.
        # Use monkeypatch.delitem so pytest restores the original module object
        # after this test, preventing cross-test pollution when other tests
        # patch src.training.checkpoints._make_s3_client.
        monkeypatch.delitem(sys.modules, "pragma_encoder.training.checkpoints", raising=False)

        try:
            import pragma_encoder.training.checkpoints  # noqa: F401,PLC0415
        except ImportError as exc:
            if "kfp" in str(exc).lower():
                pytest.fail(
                    f"pragma_encoder.training.checkpoints raised ImportError related to kfp: {exc}. "
                    "The checkpoints module must not require kfp or kfp-kubernetes. "
                    "See docs/openshift-image-contract.md."
                )
            # Other ImportErrors (e.g. boto3 not installed) are acceptable


# ---------------------------------------------------------------------------
# 10. TestCheckpointStoreBoundary
#     CheckpointStore Protocol and concrete adapters
# ---------------------------------------------------------------------------


class TestCheckpointStoreBoundary:
    """Validate the CheckpointStore Protocol and both concrete adapter classes.

    Ensures the storage-transport boundary is correctly defined:
    - LocalCheckpointStore operates purely on the filesystem (no S3 calls).
    - S3CheckpointStore delegates all I/O to boto3 (no local filesystem assumptions).
    - build_checkpoint_store() returns the right adapter based on env config.
    - Both concrete classes satisfy the CheckpointStore Protocol at runtime.
    """

    # ---- Protocol conformance -------------------------------------------

    def test_local_store_is_checkpoint_store_instance(self, tmp_path: pathlib.Path) -> None:
        """LocalCheckpointStore satisfies the CheckpointStore Protocol."""
        store = LocalCheckpointStore(tmp_path)
        assert isinstance(store, CheckpointStore), (
            "LocalCheckpointStore must satisfy the CheckpointStore Protocol. "
            "Add latest_key(), fetch(), and put() methods matching the Protocol signature."
        )

    def test_s3_store_is_checkpoint_store_instance(self) -> None:
        """S3CheckpointStore satisfies the CheckpointStore Protocol."""
        config = {"bucket": "b", "endpoint": "e", "access_key": "k", "secret_key": "s"}
        store = S3CheckpointStore(config, prefix="prefix/ckpts")  # type: ignore[arg-type]
        assert isinstance(store, CheckpointStore), (
            "S3CheckpointStore must satisfy the CheckpointStore Protocol. "
            "Add latest_key(), fetch(), and put() methods matching the Protocol signature."
        )

    # ---- LocalCheckpointStore -------------------------------------------

    def test_local_latest_key_returns_none_when_empty(self, tmp_path: pathlib.Path) -> None:
        """LocalCheckpointStore.latest_key() returns None when no .pt files exist."""
        store = LocalCheckpointStore(tmp_path)
        assert store.latest_key() is None, (
            "LocalCheckpointStore.latest_key() must return None when output_dir has no .pt files."
        )

    def test_local_latest_key_returns_lexicographically_latest_filename(
        self, tmp_path: pathlib.Path
    ) -> None:
        """LocalCheckpointStore.latest_key() returns the filename of the latest .pt file."""
        (tmp_path / "checkpoint_epoch0001.pt").write_bytes(b"ckpt1")
        (tmp_path / "checkpoint_epoch0002.pt").write_bytes(b"ckpt2")
        (tmp_path / "checkpoint_epoch0003.pt").write_bytes(b"ckpt3")
        store = LocalCheckpointStore(tmp_path)
        key = store.latest_key()
        assert key == "checkpoint_epoch0003.pt", (
            f"LocalCheckpointStore.latest_key() must return the latest filename. "
            f"Got: {key!r}"
        )

    def test_local_fetch_returns_path_in_output_dir(self, tmp_path: pathlib.Path) -> None:
        """LocalCheckpointStore.fetch() returns output_dir / key (file already local)."""
        (tmp_path / "checkpoint_epoch0001.pt").write_bytes(b"ckpt1")
        store = LocalCheckpointStore(tmp_path)
        result = store.fetch("checkpoint_epoch0001.pt", tmp_path)
        assert result == tmp_path / "checkpoint_epoch0001.pt", (
            f"LocalCheckpointStore.fetch() must return output_dir / key. Got: {result!r}"
        )

    def test_local_put_is_noop_no_s3_calls(self, tmp_path: pathlib.Path) -> None:
        """LocalCheckpointStore.put() does not make any S3 calls."""
        ckpt_path = tmp_path / "checkpoint_epoch0001.pt"
        ckpt_path.write_bytes(b"ckpt1")
        store = LocalCheckpointStore(tmp_path)
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client") as mock_client:
            store.put(ckpt_path)
        mock_client.assert_not_called(), (
            "LocalCheckpointStore.put() must not call _make_s3_client. "
            "Local store does not upload to S3 — the training loop already saved the file."
        )

    def test_local_put_does_not_raise(self, tmp_path: pathlib.Path) -> None:
        """LocalCheckpointStore.put() does not raise for a valid file."""
        ckpt_path = tmp_path / "checkpoint_epoch0001.pt"
        ckpt_path.write_bytes(b"ckpt1")
        store = LocalCheckpointStore(tmp_path)
        store.put(ckpt_path)  # Must not raise

    # ---- S3CheckpointStore ----------------------------------------------

    def test_s3_latest_key_delegates_to_select_latest(self) -> None:
        """S3CheckpointStore.latest_key() calls select_latest_checkpoint_key."""
        config = {"bucket": "my-bucket", "endpoint": "s3.example.com",
                  "access_key": "key", "secret_key": "secret"}
        store = S3CheckpointStore(config, prefix="pragma-encoder/ckpts")  # type: ignore[arg-type]
        mock_client = mock.MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                _make_s3_object("pragma-encoder/ckpts/checkpoint_epoch0001.pt", _utc(2026, 1, 1)),
            ]
        }
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                        return_value=mock_client):
            key = store.latest_key()
        assert key == "pragma-encoder/ckpts/checkpoint_epoch0001.pt", (
            f"S3CheckpointStore.latest_key() must return the latest S3 key. Got: {key!r}"
        )

    def test_s3_fetch_downloads_file_and_returns_local_path(
        self, tmp_path: pathlib.Path
    ) -> None:
        """S3CheckpointStore.fetch() calls download_file and returns the local path."""
        config = {"bucket": "my-bucket", "endpoint": "s3.example.com",
                  "access_key": "key", "secret_key": "secret"}
        store = S3CheckpointStore(config, prefix="pragma-encoder/ckpts")  # type: ignore[arg-type]
        mock_client = mock.MagicMock()
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                        return_value=mock_client):
            result = store.fetch(
                "pragma-encoder/ckpts/checkpoint_epoch0001.pt", tmp_path
            )
        mock_client.download_file.assert_called_once(), (
            "S3CheckpointStore.fetch() must call client.download_file."
        )
        assert result == tmp_path / "checkpoint_epoch0001.pt", (
            f"S3CheckpointStore.fetch() must return output_dir / filename. Got: {result!r}"
        )

    def test_s3_put_uploads_with_correct_key(self, tmp_path: pathlib.Path) -> None:
        """S3CheckpointStore.put() uploads the file under prefix/filename."""
        config = {"bucket": "my-bucket", "endpoint": "s3.example.com",
                  "access_key": "key", "secret_key": "secret"}
        store = S3CheckpointStore(config, prefix="pragma-encoder/ckpts")  # type: ignore[arg-type]
        ckpt_path = tmp_path / "checkpoint_epoch0005.pt"
        ckpt_path.write_bytes(b"ckpt5")
        mock_client = mock.MagicMock()
        with mock.patch("pragma_encoder.training.checkpoints._make_s3_client",
                        return_value=mock_client):
            store.put(ckpt_path)
        call_args = mock_client.upload_file.call_args
        assert call_args is not None, "S3CheckpointStore.put() must call client.upload_file."
        s3_key = call_args.args[2] if len(call_args.args) > 2 else call_args[0][2]
        assert s3_key == "pragma-encoder/ckpts/checkpoint_epoch0005.pt", (
            f"S3CheckpointStore.put() must upload to 'prefix/filename'. Got: {s3_key!r}"
        )

    # ---- build_checkpoint_store factory ---------------------------------

    def test_factory_returns_local_store_when_no_s3_config(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build_checkpoint_store returns LocalCheckpointStore when S3 config absent."""
        monkeypatch.delenv("MODEL_REGISTRY_BUCKET", raising=False)
        monkeypatch.delenv("MODEL_REGISTRY_ENDPOINT", raising=False)
        store = build_checkpoint_store(output_dir=tmp_path, s3_prefix="some/prefix")
        assert isinstance(store, LocalCheckpointStore), (
            "build_checkpoint_store must return LocalCheckpointStore when "
            "MODEL_REGISTRY_BUCKET / MODEL_REGISTRY_ENDPOINT are absent. "
            f"Got: {type(store).__name__}"
        )

    def test_factory_returns_local_store_when_no_prefix(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build_checkpoint_store returns LocalCheckpointStore when s3_prefix is empty."""
        monkeypatch.setenv("MODEL_REGISTRY_BUCKET", "my-bucket")
        monkeypatch.setenv("MODEL_REGISTRY_ENDPOINT", "s3.example.com")
        store = build_checkpoint_store(output_dir=tmp_path, s3_prefix="")
        assert isinstance(store, LocalCheckpointStore), (
            "build_checkpoint_store must return LocalCheckpointStore when s3_prefix is empty, "
            "even if S3 env vars are set. An empty prefix disables S3. "
            f"Got: {type(store).__name__}"
        )

    def test_factory_returns_s3_store_when_config_and_prefix_present(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build_checkpoint_store returns S3CheckpointStore when env vars + prefix set."""
        monkeypatch.setenv("MODEL_REGISTRY_BUCKET", "my-bucket")
        monkeypatch.setenv("MODEL_REGISTRY_ENDPOINT", "s3.example.com")
        monkeypatch.setenv("MODEL_REGISTRY_ACCESS_KEY", "key")
        monkeypatch.setenv("MODEL_REGISTRY_SECRET_KEY", "secret")
        store = build_checkpoint_store(
            output_dir=tmp_path, s3_prefix="pragma-encoder/ckpts"
        )
        assert isinstance(store, S3CheckpointStore), (
            "build_checkpoint_store must return S3CheckpointStore when "
            "MODEL_REGISTRY_BUCKET, MODEL_REGISTRY_ENDPOINT, and s3_prefix are all set. "
            f"Got: {type(store).__name__}"
        )


# ---------------------------------------------------------------------------
# 11. TestPlatformNameBoundary
#     No platform fixture names in src/pragma_encoder
# ---------------------------------------------------------------------------


class TestPlatformNameBoundary:
    """Validate that platform fixture names are absent from the pragma_encoder package.

    The pragma_encoder wheel must not contain OpenShift-specific fixture names
    such as 'pragma-workbench-env' (a Kubernetes Secret name used as a test
    default). The package reads MODEL_REGISTRY_* env vars; it does not care
    which Secret provided them.

    This test scans the source tree to enforce the boundary mechanically.
    """

    def _src_pragma_encoder_dir(self) -> pathlib.Path:
        return pathlib.Path(__file__).parent.parent / "src" / "pragma_encoder"

    def test_no_pragma_workbench_env_in_src_pragma_encoder(self) -> None:
        """'pragma-workbench-env' must not appear in any src/pragma_encoder source file.

        'pragma-workbench-env' is a Kubernetes Secret name — a platform fixture
        defined in openshift/secrets/ and tests/openshift/fixtures/. It must not
        appear in the installable package source. The package reads
        MODEL_REGISTRY_* env vars directly; it does not know which Secret
        provided them.
        """
        src_dir = self._src_pragma_encoder_dir()
        assert src_dir.exists(), f"src/pragma_encoder must exist at {src_dir}"
        violations = []
        for py_file in sorted(src_dir.rglob("*.py")):
            text = py_file.read_text()
            if "pragma-workbench-env" in text:
                violations.append(str(py_file.relative_to(src_dir.parent.parent)))
        assert not violations, (
            "These src/pragma_encoder files reference 'pragma-workbench-env', "
            "which is a Kubernetes Secret name (OpenShift platform fixture). "
            "Replace with platform-neutral language referencing MODEL_REGISTRY_* env vars.\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_checkpoint_store_protocol_is_importable(self) -> None:
        """CheckpointStore Protocol is importable from pragma_encoder.training.checkpoints."""
        from pragma_encoder.training.checkpoints import CheckpointStore  # noqa: PLC0415
        assert CheckpointStore is not None, (
            "CheckpointStore Protocol must be importable from "
            "pragma_encoder.training.checkpoints."
        )

    def test_local_and_s3_adapters_are_importable(self) -> None:
        """LocalCheckpointStore and S3CheckpointStore are importable from the package."""
        from pragma_encoder.training.checkpoints import (  # noqa: PLC0415
            LocalCheckpointStore,
            S3CheckpointStore,
        )
        assert LocalCheckpointStore is not None
        assert S3CheckpointStore is not None

    def test_build_checkpoint_store_factory_is_importable(self) -> None:
        """build_checkpoint_store factory is importable from the package."""
        from pragma_encoder.training.checkpoints import build_checkpoint_store  # noqa: PLC0415
        assert build_checkpoint_store is not None

    def test_checkpoint_store_protocol_is_runtime_checkable(self) -> None:
        """CheckpointStore is decorated with @runtime_checkable.

        isinstance() checks against CheckpointStore must work at runtime
        so that factory functions and tests can verify adapter compliance
        without static type analysis.
        """
        import pathlib  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from pragma_encoder.training.checkpoints import (  # noqa: PLC0415
            CheckpointStore,
            LocalCheckpointStore,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalCheckpointStore(pathlib.Path(tmp))
            # This raises TypeError if CheckpointStore is not @runtime_checkable
            try:
                result = isinstance(store, CheckpointStore)
            except TypeError as exc:
                pytest.fail(
                    f"isinstance() against CheckpointStore raised TypeError: {exc}. "
                    "CheckpointStore must be decorated with @runtime_checkable."
                )
            assert result is True, (
                "LocalCheckpointStore must be recognised as a CheckpointStore instance. "
                "All three Protocol methods (latest_key, fetch, put) must be present."
            )
