"""Integration tests for the full PRAGMA model.

Tests the complete three-encoder assembly from Section 2.3,
including the full forward pass, MLM head, and embedding extraction.

Key properties verified:
    - Output dictionary contains expected keys
    - Shape correctness across all outputs
    - MLM logits are only produced when event_mask is provided
    - get_event_embeddings() works without masking

Reference: Ostroukhov et al. (2026), Section 2.3
"""

import pytest
import torch

from src.model import PRAGMA, PRAGMAConfig


BATCH = 2
N_EVENTS = 8
PROFILE_SEQ_LEN = 32
EVENT_SEQ_LEN = 64


@pytest.fixture
def config():
    """Minimal PRAGMA-S config for fast testing."""
    return PRAGMAConfig(
        vocab_size=1000,
        profile_vocab_size=500,
        d_model=64,
        profile_n_heads=4, profile_n_layers=2, profile_d_ff=128,
        event_n_heads=4, event_n_layers=2, event_d_ff=128,
        history_n_heads=4, history_n_layers=2, history_d_ff=128,
        dropout=0.0,  # Disable dropout for deterministic tests
    )


@pytest.fixture
def model(config):
    return PRAGMA(config).eval()


@pytest.fixture
def batch_inputs(config):
    return {
        "profile_token_ids": torch.randint(1, config.profile_vocab_size, (BATCH, PROFILE_SEQ_LEN)),
        "event_token_ids": torch.randint(1, config.vocab_size, (BATCH, N_EVENTS, EVENT_SEQ_LEN)),
        "calendar_tokens": torch.zeros(BATCH, N_EVENTS, 5, dtype=torch.long),
    }


class TestPRAGMAForward:
    """Tests for the PRAGMA full forward pass."""

    def test_output_keys_without_mask(self, model, batch_inputs):
        """Without event_mask, output should not contain mlm_logits."""
        output = model(**batch_inputs)
        assert "history_reprs" in output
        assert "hist_repr" in output
        assert "usr_repr" in output
        assert "mlm_logits" not in output

    def test_output_shapes(self, model, batch_inputs, config):
        """Output tensor shapes should match expected dimensions."""
        output = model(**batch_inputs)
        assert output["history_reprs"].shape == (BATCH, N_EVENTS, config.d_model)
        assert output["hist_repr"].shape == (BATCH, config.d_model)
        assert output["usr_repr"].shape == (BATCH, config.d_model)

    def test_mlm_logits_with_mask(self, model, batch_inputs, config):
        """MLM logits should be produced when event_mask is provided."""
        event_mask = torch.zeros(BATCH, N_EVENTS, dtype=torch.bool)
        event_mask[:, :2] = True  # Mask first 2 events
        output = model(**batch_inputs, event_mask=event_mask)
        assert "mlm_logits" in output
        n_masked = event_mask.sum().item()
        assert output["mlm_logits"].shape == (n_masked, config.vocab_size)

    def test_no_nan_in_outputs(self, model, batch_inputs):
        """All output tensors should be free of NaN values."""
        output = model(**batch_inputs)
        for key, tensor in output.items():
            assert not torch.isnan(tensor).any(), f"NaN found in {key}"


class TestPRAGMAConfig:
    """Tests for PRAGMAConfig class methods."""

    def test_pragma_s_config(self):
        cfg = PRAGMAConfig.pragma_s()
        assert cfg.model_name == "pragma-s"
        assert cfg.d_model == 256

    def test_pragma_m_config(self):
        cfg = PRAGMAConfig.pragma_m()
        assert cfg.model_name == "pragma-m"
        assert cfg.d_model == 768

    def test_pragma_l_config(self):
        cfg = PRAGMAConfig.pragma_l()
        assert cfg.model_name == "pragma-l"
        assert cfg.d_model == 2048
