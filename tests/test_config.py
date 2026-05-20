"""Tests for PRAGMAConfig.

Derived from PRAGMA paper Table 1 and Section 2.3, 2.3.5, 2.4, 3.1.2:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:      N/A — PRAGMAConfig is a dataclass, not an nn.Module
  Math tests:       verify mathematical invariants from Table 1
  Gradient tests:   N/A — dataclass has no parameters to train
  Spec tests:       verify values match paper exactly (key-numbers.md)

Every value asserted in this file appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

from src.model.config import PRAGMAConfig

VARIANTS = ["pragma_s", "pragma_m", "pragma_l"]


class TestConfigContract:
    """Verify PRAGMAConfig satisfies its interface contract."""

    def test_pragma_s_callable(self) -> None:
        """Table 1: pragma_s() must return a PRAGMAConfig."""
        config = PRAGMAConfig.pragma_s()
        assert isinstance(config, PRAGMAConfig)

    def test_pragma_m_callable(self) -> None:
        """Table 1: pragma_m() must return a PRAGMAConfig."""
        config = PRAGMAConfig.pragma_m()
        assert isinstance(config, PRAGMAConfig)

    def test_pragma_l_callable(self) -> None:
        """Table 1: pragma_l() must return a PRAGMAConfig."""
        config = PRAGMAConfig.pragma_l()
        assert isinstance(config, PRAGMAConfig)

    def test_required_fields_exist(self) -> None:
        """All fields required by downstream components must be present."""
        config = PRAGMAConfig.pragma_s()
        required = [
            # Width — Table 1
            "d_model", "d_ffn", "n_heads",
            # Depth — Table 1
            "profile_encoder_layers", "event_encoder_layers", "history_encoder_layers",
            # Regularisation — §2.3
            "dropout",
            # Truncation — §2.4
            "max_event_tokens", "max_profile_tokens", "max_events",
            # Masking — §2.3.5
            "token_mask_prob", "event_mask_prob", "key_mask_prob",
            # Vocabulary — §2.2
            "key_vocab_size", "value_vocab_size",
            # LoRA — §3.1.2
            "lora_rank", "lora_alpha",
            # Identity
            "model_name",
        ]
        for field in required:
            assert hasattr(config, field), (
                f"PRAGMAConfig missing required field: '{field}'"
            )


class TestMathProperties:
    """Verify mathematical invariants from Table 1.

    These invariants hold across all three variants.
    Source: docs/paper/key-numbers.md — derived invariants.
    """

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_ffn_is_four_times_dmodel(self, variant: str) -> None:
        """Table 1: d_ffn = 4 × d_model for all variants.

        key-numbers.md: ffn_ratio = 4 (derived from Table 1)
        S: 768 = 4 × 192, M: 2048 = 4 × 512, L: 4096 = 4 × 1024
        """
        config = getattr(PRAGMAConfig, variant)()
        assert config.d_ffn == 4 * config.d_model, (
            f"{variant}: d_ffn ({config.d_ffn}) must equal "
            f"4 × d_model ({config.d_model})"
        )

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_heads_divide_dmodel_evenly(self, variant: str) -> None:
        """Table 1: n_heads must divide d_model exactly (no remainder)."""
        config = getattr(PRAGMAConfig, variant)()
        assert config.d_model % config.n_heads == 0, (
            f"{variant}: d_model ({config.d_model}) not divisible "
            f"by n_heads ({config.n_heads})"
        )

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_head_dimension_is_64(self, variant: str) -> None:
        """Table 1: head dimension = d_model // n_heads = 64 for all variants.

        key-numbers.md: head_dimension = 64 (derived from Table 1)
        S: 192//3=64, M: 512//8=64, L: 1024//16=64
        """
        config = getattr(PRAGMAConfig, variant)()
        head_dim = config.d_model // config.n_heads
        assert head_dim == 64, (
            f"{variant}: head_dim ({head_dim}) must be 64"
        )


class TestPaperSpecifications:
    """Verify exact values from the paper.

    Every value asserted here appears in docs/paper/key-numbers.md
    with its paper source section.
    """

    def test_pragma_s_table_1(self) -> None:
        """Table 1 (page 6): PRAGMA-S exact architectural values.

        key-numbers.md: d_model=192, d_ffn=768, n_heads=3,
        profile_encoder_layers=1, event_encoder_layers=5,
        history_encoder_layers=2 — all from Table 1.
        """
        config = PRAGMAConfig.pragma_s()
        assert config.model_name == "pragma-s"
        assert config.d_model == 192               # key-numbers.md: Table 1
        assert config.d_ffn == 768                 # key-numbers.md: Table 1
        assert config.n_heads == 3                 # key-numbers.md: Table 1
        assert config.profile_encoder_layers == 1  # key-numbers.md: Table 1
        assert config.event_encoder_layers == 5    # key-numbers.md: Table 1
        assert config.history_encoder_layers == 2  # key-numbers.md: Table 1

    def test_pragma_m_table_1(self) -> None:
        """Table 1 (page 6): PRAGMA-M exact architectural values.

        key-numbers.md: d_model=512, d_ffn=2048, n_heads=8,
        profile_encoder_layers=3, event_encoder_layers=16,
        history_encoder_layers=6 — all from Table 1.
        """
        config = PRAGMAConfig.pragma_m()
        assert config.model_name == "pragma-m"
        assert config.d_model == 512               # key-numbers.md: Table 1
        assert config.d_ffn == 2048                # key-numbers.md: Table 1
        assert config.n_heads == 8                 # key-numbers.md: Table 1
        assert config.profile_encoder_layers == 3  # key-numbers.md: Table 1
        assert config.event_encoder_layers == 16   # key-numbers.md: Table 1
        assert config.history_encoder_layers == 6  # key-numbers.md: Table 1

    def test_pragma_l_table_1(self) -> None:
        """Table 1 (page 6): PRAGMA-L exact architectural values.

        key-numbers.md: d_model=1024, d_ffn=4096, n_heads=16,
        profile_encoder_layers=9, event_encoder_layers=45,
        history_encoder_layers=18 — all from Table 1.
        """
        config = PRAGMAConfig.pragma_l()
        assert config.model_name == "pragma-l"
        assert config.d_model == 1024               # key-numbers.md: Table 1
        assert config.d_ffn == 4096                 # key-numbers.md: Table 1
        assert config.n_heads == 16                 # key-numbers.md: Table 1
        assert config.profile_encoder_layers == 9   # key-numbers.md: Table 1
        assert config.event_encoder_layers == 45    # key-numbers.md: Table 1
        assert config.history_encoder_layers == 18  # key-numbers.md: Table 1

    def test_masking_rates_section_2_3_5(self) -> None:
        """Section 2.3.5: masking rates must match paper exactly.

        key-numbers.md: token_mask_prob=0.15, event_mask_prob=0.10,
        key_mask_prob=0.10 — from §2.3.5.
        """
        config = PRAGMAConfig.pragma_s()
        assert config.token_mask_prob == 0.15  # key-numbers.md: §2.3.5
        assert config.event_mask_prob == 0.10  # key-numbers.md: §2.3.5
        assert config.key_mask_prob == 0.10    # key-numbers.md: §2.3.5

    def test_dropout_and_truncation_limits(self) -> None:
        """Section 2.3 (dropout) and Section 2.4 (truncation limits).

        key-numbers.md: dropout=0.1 (§2.3), max_event_tokens=24 (§2.4),
        max_profile_tokens=200 (§2.4), max_events=6500 (§2.4).
        """
        config = PRAGMAConfig.pragma_s()
        assert config.dropout == 0.1              # key-numbers.md: §2.3
        assert config.max_event_tokens == 24      # key-numbers.md: §2.4
        assert config.max_profile_tokens == 200   # key-numbers.md: §2.4
        assert config.max_events == 6500          # key-numbers.md: §2.4

    def test_lora_defaults_section_3_1_2(self) -> None:
        """Section 3.1.2: LoRA default rank and alpha.

        key-numbers.md: lora_rank=8, lora_alpha=8 — from §3.1.2.
        """
        config = PRAGMAConfig.pragma_s()
        assert config.lora_rank == 8    # key-numbers.md: §3.1.2
        assert config.lora_alpha == 8   # key-numbers.md: §3.1.2

    def test_scaling_requires_one_line_change(self) -> None:
        """DEVELOPMENT_PROCESS.md: S→M requires changing exactly one line.

        Both variants must have identical field names. Only values differ.
        """
        s = PRAGMAConfig.pragma_s()
        m = PRAGMAConfig.pragma_m()
        assert set(vars(s).keys()) == set(vars(m).keys()), (
            "pragma_s and pragma_m must have identical field names. "
            "Switching variants must require changing exactly one line."
        )
