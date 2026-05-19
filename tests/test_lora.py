"""Tests for LoRAAdapter.

Derived from PRAGMA paper Section 3.1.2:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Apply tests:      verify frozen backbone, trainable LoRA params, PeftModel return
  Paper spec tests: verify 2–4% overhead bound, rank=8, alpha=8
  Architecture tests: class name, config constructor, QKV module targeting

Key paper properties (§3.1.2):
  - LoRA introduces 2–4% parameter overhead only
  - Target modules: QKV projections and MLP layers within encoder layers
  - Default rank = 8   (key-numbers.md: lora_rank=8, §3.1.2)
  - Default alpha = 8  (key-numbers.md: lora_alpha=8, §3.1.2)
  - Backbone parameters are frozen — only LoRA deltas are trained
  - Implementation uses Hugging Face PEFT library (ADR 006)

Exact target_modules verified empirically against PRAGMA module paths:
  ["q_proj", "k_proj", "v_proj", "out_proj", "ff.0", "ff.2"]
  → 221,184 LoRA params / 9,334,432 total = 2.37% on PRAGMA-S

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
peft  = pytest.importorskip("peft",  reason="peft not installed")

from src.adaptation.lora import LoRAAdapter
from src.model.config import PRAGMAConfig
from src.model.pragma import PRAGMA

_CONFIG = PRAGMAConfig.pragma_s()


def _make_model() -> PRAGMA:
    """Return a fresh PRAGMA-S model in eval mode."""
    return PRAGMA(_CONFIG).eval()


# ---------------------------------------------------------------------------
# TestApply — frozen backbone, trainable LoRA params, PeftModel type
# ---------------------------------------------------------------------------


class TestApply:
    """Verify apply() correctly wraps the model with LoRA (§3.1.2)."""

    def test_lora_params_require_grad(self) -> None:
        """§3.1.2: LoRA adapter parameters must have requires_grad=True.

        Only the LoRA delta matrices (lora_A, lora_B) are trained during
        fine-tuning. All other parameters are frozen. This test confirms
        that at least the LoRA parameters are trainable.
        """
        adapter = LoRAAdapter(_CONFIG)
        peft_model = adapter.apply(_make_model())

        lora_params = [
            p for name, p in peft_model.named_parameters()
            if "lora_" in name
        ]
        assert len(lora_params) > 0, (
            "§3.1.2: no LoRA parameters found after apply(). "
            "check that target_modules matches actual module names."
        )
        for p in lora_params:
            assert p.requires_grad, (
                "§3.1.2: LoRA parameter does not require grad. "
                "LoRA deltas must be trainable."
            )

    def test_backbone_params_frozen(self) -> None:
        """§3.1.2: backbone parameters must have requires_grad=False after apply().

        The pre-trained PRAGMA weights are frozen during LoRA fine-tuning.
        Only the injected LoRA delta matrices are updated. If backbone params
        are not frozen, LoRA provides no efficiency benefit and full fine-tuning
        occurs instead.
        """
        adapter = LoRAAdapter(_CONFIG)
        peft_model = adapter.apply(_make_model())

        non_lora_params = [
            (name, p) for name, p in peft_model.named_parameters()
            if "lora_" not in name
        ]
        for name, p in non_lora_params:
            assert not p.requires_grad, (
                f"§3.1.2: backbone parameter '{name}' has requires_grad=True. "
                "All non-LoRA parameters must be frozen after apply()."
            )

    def test_at_least_one_lora_param_exists(self) -> None:
        """§3.1.2: at least one LoRA parameter must exist — LoRA was actually applied.

        PEFT silently produces no LoRA parameters if target_modules does not
        match any module in the model. This test guards against that silent
        failure — wrong module names produce no error, no LoRA, no training.
        """
        adapter = LoRAAdapter(_CONFIG)
        peft_model = adapter.apply(_make_model())

        lora_param_names = [
            name for name, _ in peft_model.named_parameters()
            if "lora_" in name
        ]
        assert len(lora_param_names) > 0, (
            "§3.1.2: LoRA was not applied to any module. "
            "target_modules must match actual module attribute names: "
            "['q_proj', 'k_proj', 'v_proj', 'out_proj', 'ff.0', 'ff.2']"
        )

    def test_returns_peft_model(self) -> None:
        """§3.1.2: apply() must return a peft.PeftModel instance.

        Callers depend on peft.PeftModel for adapter management
        (saving, loading, merging). Returning a plain nn.Module would
        break downstream adapter handling.
        """
        adapter = LoRAAdapter(_CONFIG)
        result = adapter.apply(_make_model())

        assert isinstance(result, peft.PeftModel), (
            f"§3.1.2: apply() must return a peft.PeftModel instance, "
            f"got {type(result).__name__}."
        )


# ---------------------------------------------------------------------------
# TestPaperSpecifications — parameter fraction, rank, alpha
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact values from key-numbers.md (§3.1.2)."""

    def test_trainable_fraction_under_10pct(self) -> None:
        """key-numbers.md: lora_param_fraction = 2–4% (§3.1.2).

        The test bound is < 10% to allow for variation across model sizes
        while catching gross mismatches (e.g. accidentally training all params).
        On PRAGMA-S with the verified target_modules, the actual fraction is
        ~2.37% — well within the paper's 2–4% overhead claim.
        """
        adapter = LoRAAdapter(_CONFIG)
        peft_model = adapter.apply(_make_model())

        n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in peft_model.parameters())
        fraction    = n_trainable / n_total

        assert fraction < 0.10, (  # key-numbers.md: lora_param_fraction = 2–4% (§3.1.2)
            f"key-numbers.md: LoRA overhead must be < 10% (paper: 2–4%). "
            f"Got {fraction:.2%} ({n_trainable:,} / {n_total:,}). "
            f"Check that backbone parameters are frozen after apply()."
        )

    def test_at_least_one_pct_trainable(self) -> None:
        """key-numbers.md: lora_param_fraction = 2–4% (§3.1.2).

        Ensures LoRA is not a no-op — at least 1% of parameters must be
        trainable. If this fails, target_modules matched nothing and LoRA
        was silently skipped (all backbone params frozen, fraction = 0%).
        """
        adapter = LoRAAdapter(_CONFIG)
        peft_model = adapter.apply(_make_model())

        n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in peft_model.parameters())
        fraction    = n_trainable / n_total

        assert fraction > 0.01, (  # key-numbers.md: lora_param_fraction = 2–4% (§3.1.2)
            f"key-numbers.md: LoRA overhead must be > 1% (paper: 2–4%). "
            f"Got {fraction:.2%} ({n_trainable:,} / {n_total:,}). "
            f"target_modules likely matched no modules — LoRA is a no-op."
        )

    def test_rank_from_config(self) -> None:
        """key-numbers.md: lora_rank = 8, §3.1.2.

        The LoRA rank must come from config.lora_rank — not hardcoded.
        Verified by checking that config.lora_rank == 8 for all variants
        (the value is set in PRAGMAConfig.pragma_s/m/l() classmethods).
        """
        assert _CONFIG.lora_rank == 8, (  # key-numbers.md: lora_rank=8, §3.1.2
            f"key-numbers.md: config.lora_rank must be 8 (§3.1.2), "
            f"got {_CONFIG.lora_rank}."
        )
        # Verify all three variants have lora_rank=8
        for config in [
            PRAGMAConfig.pragma_s(),
            PRAGMAConfig.pragma_m(),
            PRAGMAConfig.pragma_l(),
        ]:
            assert config.lora_rank == 8, (  # key-numbers.md: lora_rank=8, §3.1.2
                f"key-numbers.md: {config.model_name} lora_rank must be 8, "
                f"got {config.lora_rank}."
            )

    def test_alpha_from_config(self) -> None:
        """key-numbers.md: lora_alpha = 8, §3.1.2.

        The LoRA scaling factor alpha must come from config.lora_alpha — not
        hardcoded. The effective learning rate scale for LoRA is alpha/rank;
        with rank=alpha=8, the scale factor is 1.0.
        """
        assert _CONFIG.lora_alpha == 8, (  # key-numbers.md: lora_alpha=8, §3.1.2
            f"key-numbers.md: config.lora_alpha must be 8 (§3.1.2), "
            f"got {_CONFIG.lora_alpha}."
        )
        # Verify all three variants have lora_alpha=8
        for config in [
            PRAGMAConfig.pragma_s(),
            PRAGMAConfig.pragma_m(),
            PRAGMAConfig.pragma_l(),
        ]:
            assert config.lora_alpha == 8, (  # key-numbers.md: lora_alpha=8, §3.1.2
                f"key-numbers.md: {config.model_name} lora_alpha must be 8, "
                f"got {config.lora_alpha}."
            )


# ---------------------------------------------------------------------------
# TestArchitecture — class name, config constructor, QKV targeting
# ---------------------------------------------------------------------------


class TestArchitecture:
    """Verify LoRAAdapter architecture constraints (§3.1.2)."""

    def test_class_name_is_lora_adapter(self) -> None:
        """DEVELOPMENT_PROCESS.md: class must be named LoRAAdapter exactly.

        The naming conventions table maps src/adaptation/lora.py → LoRAAdapter.
        Any deviation (e.g. PRAGMALoRAConfig, LoraAdapter) violates the
        paper naming contract.
        """
        assert LoRAAdapter.__name__ == "LoRAAdapter", (
            f"Class must be named 'LoRAAdapter', got '{LoRAAdapter.__name__}'. "
            "DEVELOPMENT_PROCESS.md: class names must match the paper exactly."
        )

    def test_constructor_takes_config(self) -> None:
        """DEVELOPMENT_PROCESS.md: constructor must be (self, config: PRAGMAConfig).

        All three model sizes must be constructable without individual args.
        Scaling from PRAGMA-S to PRAGMA-M must require changing exactly one line.
        """
        adapter_s = LoRAAdapter(PRAGMAConfig.pragma_s())
        adapter_m = LoRAAdapter(PRAGMAConfig.pragma_m())
        adapter_l = LoRAAdapter(PRAGMAConfig.pragma_l())

        # Verify config is stored and accessible
        assert adapter_s.config.lora_rank == 8   # key-numbers.md: §3.1.2
        assert adapter_m.config.lora_rank == 8   # key-numbers.md: §3.1.2
        assert adapter_l.config.lora_rank == 8   # key-numbers.md: §3.1.2

    def test_lora_applied_to_qkv(self) -> None:
        """§3.1.2: LoRA must be applied to QKV projections within encoder layers.

        The paper targets 'QKV projections and MLP layers within encoder layers'.
        This test verifies that LoRA parameter names contain q_proj, k_proj,
        and v_proj paths — confirming the correct module names were used.
        Incorrect names (e.g. 'in_proj_weight', 'query') produce no LoRA params
        and no error — this test catches that silent failure.
        """
        adapter = LoRAAdapter(_CONFIG)
        peft_model = adapter.apply(_make_model())

        lora_names = [
            name for name, _ in peft_model.named_parameters()
            if "lora_" in name
        ]

        # Check that q_proj, k_proj, v_proj are all represented
        for proj in ("q_proj", "k_proj", "v_proj"):
            found = any(proj in name for name in lora_names)
            assert found, (
                f"§3.1.2: no LoRA parameters found for '{proj}'. "
                f"target_modules must include '{proj}' to target QKV projections. "
                f"Found LoRA param names: {lora_names[:5]}..."
            )
