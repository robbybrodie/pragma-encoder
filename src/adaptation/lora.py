"""LoRAAdapter — parameter-efficient fine-tuning for PRAGMA (§3.1.2).

Implements Low-Rank Adaptation (LoRA) of a pre-trained PRAGMA model as
described in PRAGMA paper Section 3.1.2.

LoRA introduces 2–4% parameter overhead by injecting trainable low-rank
delta matrices (A, B) into the target Linear layers of the three encoders.
All backbone weights are frozen; only the LoRA deltas are updated during
downstream fine-tuning.

Implementation uses Hugging Face PEFT library (ADR 006). Do not implement
LoRA from scratch — PEFT provides production-quality LoRA with support for
adapter saving, loading, and merging.

Target modules (verified empirically against PRAGMA module paths):
    "q_proj"   — Q projection in _RoPEMultiheadAttention / _EventAttention
    "k_proj"   — K projection in _RoPEMultiheadAttention / _EventAttention
    "v_proj"   — V projection in _RoPEMultiheadAttention / _EventAttention
    "out_proj" — output projection in _RoPEMultiheadAttention / _EventAttention
    "ff.0"     — first FFN linear (d_model → d_ffn) in *EncoderLayer.ff
    "ff.2"     — second FFN linear (d_ffn → d_model) in *EncoderLayer.ff

PEFT uses suffix matching: "q_proj" matches any module path ending in
".q_proj", so this covers all three encoders (profile, event, history)
without needing per-encoder path prefixes.

NOT targeted (excluded by design):
    event_encoder.calendar_mlp.mlp.* — feature embedding, not transformer layer
    mlm_head.proj / mlm_head.decoder — prediction head, not an encoder

Parameter coverage (PRAGMA-S, rank=8, alpha=8):
    LoRA params: 221,184 / 9,334,432 total = 2.37% — within paper's 2–4% range.

Key design decisions:
    - lora_dropout = 0.0 — implementation choice; paper does not specify dropout
      for LoRA. 0.0 is the conservative default (no additional regularisation).
    - bias = "none" — standard PEFT default; paper does not specify.
    - task_type = FEATURE_EXTRACTION — PRAGMA is encoder-only (no causal head).
    - All LoRA hyperparameters sourced from PRAGMAConfig (no hardcoded values).

Reference: Ostroukhov et al. (2026), Section 3.1.2
LoRA paper: Hu et al. (2022), arXiv:2106.09685
PEFT library: https://github.com/huggingface/peft
"""

from src.model.config import PRAGMAConfig

# ---------------------------------------------------------------------------
# Target module names — exact attribute paths in the PRAGMA encoder hierarchy
# ---------------------------------------------------------------------------

# These strings are suffix-matched by PEFT against the full named_modules() paths.
# Verified against PRAGMA-S: 221,184 LoRA params / 9,334,432 total = 2.37%.
# Do not change these without re-verifying the parameter fraction.
_TARGET_MODULES: list[str] = [
    "q_proj",    # Q projection  — Linear(d_model, d_model)
    "k_proj",    # K projection  — Linear(d_model, d_model)
    "v_proj",    # V projection  — Linear(d_model, d_model)
    "out_proj",  # output proj   — Linear(d_model, d_model)
    "ff.0",      # FFN up-proj   — Linear(d_model, d_ffn)
    "ff.2",      # FFN down-proj — Linear(d_ffn, d_model)
]

# LoRA dropout — implementation choice.
# The paper (§3.1.2) specifies LoRA but does not give a dropout value.
# 0.0 is the conservative default (no additional stochastic regularisation).
_LORA_DROPOUT: float = 0.0


class LoRAAdapter:
    """Parameter-efficient LoRA fine-tuning adapter for PRAGMA (§3.1.2).

    Wraps a pre-trained PRAGMA model with Low-Rank Adaptation (LoRA),
    freezing the backbone and injecting trainable low-rank delta matrices
    into the attention projections and FFN layers of all three encoders.

    The adapter is constructed from PRAGMAConfig only — no individual args.
    Scaling from PRAGMA-S to PRAGMA-M requires changing exactly one line
    (the PRAGMAConfig classmethod call).

    Args:
        config: PRAGMAConfig — provides lora_rank and lora_alpha.
                All LoRA hyperparameters are sourced from config.

    Attributes:
        config: PRAGMAConfig — stored for callers.

    Usage:
        model   = PRAGMA(config)
        adapter = LoRAAdapter(config)
        peft_model = adapter.apply(model)
        # peft_model.parameters() only yields LoRA deltas as trainable
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        self.config = config

        # Build LoraConfig now — validated at construction, not at apply() time.
        # Import is deferred to this point so that peft is only required when
        # LoRAAdapter is actually instantiated (not at module import time).
        try:
            from peft import LoraConfig, TaskType
        except ImportError as exc:
            raise ImportError(
                "The `peft` library is required for LoRAAdapter. "
                "Install with: pip install peft"
            ) from exc

        self._lora_config = LoraConfig(
            r=config.lora_rank,              # key-numbers.md: lora_rank=8, §3.1.2
            lora_alpha=config.lora_alpha,    # key-numbers.md: lora_alpha=8, §3.1.2
            target_modules=_TARGET_MODULES,  # QKV + FFN — see module docstring
            lora_dropout=_LORA_DROPOUT,      # 0.0; paper does not specify dropout
            bias="none",                     # standard PEFT default
            task_type=TaskType.FEATURE_EXTRACTION,  # PRAGMA is encoder-only
        )

    def apply(self, model: "PRAGMA") -> "peft.PeftModel":  # noqa: F821
        """Wrap a PRAGMA model with LoRA adapters.

        Freezes all backbone parameters and injects trainable LoRA delta
        matrices into the target modules. The returned PeftModel has:
          - LoRA parameters (lora_A, lora_B): requires_grad=True
          - All other parameters:             requires_grad=False

        Args:
            model: A PRAGMA model instance — pre-trained or randomly initialised.
                   The model is modified in-place by get_peft_model().

        Returns:
            peft_model: A peft.PeftModel wrapping the PRAGMA backbone.
                        Callers use this for fine-tuning, saving adapters,
                        and merging LoRA weights back into the backbone.

        Raises:
            ImportError: If the `peft` library is not installed.
        """
        from peft import get_peft_model

        return get_peft_model(model, self._lora_config)
