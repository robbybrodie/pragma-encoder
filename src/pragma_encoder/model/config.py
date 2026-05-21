"""PRAGMAConfig — architecture hyperparameters dataclass.

Single source of truth for all PRAGMA model variants.

Three scale variants from Table 1 (Ostroukhov et al., 2026):
    PRAGMA-S:  ~10M  parameters (d_model=192)
    PRAGMA-M: ~100M  parameters (d_model=512)
    PRAGMA-L:   ~1B  parameters (d_model=1024)

All values in this file appear in docs/paper/key-numbers.md
with their paper source section.

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1
"""

from dataclasses import dataclass


@dataclass
class PRAGMAConfig:
    """Hyperparameter configuration for the full PRAGMA model.

    Fields derived from:
      - Table 1 (p.6): width, depth, heads per variant
      - §2.3: dropout
      - §2.3.5: masking probabilities
      - §2.4: truncation limits
      - §2.2: vocabulary sizes
      - §3.1.2: LoRA defaults

    All three variants share the same field names; only values differ.
    This means switching variants requires changing exactly one line
    (the classmethod call site).
    """

    # Width — Table 1
    d_model: int = 192          # key-numbers.md: Table 1
    d_ffn: int = 768            # key-numbers.md: Table 1  (4 × d_model)
    n_heads: int = 3            # key-numbers.md: Table 1

    # Depth — Table 1
    profile_encoder_layers: int = 1    # key-numbers.md: Table 1
    event_encoder_layers: int = 5      # key-numbers.md: Table 1
    history_encoder_layers: int = 2    # key-numbers.md: Table 1

    # Regularisation — §2.3
    dropout: float = 0.1        # key-numbers.md: §2.3

    # Truncation — §2.4
    max_event_tokens: int = 24       # key-numbers.md: §2.4
    max_profile_tokens: int = 200    # key-numbers.md: §2.4
    max_events: int = 6500           # key-numbers.md: §2.4

    # Masking — §2.3.5
    token_mask_prob: float = 0.15    # key-numbers.md: §2.3.5
    event_mask_prob: float = 0.10    # key-numbers.md: §2.3.5
    key_mask_prob: float = 0.10      # key-numbers.md: §2.3.5

    # Vocabulary — §2.2
    key_vocab_size: int = 60             # key-numbers.md: §2.2 (~60 field types)
    value_vocab_size: int = 28_000       # key-numbers.md: §2.2 (~28,000 values)

    # LoRA — §3.1.2
    lora_rank: int = 8           # key-numbers.md: §3.1.2
    lora_alpha: int = 8          # key-numbers.md: §3.1.2

    # Identity
    model_name: str = "pragma-s"

    @classmethod
    def pragma_s(cls) -> "PRAGMAConfig":
        """PRAGMA-S: ~10M parameters. Table 1 (p.6).

        key-numbers.md: d_model=192, d_ffn=768, n_heads=3,
        profile_encoder_layers=1, event_encoder_layers=5,
        history_encoder_layers=2
        """
        return cls(
            model_name="pragma-s",
            d_model=192,
            d_ffn=768,
            n_heads=3,
            profile_encoder_layers=1,
            event_encoder_layers=5,
            history_encoder_layers=2,
        )

    @classmethod
    def pragma_m(cls) -> "PRAGMAConfig":
        """PRAGMA-M: ~100M parameters. Table 1 (p.6).

        key-numbers.md: d_model=512, d_ffn=2048, n_heads=8,
        profile_encoder_layers=3, event_encoder_layers=16,
        history_encoder_layers=6
        """
        return cls(
            model_name="pragma-m",
            d_model=512,
            d_ffn=2048,
            n_heads=8,
            profile_encoder_layers=3,
            event_encoder_layers=16,
            history_encoder_layers=6,
        )

    @classmethod
    def pragma_l(cls) -> "PRAGMAConfig":
        """PRAGMA-L: ~1B parameters. Table 1 (p.6).

        key-numbers.md: d_model=1024, d_ffn=4096, n_heads=16,
        profile_encoder_layers=9, event_encoder_layers=45,
        history_encoder_layers=18
        """
        return cls(
            model_name="pragma-l",
            d_model=1024,
            d_ffn=4096,
            n_heads=16,
            profile_encoder_layers=9,
            event_encoder_layers=45,
            history_encoder_layers=18,
        )
