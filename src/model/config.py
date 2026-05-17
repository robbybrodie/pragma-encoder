"""PRAGMAConfig — architecture hyperparameters dataclass.

Centralises all hyperparameters for the PRAGMA model variants described
in PRAGMA paper Section 2.3.1.

The paper describes three scale variants:
    PRAGMA-S:  ~10M parameters (small — for research and development)
    PRAGMA-M: ~100M parameters (medium — aspirational open implementation)
    PRAGMA-L:   ~1B parameters (large  — aspirational open implementation)

This config covers all three scales. See configs/pragma_s.yaml,
configs/pragma_m.yaml, and configs/pragma_l.yaml for concrete values.

Key architectural constraints (non-negotiable, from the paper):
    - Encoder-only (not decoder-only)
    - Bidirectional attention (not causal)
    - Three separate encoders (not one monolithic encoder)
    - RoPE positional encoding (not sinusoidal or absolute)

Reference: Ostroukhov et al. (2026), Section 2.3.1
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PRAGMAConfig:
    """Hyperparameter configuration for the full PRAGMA model.

    Attributes:
        # Vocabulary
        vocab_size: Total token vocabulary size (from TokenizerPipeline).
        profile_vocab_size: Profile field vocabulary size.

        # Shared hidden dimension (all three encoders use the same d_model)
        d_model: Hidden dimension. Must be divisible by n_heads.

        # Profile State Encoder (Section 2.3.2)
        profile_n_heads: Attention heads in profile encoder.
        profile_n_layers: Transformer layers in profile encoder.
        profile_d_ff: Feed-forward dimension in profile encoder.
        profile_max_seq_len: Max profile sequence length.

        # Event Encoder (Section 2.3.3)
        event_n_heads: Attention heads in event encoder.
        event_n_layers: Transformer layers in event encoder.
        event_d_ff: Feed-forward dimension in event encoder.
        event_max_seq_len: Max tokens per event.

        # History Encoder (Section 2.3.4)
        history_n_heads: Attention heads in history encoder.
        history_n_layers: Transformer layers in history encoder.
        history_d_ff: Feed-forward dimension in history encoder.
        history_max_seq_len: Max events in history.

        # Regularisation
        dropout: Dropout probability.

        # Training objective
        mask_prob: Token/event masking probability for MLM. Default: 0.15.
        label_smoothing: Label smoothing factor for MLM loss. Default: 0.1.

        # Model name / variant identifier
        model_name: Human-readable model identifier.
    """

    # Vocabulary
    vocab_size: int = 50_000
    profile_vocab_size: int = 10_000

    # Shared hidden dimension
    d_model: int = 256

    # Profile State Encoder
    profile_n_heads: int = 8
    profile_n_layers: int = 4
    profile_d_ff: int = 1024
    profile_max_seq_len: int = 512

    # Event Encoder
    event_n_heads: int = 8
    event_n_layers: int = 4
    event_d_ff: int = 1024
    event_max_seq_len: int = 128

    # History Encoder
    history_n_heads: int = 8
    history_n_layers: int = 6
    history_d_ff: int = 1024
    history_max_seq_len: int = 512

    # Regularisation
    dropout: float = 0.1

    # Training objective
    mask_prob: float = 0.15
    label_smoothing: float = 0.1

    # Model variant name
    model_name: str = "pragma-s"

    @classmethod
    def pragma_s(cls) -> "PRAGMAConfig":
        """PRAGMA-S: ~10M parameters. Suitable for research and development."""
        return cls(
            d_model=256,
            profile_n_heads=8, profile_n_layers=4, profile_d_ff=1024,
            event_n_heads=8, event_n_layers=4, event_d_ff=1024,
            history_n_heads=8, history_n_layers=6, history_d_ff=1024,
            model_name="pragma-s",
        )

    @classmethod
    def pragma_m(cls) -> "PRAGMAConfig":
        """PRAGMA-M: ~100M parameters (aspirational)."""
        return cls(
            d_model=768,
            profile_n_heads=12, profile_n_layers=6, profile_d_ff=3072,
            event_n_heads=12, event_n_layers=6, event_d_ff=3072,
            history_n_heads=12, history_n_layers=12, history_d_ff=3072,
            model_name="pragma-m",
        )

    @classmethod
    def pragma_l(cls) -> "PRAGMAConfig":
        """PRAGMA-L: ~1B parameters (aspirational)."""
        return cls(
            d_model=2048,
            profile_n_heads=16, profile_n_layers=8, profile_d_ff=8192,
            event_n_heads=16, event_n_layers=8, event_d_ff=8192,
            history_n_heads=16, history_n_layers=24, history_d_ff=8192,
            model_name="pragma-l",
        )
