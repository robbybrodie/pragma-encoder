"""Masked event modelling prediction head.

Implements the masked language modelling (MLM) head used during PRAGMA
pretraining, as described in PRAGMA paper Section 2.3.5.

The MLM head projects the History Encoder's contextualised representations
at masked positions back to the vocabulary space to predict the original
token IDs. This is the same architecture as the BERT MLM head: a linear
projection with GELU activation and layer norm, followed by an output
projection to vocab_size.

The MLM objective is applied with label smoothing (Section 2.3.5) to
reduce overconfidence on the large vocabulary.

Reference: Ostroukhov et al. (2026), Section 2.3.5
BERT reference: Devlin et al. (2019), arXiv:1810.04805
"""

import torch
import torch.nn as nn


class MLMHead(nn.Module):
    """Masked event modelling prediction head.

    Projects contextualised token representations to logits over the
    full vocabulary. Applied only at masked positions during pretraining.

    Architecture:
        Linear(d_model → d_model) → GELU → LayerNorm → Linear(d_model → vocab_size)

    This two-layer design (same as BERT) provides a non-linear projection
    that decouples the MLM prediction space from the encoder's representation
    space, allowing the encoder to learn richer representations without
    being constrained by the token prediction objective.

    Args:
        d_model: Hidden dimension (must match History Encoder output).
        vocab_size: Total token vocabulary size.
        dropout: Dropout on the intermediate projection. Default: 0.0.
    """

    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.0):
        super().__init__()

        self.dense = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)

        # Bias separate from weight matrix (same as BERT)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights with normal(0, 0.02) — same as BERT."""
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states: (..., d_model) — representations at masked positions.

        Returns:
            (..., vocab_size) — unnormalised logits for token prediction.
        """
        x = self.dense(hidden_states)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        logits = self.decoder(x)
        return logits
