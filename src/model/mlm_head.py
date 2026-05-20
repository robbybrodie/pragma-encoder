"""MLMHead — masked event modelling prediction head (§2.3.5, Equation 8).

Implements the MLM head described in PRAGMA paper Section 2.3.5.

The MLM head receives three pre-gathered context vectors for each masked
token position and projects their concatenation to logits over the value
vocabulary for cross-entropy loss computation.

Contract with callers (PRAGMA.forward):
    GATHERING (done by caller, NOT by this class):
        For each position (b, i, j) where mask[b, i, j] = True, the caller
        gathers three d-dimensional vectors and stacks them to (n_masked, d_model):
          z_hat_e_ij = z_hat_e[b, i, j, :]   — local token context
          zh_i       = zh[b, i+1, :]          — event i is at zh position i+1
                                                 ([USR] occupies position 0)
          zh_0       = zh[b, 0, :]            — [USR] token

    FORWARD (Equation 8):
        mlm_input = cat([z_hat_e_ij, zh_i, zh_0], dim=-1)  # (n_masked, 3*d_model)
        hidden    = Linear(3*d_model → d_model)(mlm_input)  # (n_masked, d_model)
        hidden    = GELU(hidden)
        logits    = Linear(d_model → value_vocab_size)(hidden)  # (n_masked, vocab)

    LOSS:
        Cross-entropy with label smoothing (§2.3.5).
        Label smoothing ε = 0.1 — implementation choice.
        The paper specifies "label smoothing" but does not give ε.
        Applied only at mask=True positions — the caller passes only the
        n_masked gathered tensors, so all positions here are in the loss.

Key design decisions:
    - No LayerNorm between the two Linear layers (not in paper — stub was wrong)
    - No separate bias parameter (BERT pattern not needed — standard nn.Linear
      bias is sufficient)
    - No dropout in the MLM head (paper does not specify dropout here)
    - Intermediate dimension = d_model (key-numbers.md: mlm_head_output_dim = d_model)
    - Input dimension = 3*d_model (key-numbers.md: mlm_head_input_dim = 3×d_model)

VOCABULARY DESIGN DECISION:
    The MLM head predicts over the VALUE vocabulary only
    (config.value_vocab_size ≈ 28k tokens).
    Target IDs passed as labels must be value-vocab-local IDs
    in range [0, config.value_vocab_size).
    See src/model/assembler.py — EmbeddingAssembler is responsible
    for converting global token IDs to value-vocab-local IDs
    before passing them as MLM targets.
    Global IDs (including key vocab offsets) must NOT be
    used as targets — this would cause out-of-range errors.

Reference: Ostroukhov et al. (2026), Section 2.3.5, Equation 8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import PRAGMAConfig

# Label smoothing epsilon — implementation choice.
# The paper (§2.3.5) specifies cross-entropy with label smoothing but does
# not give the exact epsilon. 0.1 is standard in the MLM literature.
_LABEL_SMOOTHING: float = 0.1


class MLMHead(nn.Module):
    """Masked event modelling prediction head (§2.3.5, Equation 8).

    Projects the concatenation of three context vectors at each masked token
    position to logits over the value vocabulary.

    Architecture (Equation 8):
        input  = cat([z_hat_e_ij, zh_i, zh_0], dim=-1)   # (n_masked, 3*d_model)
        hidden = Linear(3*d_model, d_model)(input)        # (n_masked, d_model)
        hidden = GELU(hidden)
        logits = Linear(d_model, value_vocab_size)(hidden) # (n_masked, value_vocab_size)

    No LayerNorm between the two layers — the paper does not specify one.

    Args:
        config: PRAGMAConfig — provides d_model and value_vocab_size.
                No individual arguments.

    Attributes:
        value_vocab_size: int — output vocabulary dimension (config.value_vocab_size).
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        super().__init__()
        self.value_vocab_size: int = config.value_vocab_size

        # Layer 1: project from concatenated 3-vector input to d_model
        # key-numbers.md: mlm_head_input_dim = 3×d_model (§2.3.5)
        self.proj = nn.Linear(3 * config.d_model, config.d_model)

        # Layer 2: project from d_model to value vocabulary
        # key-numbers.md: mlm_head_output_dim = d_model → value_vocab_size (§2.3.5)
        self.decoder = nn.Linear(config.d_model, config.value_vocab_size)

        self.gelu = nn.GELU()

    def forward(
        self,
        z_hat_e_ij: torch.Tensor,  # (n_masked, d_model) — local token context
        zh_i:       torch.Tensor,  # (n_masked, d_model) — event i context
        zh_0:       torch.Tensor,  # (n_masked, d_model) — [USR] context
    ) -> torch.Tensor:             # (n_masked, value_vocab_size) — logits
        """Project three context vectors to vocabulary logits (Equation 8).

        Args:
            z_hat_e_ij: Event Encoder token output at the masked position.
                        Shape: (n_masked, d_model). Pre-gathered by caller.
            zh_i:       History Encoder output for event i (position i+1 in zh,
                        because [USR] is at position 0). Shape: (n_masked, d_model).
            zh_0:       History Encoder [USR] token output (position 0 in zh).
                        Shape: (n_masked, d_model). Broadcast by caller.

        Returns:
            logits: Unnormalised logits over value vocabulary.
                    Shape: (n_masked, value_vocab_size).
        """
        # Concatenate along the feature dimension — Equation 8
        # (n_masked, d_model) × 3 → (n_masked, 3*d_model)
        mlm_input = torch.cat([z_hat_e_ij, zh_i, zh_0], dim=-1)

        # Linear(3*d_model → d_model) → GELU → Linear(d_model → value_vocab_size)
        hidden = self.gelu(self.proj(mlm_input))   # (n_masked, d_model)
        logits = self.decoder(hidden)              # (n_masked, value_vocab_size)
        return logits

    def compute_loss(
        self,
        logits:  torch.Tensor,  # (n_masked, value_vocab_size) — from forward()
        targets: torch.Tensor,  # (n_masked,) — integer token IDs (original)
    ) -> torch.Tensor:          # scalar — mean cross-entropy with label smoothing
        """Cross-entropy loss with label smoothing (§2.3.5).

        Computes the mean cross-entropy loss over all n_masked positions.
        Label smoothing ε = 0.1 is an implementation choice — the paper
        specifies label smoothing but does not give the exact epsilon.

        Args:
            logits:  Logits from forward(). Shape: (n_masked, value_vocab_size).
            targets: Original token IDs at masked positions. Shape: (n_masked,).
                     Integer dtype. These are the labels for the MLM objective.

        Returns:
            loss: Scalar cross-entropy loss with label smoothing.
                  Positive. Differentiable (requires_grad=True when logits
                  are part of the computation graph).
        """
        return F.cross_entropy(logits, targets, label_smoothing=_LABEL_SMOOTHING)
