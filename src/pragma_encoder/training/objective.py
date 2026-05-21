"""Masked event modelling loss with label smoothing.

Implements the pretraining objective from PRAGMA paper Section 2.3.5.

PRAGMA is pretrained with a masked event modelling (MEM) objective, which
is a variant of masked language modelling (MLM) applied to financial event
sequences. The loss is cross-entropy over the vocabulary at masked positions,
with label smoothing to prevent overconfidence.

Label smoothing (from Section 2.3.5):
    The PRAGMA vocabulary is large and many tokens are plausible predictions
    for a masked position (e.g. different amounts in similar transactions).
    Label smoothing distributes a small probability mass ε across all
    vocabulary tokens, reducing overconfident predictions and improving
    generalisation.

Loss formula:
    L = (1 - ε) * CE(logits, targets) + ε * mean(CE(logits, uniform))

where ε = label_smoothing (default 0.1).

Reference: Ostroukhov et al. (2026), Section 2.3.5
Label smoothing: Szegedy et al. (2016), arXiv:1512.00567
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedEventModellingLoss(nn.Module):
    """Cross-entropy MLM loss with label smoothing.

    Applied only at masked positions (positions where labels != -100).
    Label smoothing distributes ε probability mass uniformly across
    the vocabulary.

    Args:
        vocab_size: Total token vocabulary size.
        label_smoothing: Label smoothing factor ε. Default: 0.1.
        ignore_index: Token ID to ignore in loss computation. Default: -100.
    """

    def __init__(
        self,
        vocab_size: int,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked event modelling loss.

        Args:
            logits: (..., vocab_size) — unnormalised predictions from MLMHead.
            labels: (...) — target token IDs, -100 at non-masked positions.

        Returns:
            Scalar loss value.
        """
        # Flatten for cross_entropy
        logits_flat = logits.view(-1, self.vocab_size)
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return loss

    def perplexity(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perplexity at masked positions (for logging).

        Args:
            logits: (..., vocab_size)
            labels: (...) with -100 at non-masked positions.

        Returns:
            Scalar perplexity value.
        """
        # Use no label smoothing for perplexity (raw NLL)
        logits_flat = logits.view(-1, self.vocab_size)
        labels_flat = labels.view(-1)
        nll = F.cross_entropy(logits_flat, labels_flat, ignore_index=self.ignore_index)
        return torch.exp(nll)
