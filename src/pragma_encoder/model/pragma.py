"""PRAGMA — full three-encoder foundation model (Section 2.3).

Implements the complete PRAGMA model described in PRAGMA paper Section 2.3,
assembling the three-encoder architecture and MLM head into a single nn.Module.

Architecture (Equations 4–8):
    ProfileStateEncoder  (§2.3.2) — static customer profile + RoPE on ta
    EventEncoder         (§2.3.3) — per-event token encoding + calendar MLP
    HistoryEncoder       (§2.3.4) — [USR:EVT] sequence + RoPE on te
    MLMHead              (§2.3.5) — 3×d → d → value_vocab_size prediction head

Forward pass — exact six-step sequence:

    Step 1: za     = ProfileStateEncoder(xa, ta)          (batch, na, d_model)
            za_usr = za[:,0:1,:]                          (batch, 1, d_model)
            [Only [USR] at position 0 passes to History Encoder — Equation 4]

    Step 2: z_hat_e, ze = EventEncoder(xe, xt)
            z_hat_e: (batch, ne, ni, d_model) — token-level output for MLM
            ze:      (batch, ne, d_model)     — calendar-augmented [EVT] tokens

    Step 3: z = cat([za_usr, ze], dim=1)                  (batch, 1+ne, d_model)
            [Equation 6: z = [za : ze]]

    Step 4: zh = HistoryEncoder(z, te)                    (batch, 1+ne, d_model)
            [Equation 7: zh with [USR] at 0, [EVT] at 1..ne]

    Step 5: Gather masked positions (when mask is not None):
            b_idx, i_idx, j_idx = mask.nonzero(as_tuple=True)
            z_hat_e_ij = z_hat_e[b_idx, i_idx, j_idx, :]   (n_masked, d_model)
            zh_i       = zh[b_idx, i_idx + 1, :]            (n_masked, d_model)
              ↑ event i is at position i+1 in zh ([USR] occupies position 0)
            zh_0       = zh[b_idx, 0, :]                    (n_masked, d_model)

    Step 6: logits = MLMHead(z_hat_e_ij, zh_i, zh_0)     (n_masked, value_vocab_size)
            [Equation 8: input = [z_hat_e_ij : zh_i : zh_0] ∈ R^(3d)]

Contract with callers:
    INPUT — all pre-embedded float tensors (Equation 1 applied OUTSIDE this class):
        xa:        (batch, na, d_model)     — profile state embeddings
        ta:        (batch, na)              — profile temporal coords (log-seconds)
        xe:        (batch, ne, ni, d_model) — event token embeddings ([EVT] at pos 0)
        xt:        (batch, ne, 3)           — calendar features (integer: h, dow, dom)
        te:        (batch, 1+ne)            — history temporal coords (0.0 for [USR])
        token_ids: (batch, ne, ni)          — original integer token IDs (optional;
                                              reserved for target gathering; not
                                              used in the current forward pass)
        mask:      (batch, ne, ni) bool     — True at positions in MLM loss

    OUTPUT:
        {"zh": zh}                   — always (embedding extraction path)
        {"zh": zh, "logits": logits} — when mask is not None (pre-training path)
        zh:     (batch, 1+ne, d_model)        — full history encoder output
        logits: (n_masked, value_vocab_size)  — MLM logits at masked positions

Key design decisions:
    - No embedding table in PRAGMA — Equation 1 is handled externally
      by EmbeddingAssembler (src/model/assembler.py) using VocabularySpec
      exported from TokenizerPipeline.
    - No causal masking anywhere — PRAGMA is encoder-only (§2.3)
    - No weight sharing between encoders (ADR 002)
    - token_ids is accepted but not used internally; it is reserved for callers
      that need it for loss computation (targets = token_ids at masked positions)
    - Gathering uses i+1 (not i) because [USR] occupies position 0 of zh

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

Reference: Ostroukhov et al. (2026), Section 2.3, Equations 4–8
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from pragma_encoder.encoders import EventEncoder, HistoryEncoder, ProfileStateEncoder
from pragma_encoder.model.config import PRAGMAConfig
from pragma_encoder.model.mlm_head import MLMHead


class PRAGMA(nn.Module):
    """Full PRAGMA foundation model — three-encoder assembly (§2.3).

    Composes ProfileStateEncoder, EventEncoder, HistoryEncoder, and MLMHead
    into the complete architecture from Section 2.3. Accepts pre-embedded
    float tensors — embedding (Equation 1) is done externally.

    Args:
        config: PRAGMAConfig — single source of truth for all hyperparameters.
                All four sub-modules are constructed from config only.

    Attributes:
        config:           PRAGMAConfig — stored for callers (e.g. config.d_model)
        profile_encoder:  ProfileStateEncoder — §2.3.2
        event_encoder:    EventEncoder        — §2.3.3
        history_encoder:  HistoryEncoder      — §2.3.4
        mlm_head:         MLMHead             — §2.3.5
    """

    def __init__(self, config: PRAGMAConfig) -> None:
        super().__init__()
        self.config = config

        # Three independent encoders — no shared weights (ADR 002)
        self.profile_encoder = ProfileStateEncoder(config)  # §2.3.2
        self.event_encoder   = EventEncoder(config)         # §2.3.3
        self.history_encoder = HistoryEncoder(config)       # §2.3.4

        # MLM prediction head for pre-training (§2.3.5, Equation 8)
        self.mlm_head = MLMHead(config)

    def forward(
        self,
        xa:          torch.Tensor,            # (batch, na, d_model)     — profile embeddings
        ta:          torch.Tensor,            # (batch, na)              — profile temporal
        xe:          torch.Tensor,            # (batch, ne, ni, d_model) — event embeddings
        xt:          torch.Tensor,            # (batch, ne, 3)           — calendar features
        te:          torch.Tensor,            # (batch, 1+ne)            — history temporal
        token_ids:   Optional[torch.Tensor] = None,  # (batch, ne, ni)  — original IDs
        mask:        Optional[torch.Tensor] = None,  # (batch, ne, ni) bool — MLM mask
        xe_valid:    Optional[torch.Tensor] = None,  # (batch, ne, ni) bool — True=real token
        event_valid: Optional[torch.Tensor] = None,  # (batch, ne) bool — False = padding
    ) -> Dict[str, torch.Tensor]:
        """Six-step PRAGMA forward pass (Equations 4–8).

        Args:
            xa:        Profile state token embeddings. Shape: (batch, na, d_model).
                       Float tensor. [USR] token is at position 0 (prepended by caller).
            ta:        Profile temporal coordinates (log-seconds). Shape: (batch, na).
                       Used by ProfileStateEncoder for RoPE. 0.0 for non-life-long tokens.
            xe:        Event token embeddings. Shape: (batch, ne, ni, d_model).
                       Float tensor. [EVT] token is at position 0 of each event.
            xt:        Calendar features. Shape: (batch, ne, 3). Integer tensor.
                       Values: [hour_of_day, day_of_week, day_of_month] (§2.2).
            te:        History temporal coordinates (log-seconds). Shape: (batch, 1+ne).
                       te[:,0] = 0.0 for [USR]; te[:,1:] = log-seconds per event.
            token_ids: Original integer token IDs. Shape: (batch, ne, ni). Optional.
                       Reserved for callers that compute loss externally using
                       token_ids at masked positions as targets.
            mask:        Token-level MLM mask. Shape: (batch, ne, ni). Bool. Optional.
                         True = position is in MLM loss (replaced with [MASK] by caller).
                         When None, MLM head is not called (embedding extraction mode).
            xe_valid:    Token-level validity mask. Shape: (batch, ne, ni). Bool. Optional.
                         True = real token, False = padding. Passed to EventEncoder so
                         padding token positions cannot influence [EVT] outputs (DEF-005a).
            event_valid: Event-level validity mask. Shape: (batch, ne). Bool. Optional.
                         True = real event, False = padding. When provided, ze for
                         False events is zeroed before entering HistoryEncoder so
                         padding noise cannot leak into the history representation.
                         Callers derive this as xe_valid.any(dim=-1) where xe_valid
                         is the (batch, ne, ni) token-level validity tensor.

        Returns:
            Dict with keys:
                "zh":     (batch, 1+ne, d_model) — History Encoder output. Always present.
                          zh[:,0,:]  = [USR] embedding (user-level representation)
                          zh[:,1:,:] = [EVT] embeddings (per-event representations)
                "logits": (n_masked, value_vocab_size) — MLM logits. Present only when
                          mask is not None. n_masked = mask.sum().
        """
        # ------------------------------------------------------------------
        # Step 1: Profile State Encoder → za → za_usr (Equation 4)
        # ------------------------------------------------------------------
        za     = self.profile_encoder(xa, ta)   # (batch, na, d_model)
        za_usr = za[:, 0:1, :]                  # (batch, 1, d_model) — [USR] only

        # ------------------------------------------------------------------
        # Step 2: Event Encoder → z_hat_e (token-level) + ze (EVT tokens)
        #         (Equations 3 and 5)
        # ------------------------------------------------------------------
        z_hat_e, ze = self.event_encoder(xe, xt, xe_valid=xe_valid)
        # z_hat_e: (batch, ne, ni, d_model) — token-level output for MLM gathering
        # ze:      (batch, ne, d_model)      — calendar-augmented [EVT] tokens

        # Zero out padded event slots so they cannot leak into HistoryEncoder.
        # event_valid: (batch, ne) bool — broadcast over d_model dimension.
        if event_valid is not None:
            ze = ze * event_valid.unsqueeze(-1).to(ze.dtype)

        # ------------------------------------------------------------------
        # Step 3: Assemble [USR:EVT] sequence for History Encoder (Equation 6)
        # ------------------------------------------------------------------
        z = torch.cat([za_usr, ze], dim=1)   # (batch, 1+ne, d_model)

        # ------------------------------------------------------------------
        # Step 4: History Encoder → zh (Equation 7)
        # ------------------------------------------------------------------
        zh = self.history_encoder(z, te, event_valid=event_valid)  # (batch, 1+ne, d_model)
        # zh[:,0,:]   = [USR] token representation
        # zh[:,1:,:]  = [EVT] token representations (event i is at position i+1)

        output: Dict[str, torch.Tensor] = {"zh": zh}

        # ------------------------------------------------------------------
        # Steps 5–6: Gather masked positions + MLM head (Equation 8)
        #            Only executed during pre-training when mask is provided.
        # ------------------------------------------------------------------
        if mask is not None:
            # Gather the three context vectors for each masked token position.
            # b_idx, i_idx, j_idx index (batch, event, token_within_event).
            b_idx, i_idx, j_idx = mask.nonzero(as_tuple=True)

            # Local context: Event Encoder token output at masked position j in event i
            z_hat_e_ij = z_hat_e[b_idx, i_idx, j_idx, :]   # (n_masked, d_model)

            # Cross-event context: History Encoder output for event i.
            # Event i is at position i+1 in zh because [USR] occupies position 0.
            zh_i = zh[b_idx, i_idx + 1, :]                  # (n_masked, d_model)

            # User-level context: History Encoder [USR] token (position 0 in zh).
            zh_0 = zh[b_idx, 0, :]                          # (n_masked, d_model)

            # MLM head: cat([z_hat_e_ij, zh_i, zh_0]) → logits (Equation 8)
            logits = self.mlm_head.forward(z_hat_e_ij, zh_i, zh_0)
            output["logits"] = logits                        # (n_masked, value_vocab_size)

        return output
