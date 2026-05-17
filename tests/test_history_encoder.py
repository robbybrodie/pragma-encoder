"""Tests for HistoryEncoder.

Derived from PRAGMA paper Section 2.3.4:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:    verify forward(z, te) output shapes
  Math tests:     verify RoPE application, bidirectionality, [USR] conditioning
  Gradient tests: verify all parameters receive gradients; no hist_token; no mask_embedding
  Spec tests:     verify config-driven construction, layer count, no cross-attention

Key paper properties (§2.3.4):
  - Input: z (batch, 1+ne, d_model) — [USR:EVT] assembled by caller (Equation 6)
  - Input: te (batch, 1+ne) — temporal coords (0 for [USR], log-secs for [EVT])
  - Output: zh (batch, 1+ne, d_model) — full encoder output, unsliced (Equation 7)
  - Bidirectional attention — NEVER causal
  - RoPE applied to te in every attention layer (Equation 9)
  - Pure self-attention — NO cross-attention sublayer
  - [USR] at position 0 conditions [EVT] through self-attention naturally

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
nn = torch.nn

from src.model.config import PRAGMAConfig
from src.encoders.history_encoder import HistoryEncoder

_CONFIG = PRAGMAConfig.pragma_s()


# ---------------------------------------------------------------------------
# TestShapes — verify forward(z, te) output shapes
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify HistoryEncoder input/output shapes (§2.3.4, Equations 6 and 7)."""

    def test_zh_shape(self) -> None:
        """§2.3.4 / Eq 7: zh must be (batch, 1+ne, d_model) — full sequence, unsliced.

        The encoder returns the complete output sequence. The caller is responsible
        for extracting zh[:,0,:] ([USR]) or zh[:,1:,:] ([EVT] tokens).
        Returning a sliced or squeezed tensor is a deviation from Equation 7.
        """
        encoder = HistoryEncoder(_CONFIG)
        batch, ne = 2, 8
        z = torch.randn(batch, 1 + ne, _CONFIG.d_model)
        te = torch.zeros(batch, 1 + ne)

        zh = encoder(z, te)

        assert zh.shape == (batch, 1 + ne, _CONFIG.d_model), (
            f"§2.3.4 / Eq 7: zh must be (batch, 1+ne, d_model) = "
            f"({batch}, {1+ne}, {_CONFIG.d_model}), got {tuple(zh.shape)}"
        )

    def test_usr_token_sliceable(self) -> None:
        """§2.3.4 / Eq 7: zh[:,0:1,:] is (batch, 1, d_model) — [USR] output.

        The [USR] token output is at position 0 of zh. The History Encoder
        does NOT slice it — that is the caller's responsibility.
        zh[:,0,:] is used by the embedding probe and fine-tuning head.
        """
        encoder = HistoryEncoder(_CONFIG)
        batch, ne = 2, 5
        z = torch.randn(batch, 1 + ne, _CONFIG.d_model)
        te = torch.zeros(batch, 1 + ne)

        zh = encoder(z, te)
        usr_out = zh[:, 0:1, :]

        assert usr_out.shape == (batch, 1, _CONFIG.d_model), (
            f"§2.3.4: [USR] output slice must be (batch, 1, d_model) = "
            f"({batch}, 1, {_CONFIG.d_model}), got {tuple(usr_out.shape)}"
        )

    def test_evt_tokens_sliceable(self) -> None:
        """§2.3.4 / Eq 7: zh[:,1:,:] is (batch, ne, d_model) — [EVT] outputs.

        The [EVT] token outputs are at positions 1..ne of zh. The caller
        uses zh[:,1:,:] to feed the MLM head and downstream probes.
        """
        encoder = HistoryEncoder(_CONFIG)
        batch, ne = 2, 6
        z = torch.randn(batch, 1 + ne, _CONFIG.d_model)
        te = torch.zeros(batch, 1 + ne)

        zh = encoder(z, te)
        evt_out = zh[:, 1:, :]

        assert evt_out.shape == (batch, ne, _CONFIG.d_model), (
            f"§2.3.4: [EVT] output slice must be (batch, ne, d_model) = "
            f"({batch}, {ne}, {_CONFIG.d_model}), got {tuple(evt_out.shape)}"
        )


# ---------------------------------------------------------------------------
# TestMathProperties — RoPE, bidirectionality, [USR] conditioning
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from §2.3.4."""

    def test_rope_affects_output(self) -> None:
        """§2.3.4 / Eq 9: RoPE is applied — different te → different zh.

        The HistoryEncoder applies RoPE to te in every attention layer.
        Changing te changes the rotation angles, which changes attention scores,
        which changes zh. If zh is unchanged, RoPE is not being applied.
        """
        encoder = HistoryEncoder(_CONFIG).eval()
        batch, ne = 1, 5
        z = torch.randn(batch, 1 + ne, _CONFIG.d_model)

        # te[:,0] = 0.0 ([USR] position, always 0 per §2.3.4)
        # te[:,1:] = log-seconds to most recent event (Equation 2)
        te_zero = torch.zeros(batch, 1 + ne)
        te_nonzero = torch.tensor([[0.0, 5.545, 11.09, 16.64, 21.97, 27.31]])

        with torch.no_grad():
            zh_zero = encoder(z, te_zero)
            zh_nonzero = encoder(z, te_nonzero)

        assert not torch.allclose(zh_zero, zh_nonzero, atol=1e-6), (
            "§2.3.4: RoPE must produce different outputs for different temporal "
            "coordinates te. If zh is unchanged, RoPE is not applied."
        )

    def test_bidirectional_attention(self) -> None:
        """§2.3.4 / CLAUDE.md: attention is bidirectional — NO causal mask.

        In bidirectional attention, position 0 (the [USR] token) can attend
        to ALL positions, including the last [EVT]. Changing the last [EVT]
        token MUST change zh at position 0.

        In causal attention, position 0 attends only to itself — changing
        the last position would leave zh[:,0,:] unchanged.
        """
        encoder = HistoryEncoder(_CONFIG).eval()
        batch, ne = 1, 6
        te = torch.zeros(batch, 1 + ne)

        torch.manual_seed(0)
        z_base = torch.randn(batch, 1 + ne, _CONFIG.d_model)

        # Modify ONLY the last [EVT] token (position ne)
        z_modified = z_base.clone()
        z_modified[:, -1, :] = torch.randn(_CONFIG.d_model)

        with torch.no_grad():
            zh_base = encoder(z_base, te)
            zh_modified = encoder(z_modified, te)

        assert not torch.allclose(zh_base[:, 0, :], zh_modified[:, 0, :], atol=1e-6), (
            "§2.3.4: HistoryEncoder must use bidirectional attention. "
            "Changing the last [EVT] token must change [USR] output at position 0. "
            "If zh[:,0,:] is unchanged, attention is causal (wrong)."
        )

    def test_usr_conditions_evt_representations(self) -> None:
        """§2.3.4: changing the [USR] token (position 0) changes [EVT] outputs.

        The [USR] token from ProfileStateEncoder is at position 0 of z.
        In bidirectional self-attention, [EVT] tokens at positions 1+ attend
        to [USR] at position 0 — profile information conditions history encoding.

        This is the mechanism that makes PRAGMA profile-conditioned:
        not separate cross-attention (stub was wrong), but self-attention
        over the concatenated [USR:EVT] sequence.
        """
        encoder = HistoryEncoder(_CONFIG).eval()
        batch, ne = 1, 5
        te = torch.zeros(batch, 1 + ne)

        torch.manual_seed(1)
        z_base = torch.randn(batch, 1 + ne, _CONFIG.d_model)

        # Modify ONLY position 0 — the [USR] token
        z_modified = z_base.clone()
        z_modified[:, 0, :] = torch.randn(_CONFIG.d_model)

        with torch.no_grad():
            zh_base = encoder(z_base, te)
            zh_modified = encoder(z_modified, te)

        # [EVT] positions (1+) must change when [USR] (position 0) changes
        assert not torch.allclose(zh_base[:, 1:, :], zh_modified[:, 1:, :], atol=1e-6), (
            "§2.3.4: Changing the [USR] input (position 0) must change [EVT] outputs "
            "(positions 1+). [USR] conditions the history through bidirectional "
            "self-attention — this is the core profile-conditioning mechanism."
        )

    def test_deterministic_in_eval_mode(self) -> None:
        """§2.3.4: same inputs → same outputs in eval mode (dropout disabled)."""
        encoder = HistoryEncoder(_CONFIG).eval()
        batch, ne = 2, 4
        z = torch.randn(batch, 1 + ne, _CONFIG.d_model)
        te = torch.zeros(batch, 1 + ne)

        with torch.no_grad():
            zh1 = encoder(z, te)
            zh2 = encoder(z, te)

        assert torch.allclose(zh1, zh2), (
            "HistoryEncoder output must be deterministic in eval mode. "
            "Dropout must be disabled via encoder.eval()."
        )


# ---------------------------------------------------------------------------
# TestGradientFlow — all parameters trained; no prohibited parameters
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradient flow and absence of prohibited parameters (§2.3.4)."""

    def test_all_parameters_receive_gradients(self) -> None:
        """§2.3.4: every trainable parameter must receive gradients.

        All Transformer layers (attention + FFN) and the final LayerNorm
        must be trained. Frozen or detached parameters fail silently.
        """
        encoder = HistoryEncoder(_CONFIG)
        batch, ne = 2, 4
        z = torch.randn(batch, 1 + ne, _CONFIG.d_model)
        te = torch.zeros(batch, 1 + ne)

        zh = encoder(z, te)
        zh.sum().backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient — it is not being trained"
            )
            assert not torch.all(param.grad == 0), (
                f"Parameter '{name}' has all-zero gradients"
            )

    def test_no_hist_token_parameter(self) -> None:
        """§2.3.4: HistoryEncoder must NOT have a self.hist_token parameter.

        There is no [HIST] token in the PRAGMA paper. The input z is already
        assembled as z = [za : ze] (Equation 6) by the caller, with [USR] at
        position 0. The encoder does NOT prepend any extra token.
        """
        encoder = HistoryEncoder(_CONFIG)

        for name, _ in encoder.named_parameters():
            assert "hist_token" not in name, (
                f"HistoryEncoder must not have a hist_token parameter (found '{name}'). "
                f"z is assembled by the caller with [USR] already at position 0 (Eq 6)."
            )

    def test_no_mask_embedding_parameter(self) -> None:
        """§2.3.4: HistoryEncoder must NOT have a self.mask_embedding parameter.

        Masking is handled externally by MaskingStrategy before forward() is
        called. The HistoryEncoder receives already-masked input — it does not
        replace events with a mask embedding itself.
        """
        encoder = HistoryEncoder(_CONFIG)

        for name, _ in encoder.named_parameters():
            assert "mask" not in name.lower() or "norm" in name.lower(), (
                f"HistoryEncoder must not have a mask embedding parameter "
                f"(found '{name}'). Masking is done externally by MaskingStrategy."
            )


# ---------------------------------------------------------------------------
# TestPaperSpecifications — exact values from key-numbers.md
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact values from the paper (all sourced from key-numbers.md)."""

    def test_constructor_takes_config(self) -> None:
        """DEVELOPMENT_PROCESS.md: constructor must be (self, config: PRAGMAConfig).

        No individual arguments. Scaling must require changing exactly one line.
        """
        encoder_s = HistoryEncoder(PRAGMAConfig.pragma_s())
        encoder_m = HistoryEncoder(PRAGMAConfig.pragma_m())
        encoder_l = HistoryEncoder(PRAGMAConfig.pragma_l())

        assert encoder_s.d_model == 192   # key-numbers.md: Table 1
        assert encoder_m.d_model == 512   # key-numbers.md: Table 1
        assert encoder_l.d_model == 1024  # key-numbers.md: Table 1

    def test_pragma_s_uses_two_history_encoder_layers(self) -> None:
        """key-numbers.md: history_encoder_layers = 2 for PRAGMA-S (Table 1)."""
        config = PRAGMAConfig.pragma_s()
        assert config.history_encoder_layers == 2  # key-numbers.md: Table 1

        encoder = HistoryEncoder(config)
        assert len(encoder.layers) == 2, (
            f"key-numbers.md: PRAGMA-S must have 2 history encoder layers, "
            f"got {len(encoder.layers)}"
        )

    def test_d_model_from_config(self) -> None:
        """key-numbers.md: d_model = 192 / 512 / 1024, Table 1.

        The output dimension must match config.d_model for all three sizes.
        """
        for config in [
            PRAGMAConfig.pragma_s(),
            PRAGMAConfig.pragma_m(),
            PRAGMAConfig.pragma_l(),
        ]:
            encoder = HistoryEncoder(config)
            batch, ne = 1, 3
            z = torch.randn(batch, 1 + ne, config.d_model)
            te = torch.zeros(batch, 1 + ne)

            with torch.no_grad():
                zh = encoder(z, te)

            assert zh.shape[-1] == config.d_model, (  # key-numbers.md: Table 1
                f"key-numbers.md: output d_model must be {config.d_model} "
                f"for {config.model_name}, got {zh.shape[-1]}"
            )

    def test_no_cross_attention_sublayer(self) -> None:
        """§2.3.4: HistoryEncoder uses pure self-attention — NO cross-attention.

        The paper specifies a bidirectional Transformer (same as ProfileStateEncoder
        and EventEncoder). There is no separate cross-attention sub-layer.
        The [USR] token at position 0 of z naturally conditions the [EVT] tokens
        through bidirectional self-attention.

        A cross_attn sub-layer would be an extra mechanism not in the paper —
        it would add parameters and a norm not accounted for by the architecture.
        """
        encoder = HistoryEncoder(_CONFIG)

        for name, _ in encoder.named_parameters():
            assert "cross_attn" not in name, (
                f"HistoryEncoder must not have a cross_attn parameter (found '{name}'). "
                f"§2.3.4 specifies pure self-attention. [USR] conditions [EVT] "
                f"tokens through bidirectional self-attention over z = [za : ze]."
            )
