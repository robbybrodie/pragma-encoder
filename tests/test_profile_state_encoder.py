"""Tests for ProfileStateEncoder.

Derived from PRAGMA paper Section 2.3.2:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:    verify forward(xa, ta) output shapes
  Math tests:     verify RoPE application, bidirectional attention, determinism
  Gradient tests: verify all parameters receive gradients; no nn.Embedding
  Spec tests:     verify config-driven construction, layer count, no [USR] param

Key paper properties (§2.3.2):
  - Input:  xa (batch, na, d_model) — pre-embedded; [USR] already at position 0
  - Input:  ta (batch, na) — temporal coordinates (log-seconds, Equation 2)
  - Output: za (batch, na, d_model) — full encoder output; caller slices [USR]
  - Bidirectional attention — NEVER causal
  - RoPE applied to ta in every attention layer

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
nn = torch.nn

from src.model.config import PRAGMAConfig
from src.encoders.profile_state_encoder import ProfileStateEncoder

_CONFIG = PRAGMAConfig.pragma_s()


# ---------------------------------------------------------------------------
# TestShapes — verify forward(xa, ta) output shapes
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify ProfileStateEncoder input/output shapes (§2.3.2, Equation 4)."""

    def test_output_shape_full_sequence(self) -> None:
        """§2.3.2 / Eq 4: forward() returns za (batch, na, d_model) — full sequence.

        The encoder returns the complete output sequence, not just position 0.
        The caller extracts za[:,0:1,:] for the [USR] token externally.
        Returning (batch, d_model) or (batch, 1, d_model) is a deviation.
        """
        encoder = ProfileStateEncoder(_CONFIG)
        batch, na = 2, 6
        xa = torch.randn(batch, na, _CONFIG.d_model)
        ta = torch.zeros(batch, na)

        za = encoder(xa, ta)

        assert za.shape == (batch, na, _CONFIG.d_model), (
            f"§2.3.2: za must be (batch, na, d_model) = ({batch}, {na}, {_CONFIG.d_model}), "
            f"got {tuple(za.shape)}"
        )

    def test_accepts_float_embeddings_not_token_ids(self) -> None:
        """§2.3.2 / Eq 1: xa is pre-embedded float tensor, NOT integer token IDs.

        Embedding (Equation 1: x = PosEmb(E(k) + E(v))) happens before the
        encoder in the tokeniser pipeline. ProfileStateEncoder must accept
        float embeddings directly, not raw integer token IDs.
        """
        encoder = ProfileStateEncoder(_CONFIG)
        batch, na = 1, 4
        xa = torch.randn(batch, na, _CONFIG.d_model)  # float embeddings
        ta = torch.zeros(batch, na)

        za = encoder(xa, ta)

        assert za.dtype == torch.float32, (
            f"Output must be float32, got {za.dtype}"
        )
        assert za.shape == (batch, na, _CONFIG.d_model)

    def test_usr_token_sliceable_externally(self) -> None:
        """§2.3.2 / Eq 4: za[:,0:1,:] is (batch, 1, d_model) — sliced by caller.

        The [USR] token is at position 0 of za. The History Encoder receives
        za[:,0:1,:] — this slicing is the CALLER's responsibility, not the
        encoder's. The encoder returns the full za sequence.
        """
        encoder = ProfileStateEncoder(_CONFIG)
        batch, na = 2, 5
        xa = torch.randn(batch, na, _CONFIG.d_model)
        ta = torch.zeros(batch, na)

        za = encoder(xa, ta)
        usr_token = za[:, 0:1, :]  # caller slices position 0

        assert usr_token.shape == (batch, 1, _CONFIG.d_model), (
            f"§2.3.2: [USR] token slice must be (batch, 1, d_model) = "
            f"({batch}, 1, {_CONFIG.d_model}), got {tuple(usr_token.shape)}"
        )


# ---------------------------------------------------------------------------
# TestMathProperties — verify RoPE application, bidirectionality, determinism
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from §2.3.2."""

    def test_rope_affects_output(self) -> None:
        """§2.3.2 / Eq 9: RoPE is actually applied — different ta → different za.

        RoPE rotates Q and K based on temporal coordinates ta. If ta changes,
        the rotation angles change, the attention scores change, and za changes.
        If za is unchanged, RoPE is not being applied in forward().
        """
        encoder = ProfileStateEncoder(_CONFIG).eval()
        batch, na = 1, 4
        xa = torch.randn(batch, na, _CONFIG.d_model)

        # Temporal coordinates from Equation 2: t' = 8·ln(1+t/8)
        ta_zero    = torch.zeros(batch, na)
        ta_nonzero = torch.tensor([[0.0, 5.545, 11.09, 21.97]])

        with torch.no_grad():
            za_zero    = encoder(xa, ta_zero)
            za_nonzero = encoder(xa, ta_nonzero)

        assert not torch.allclose(za_zero, za_nonzero, atol=1e-6), (
            "§2.3.2: RoPE must produce different outputs for different "
            "temporal coordinates ta. If za is unchanged, RoPE is not applied."
        )

    def test_bidirectional_attention(self) -> None:
        """§2.3.2 / CLAUDE.md: attention is bidirectional — NO causal mask.

        In bidirectional attention, position 0 (the [USR] token) can attend
        to ALL other positions, including the last one. Changing xa at the
        last position MUST change za at position 0.

        In causal attention, position 0 can only attend to itself — changing
        the last position would leave za[:,0,:] unchanged.
        This test distinguishes bidirectional from causal.
        """
        encoder = ProfileStateEncoder(_CONFIG).eval()
        batch, na = 1, 6
        ta = torch.zeros(batch, na)

        torch.manual_seed(0)
        xa_base = torch.randn(batch, na, _CONFIG.d_model)

        # Modify ONLY the last token
        xa_modified = xa_base.clone()
        xa_modified[:, -1, :] = torch.randn(batch, _CONFIG.d_model)

        with torch.no_grad():
            za_base     = encoder(xa_base, ta)
            za_modified = encoder(xa_modified, ta)

        assert not torch.allclose(za_base[:, 0, :], za_modified[:, 0, :], atol=1e-6), (
            "§2.3.2: ProfileStateEncoder must use bidirectional attention. "
            "Changing the last token must change the [USR] output at position 0. "
            "If za[:,0,:] is unchanged, attention is causal (wrong)."
        )

    def test_output_is_deterministic_in_eval_mode(self) -> None:
        """§2.3.2: same inputs → same outputs in eval mode (dropout disabled).

        Dropout must be disabled when encoder.eval() is called. Two forward
        passes with the same inputs must produce identical outputs.
        """
        encoder = ProfileStateEncoder(_CONFIG).eval()
        batch, na = 2, 4
        xa = torch.randn(batch, na, _CONFIG.d_model)
        ta = torch.zeros(batch, na)

        with torch.no_grad():
            za1 = encoder(xa, ta)
            za2 = encoder(xa, ta)

        assert torch.allclose(za1, za2), (
            "ProfileStateEncoder output must be deterministic in eval mode. "
            "Dropout must be disabled via encoder.eval()."
        )


# ---------------------------------------------------------------------------
# TestGradientFlow — all parameters trained; no embedding table
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradient flow and absence of embedding table (§2.3.2)."""

    def test_all_parameters_receive_gradients(self) -> None:
        """§2.3.2: every trainable parameter must receive gradients.

        Frozen or detached parameters mean parts of the encoder are not
        being trained. This fails silently without this test.
        """
        encoder = ProfileStateEncoder(_CONFIG)
        batch, na = 2, 4
        xa = torch.randn(batch, na, _CONFIG.d_model)
        ta = torch.zeros(batch, na)

        za = encoder(xa, ta)
        za.sum().backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient — it is not being trained"
            )
            assert not torch.all(param.grad == 0), (
                f"Parameter '{name}' has all-zero gradients"
            )

    def test_no_embedding_table(self) -> None:
        """§2.3.2 / Eq 1: ProfileStateEncoder must contain NO nn.Embedding.

        Token embedding (Equation 1: x = PosEmb(E(k) + E(v))) is performed
        externally by the tokeniser pipeline before the encoder is called.
        An nn.Embedding inside the encoder means the encoder is re-embedding
        already-embedded inputs — a silent double-embedding bug.
        """
        encoder = ProfileStateEncoder(_CONFIG)

        for name, module in encoder.named_modules():
            assert not isinstance(module, nn.Embedding), (
                f"ProfileStateEncoder must not contain nn.Embedding "
                f"(found at '{name}'). Embedding is done externally via Equation 1."
            )


# ---------------------------------------------------------------------------
# TestPaperSpecifications — config-driven construction, layer count, [USR]
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact values from the paper (all sourced from key-numbers.md)."""

    def test_constructor_takes_config(self) -> None:
        """DEVELOPMENT_PROCESS.md: constructor must be (self, config: PRAGMAConfig).

        No individual arguments (vocab_size, d_model, n_heads, etc.).
        Scaling from PRAGMA-S to PRAGMA-M must require changing exactly one line.
        """
        # Must construct with config only — no other required args
        encoder_s = ProfileStateEncoder(PRAGMAConfig.pragma_s())
        encoder_m = ProfileStateEncoder(PRAGMAConfig.pragma_m())
        encoder_l = ProfileStateEncoder(PRAGMAConfig.pragma_l())

        assert encoder_s.d_model == 192   # key-numbers.md: Table 1
        assert encoder_m.d_model == 512   # key-numbers.md: Table 1
        assert encoder_l.d_model == 1024  # key-numbers.md: Table 1

    def test_pragma_s_uses_one_encoder_layer(self) -> None:
        """key-numbers.md: profile_encoder_layers = 1 for PRAGMA-S (Table 1).

        The encoder stack must have exactly 1 layer for PRAGMA-S,
        driven by config.profile_encoder_layers, not hardcoded.
        """
        config = PRAGMAConfig.pragma_s()
        assert config.profile_encoder_layers == 1  # key-numbers.md: Table 1

        encoder = ProfileStateEncoder(config)
        assert len(encoder.layers) == 1, (
            f"key-numbers.md: PRAGMA-S must have 1 profile encoder layer, "
            f"got {len(encoder.layers)}"
        )

    def test_no_usr_token_parameter(self) -> None:
        """§2.3.2: ProfileStateEncoder must NOT have a self.usr_token parameter.

        The [USR] token is prepended to xa by the caller BEFORE forward() is
        called. The encoder processes it as position 0 of xa — it does not
        create or inject the [USR] token itself.
        """
        encoder = ProfileStateEncoder(_CONFIG)

        for name, _ in encoder.named_parameters():
            assert "usr_token" not in name, (
                f"ProfileStateEncoder must not have a usr_token parameter "
                f"(found '{name}'). The [USR] token is prepended externally in xa."
            )

    def test_d_model_from_config(self) -> None:
        """key-numbers.md: d_model = 192 / 512 / 1024, Table 1.

        The hidden dimension must come from config.d_model, not hardcoded.
        The encoder output dimension must match config.d_model exactly.
        """
        for config in [
            PRAGMAConfig.pragma_s(),
            PRAGMAConfig.pragma_m(),
            PRAGMAConfig.pragma_l(),
        ]:
            encoder = ProfileStateEncoder(config)
            batch, na = 1, 4
            xa = torch.randn(batch, na, config.d_model)
            ta = torch.zeros(batch, na)

            with torch.no_grad():
                za = encoder(xa, ta)

            assert za.shape[-1] == config.d_model, (  # key-numbers.md: Table 1
                f"key-numbers.md: output d_model must be {config.d_model} "
                f"for {config.model_name}, got {za.shape[-1]}"
            )
