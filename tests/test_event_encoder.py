"""Tests for EventEncoder.

Derived from PRAGMA paper Section 2.3.3:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:    verify forward(xe, xt) output shapes (z_hat_e and ze)
  Math tests:     verify independence constraint, bidirectionality, calendar placement
  Gradient tests: verify all parameters receive gradients; no nn.Embedding; no evt_token
  Spec tests:     verify config-driven construction, layer count, calendar dims

Key paper properties (§2.3.3):
  - Input:  xe (batch, ne, ni, d_model) — pre-embedded; [EVT] already at position 0
  - Input:  xt (batch, ne, 3) — calendar features (hour, day_of_week, day_of_month)
  - Output: z_hat_e (batch, ne, ni, d_model) — full token-level encoder output
  - Output: ze (batch, ne, d_model) — calendar-augmented [EVT] tokens (ze = z'e + zt)
  - Each event processed independently — no cross-event attention
  - Calendar (Equation 3): sincos → 2-layer MLP → zt; added AFTER encoder
  - Bidirectional attention within each event — NEVER causal

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
nn = torch.nn

from pragma_encoder.encoders.event_encoder import EventEncoder
from pragma_encoder.model.config import PRAGMAConfig

_CONFIG = PRAGMAConfig.pragma_s()


# ---------------------------------------------------------------------------
# TestShapes — verify forward(xe, xt) output shapes
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify EventEncoder input/output shapes (§2.3.3, Equations 3 and 5)."""

    def test_z_hat_e_shape(self) -> None:
        """§2.3.3 / Eq 5: z_hat_e must be (batch, ne, ni, d_model).

        The full token-level encoder output is returned for the MLM head.
        It is NOT sliced inside the encoder — the caller extracts [EVT] tokens.
        """
        encoder = EventEncoder(_CONFIG)
        batch, ne, ni = 2, 5, _CONFIG.max_event_tokens  # ni=24, key-numbers.md §2.4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        z_hat_e, ze = encoder(xe, xt)

        assert z_hat_e.shape == (batch, ne, ni, _CONFIG.d_model), (
            f"§2.3.3: z_hat_e must be (batch, ne, ni, d_model) = "
            f"({batch}, {ne}, {ni}, {_CONFIG.d_model}), got {tuple(z_hat_e.shape)}"
        )

    def test_ze_shape(self) -> None:
        """§2.3.3 / Eq 3: ze must be (batch, ne, d_model).

        ze = z'e + zt where z'e is the [EVT] token and zt is the calendar MLP
        output. The result aggregates each event into a single d_model vector.
        """
        encoder = EventEncoder(_CONFIG)
        batch, ne, ni = 2, 5, _CONFIG.max_event_tokens
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        z_hat_e, ze = encoder(xe, xt)

        assert ze.shape == (batch, ne, _CONFIG.d_model), (
            f"§2.3.3: ze must be (batch, ne, d_model) = "
            f"({batch}, {ne}, {_CONFIG.d_model}), got {tuple(ze.shape)}"
        )

    def test_accepts_float_embeddings(self) -> None:
        """§2.3.3 / Eq 1: xe is pre-embedded float tensor, NOT integer token IDs.

        Embedding (Equation 1: x = PosEmb(E(k) + E(v))) is done externally by
        the tokeniser pipeline. EventEncoder must accept float embeddings.
        """
        encoder = EventEncoder(_CONFIG)
        batch, ne, ni = 1, 3, 4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)  # float embeddings
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        z_hat_e, ze = encoder(xe, xt)

        assert z_hat_e.dtype == torch.float32, (
            f"z_hat_e must be float32, got {z_hat_e.dtype}"
        )
        assert ze.dtype == torch.float32, (
            f"ze must be float32, got {ze.dtype}"
        )

    def test_evt_token_at_position_zero(self) -> None:
        """§2.3.3 / Eq 5: z_hat_e[:,:,0,:] is (batch, ne, d_model) — sliceable.

        The [EVT] token is at position 0 of each event's output in z_hat_e.
        The caller extracts it as z'e = z_hat_e[:,:,0,:] — NOT the encoder's job.
        """
        encoder = EventEncoder(_CONFIG)
        batch, ne, ni = 2, 4, 6
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        z_hat_e, _ = encoder(xe, xt)
        evt_tokens = z_hat_e[:, :, 0, :]  # slice [EVT] at position 0

        assert evt_tokens.shape == (batch, ne, _CONFIG.d_model), (
            f"§2.3.3: [EVT] token slice must be (batch, ne, d_model) = "
            f"({batch}, {ne}, {_CONFIG.d_model}), got {tuple(evt_tokens.shape)}"
        )


# ---------------------------------------------------------------------------
# TestMathProperties — independence, bidirectionality, calendar placement
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from §2.3.3."""

    def test_calendar_affects_ze_not_z_hat_e(self) -> None:
        """§2.3.3 / Eq 3: calendar xt affects ze but NOT z_hat_e.

        The calendar MLP runs AFTER the encoder:
          z_hat_e = EventEncoder(xe)         — depends on xe only
          zt = CalendarMLP(sincos(xt))       — depends on xt only
          ze = z_hat_e[:,:,0,:] + zt         — sum of both

        If xt changes:
          - ze must change (because zt changes)
          - z_hat_e must NOT change (because z_hat_e doesn't depend on xt)

        Failure here means the calendar was incorrectly added BEFORE or INSIDE
        the encoder (as in the stub), not after it.
        """
        encoder = EventEncoder(_CONFIG).eval()
        batch, ne, ni = 1, 4, 6
        torch.manual_seed(1)
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)

        # Two different calendar inputs — same xe
        xt_a = torch.zeros(batch, ne, 3, dtype=torch.long)
        xt_b = torch.full((batch, ne, 3), 10, dtype=torch.long)

        with torch.no_grad():
            z_hat_e_a, ze_a = encoder(xe, xt_a)
            z_hat_e_b, ze_b = encoder(xe, xt_b)

        assert torch.allclose(z_hat_e_a, z_hat_e_b, atol=1e-6), (
            "§2.3.3: z_hat_e must not depend on xt (calendar added after encoder). "
            "If z_hat_e changes when xt changes, calendar is inside the encoder (wrong)."
        )
        assert not torch.allclose(ze_a, ze_b, atol=1e-6), (
            "§2.3.3: ze must depend on xt via ze = z'e + zt. "
            "If ze is unchanged, the calendar MLP output zt is not being added."
        )

    def test_independence_across_events(self) -> None:
        """§2.3.3: each event is processed independently — no cross-event attention.

        Reshaping xe to (batch*ne, ni, d_model) before the encoder ensures each
        event is a separate batch item — no attention tokens cross event boundaries.

        Changing event i must:
          - Change output for event i (attending to its own modified tokens)
          - Leave output for events j≠i UNCHANGED (no cross-event attention)

        This is the independence constraint from §2.3.3.
        """
        encoder = EventEncoder(_CONFIG).eval()
        batch, ne, ni = 1, 3, 5
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        torch.manual_seed(2)
        xe_base = torch.randn(batch, ne, ni, _CONFIG.d_model)

        # Modify ONLY event 1 (middle event)
        xe_modified = xe_base.clone()
        xe_modified[:, 1, :, :] = torch.randn(batch, ni, _CONFIG.d_model)

        with torch.no_grad():
            z_hat_e_base, _ = encoder(xe_base, xt)
            z_hat_e_mod, _ = encoder(xe_modified, xt)

        # Event 0: must be UNCHANGED
        assert torch.allclose(z_hat_e_base[:, 0, :, :], z_hat_e_mod[:, 0, :, :], atol=1e-6), (
            "§2.3.3: Independence constraint violated. "
            "Modifying event 1 changed event 0 output — cross-event attention present."
        )
        # Event 1: must CHANGE
        assert not torch.allclose(z_hat_e_base[:, 1, :, :], z_hat_e_mod[:, 1, :, :], atol=1e-6), (
            "§2.3.3: Event 1 must change when its tokens are modified."
        )
        # Event 2: must be UNCHANGED
        assert torch.allclose(z_hat_e_base[:, 2, :, :], z_hat_e_mod[:, 2, :, :], atol=1e-6), (
            "§2.3.3: Independence constraint violated. "
            "Modifying event 1 changed event 2 output — cross-event attention present."
        )

    def test_bidirectional_attention_within_event(self) -> None:
        """§2.3.3 / CLAUDE.md: attention within each event is bidirectional.

        In bidirectional attention, position 0 (the [EVT] token) can attend
        to ALL tokens in the event, including the last one. Changing the last
        token MUST change the [EVT] output at position 0.

        Also verifies independence: modifying event 0's last token must NOT
        affect event 1's output.
        """
        encoder = EventEncoder(_CONFIG).eval()
        batch, ne, ni = 1, 2, 6
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        torch.manual_seed(3)
        xe_base = torch.randn(batch, ne, ni, _CONFIG.d_model)

        # Modify ONLY the last token of event 0
        xe_modified = xe_base.clone()
        xe_modified[:, 0, -1, :] = torch.randn(_CONFIG.d_model)

        with torch.no_grad():
            z_hat_e_base, _ = encoder(xe_base, xt)
            z_hat_e_mod, _ = encoder(xe_modified, xt)

        # [EVT] of event 0 must CHANGE (bidirectional: pos 0 attends to last token)
        assert not torch.allclose(
            z_hat_e_base[:, 0, 0, :], z_hat_e_mod[:, 0, 0, :], atol=1e-6
        ), (
            "§2.3.3: EventEncoder must use bidirectional attention. "
            "Changing the last token must change [EVT] output at position 0. "
            "If unchanged, attention is causal (wrong)."
        )
        # Event 1 must be UNCHANGED (independence)
        assert torch.allclose(z_hat_e_base[:, 1, :, :], z_hat_e_mod[:, 1, :, :], atol=1e-6), (
            "§2.3.3: Independence constraint — modifying event 0 must not affect event 1."
        )

    def test_output_is_deterministic_in_eval_mode(self) -> None:
        """§2.3.3: same inputs → same outputs in eval mode (dropout disabled).

        Dropout must be disabled when encoder.eval() is called.
        """
        encoder = EventEncoder(_CONFIG).eval()
        batch, ne, ni = 2, 3, 4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        with torch.no_grad():
            z_hat_e_1, ze_1 = encoder(xe, xt)
            z_hat_e_2, ze_2 = encoder(xe, xt)

        assert torch.allclose(z_hat_e_1, z_hat_e_2), (
            "EventEncoder z_hat_e must be deterministic in eval mode."
        )
        assert torch.allclose(ze_1, ze_2), (
            "EventEncoder ze must be deterministic in eval mode."
        )


# ---------------------------------------------------------------------------
# TestGradientFlow — all parameters trained; no embedding table; no evt_token
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradient flow and absence of prohibited parameters (§2.3.3)."""

    def test_all_parameters_receive_gradients(self) -> None:
        """§2.3.3: every trainable parameter must receive gradients.

        Both z_hat_e and ze contribute to the loss in pre-training.
        All parameters (Transformer layers + calendar MLP) must be trained.
        """
        encoder = EventEncoder(_CONFIG)
        batch, ne, ni = 2, 3, 4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        z_hat_e, ze = encoder(xe, xt)
        # Both outputs contribute — sum both to drive gradients through all paths
        (z_hat_e.sum() + ze.sum()).backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient — it is not being trained"
            )
            assert not torch.all(param.grad == 0), (
                f"Parameter '{name}' has all-zero gradients"
            )

    def test_no_embedding_table(self) -> None:
        """§2.3.3 / Eq 1: EventEncoder must contain NO nn.Embedding.

        Token embedding (Equation 1) is performed externally by the tokeniser
        pipeline. An nn.Embedding inside the encoder means double-embedding.
        Calendar features use sincos (fixed, periodic) — NOT nn.Embedding.
        """
        encoder = EventEncoder(_CONFIG)

        for name, module in encoder.named_modules():
            assert not isinstance(module, nn.Embedding), (
                f"EventEncoder must not contain nn.Embedding (found at '{name}'). "
                f"Token embedding is done externally. Calendar uses sincos, not nn.Embedding."
            )

    def test_no_evt_token_parameter(self) -> None:
        """§2.3.3: EventEncoder must NOT have a self.evt_token parameter.

        The [EVT] token is prepended to xe by the caller BEFORE forward() is
        called. The encoder receives it as position 0 of xe — it does not
        create or inject the [EVT] token itself.
        """
        encoder = EventEncoder(_CONFIG)

        for name, _ in encoder.named_parameters():
            assert "evt_token" not in name, (
                f"EventEncoder must not have an evt_token parameter (found '{name}'). "
                f"The [EVT] token is prepended externally at xe[:,:,0,:]."
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
        encoder_s = EventEncoder(PRAGMAConfig.pragma_s())
        encoder_m = EventEncoder(PRAGMAConfig.pragma_m())
        encoder_l = EventEncoder(PRAGMAConfig.pragma_l())

        assert encoder_s.d_model == 192   # key-numbers.md: Table 1
        assert encoder_m.d_model == 512   # key-numbers.md: Table 1
        assert encoder_l.d_model == 1024  # key-numbers.md: Table 1

    def test_pragma_s_uses_five_event_encoder_layers(self) -> None:
        """key-numbers.md: event_encoder_layers = 5 for PRAGMA-S (Table 1)."""
        config = PRAGMAConfig.pragma_s()
        assert config.event_encoder_layers == 5  # key-numbers.md: Table 1

        encoder = EventEncoder(config)
        assert len(encoder.layers) == 5, (
            f"key-numbers.md: PRAGMA-S must have 5 event encoder layers, "
            f"got {len(encoder.layers)}"
        )

    def test_calendar_feature_dims_is_3(self) -> None:
        """key-numbers.md: calendar_feature_dims=3 (hour, day_of_week, day_of_month), §2.2.

        xt has exactly 3 calendar values per event. The sincos step doubles
        these to 6 inputs for the first MLP layer. Verifying in_features=6
        confirms both the 3-dim input AND the sincos-before-MLP ordering.
        """
        encoder = EventEncoder(_CONFIG)

        # Verify the encoder accepts xt with exactly 3 calendar features
        batch, ne, ni = 1, 3, 4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)  # exactly 3 dims

        with torch.no_grad():
            _, ze = encoder(xe, xt)

        assert ze.shape == (batch, ne, _CONFIG.d_model)

        # sincos doubles 3 features to 6 before MLP — verify first linear layer
        first_linear = next(
            m for m in encoder.calendar_mlp.modules() if isinstance(m, nn.Linear)
        )
        assert first_linear.in_features == 6, (  # key-numbers.md: calendar_feature_dims=3
            f"key-numbers.md: calendar_feature_dims=3, sincos doubles to 6. "
            f"First MLP layer must have in_features=6 (= 3×2 from sincos), "
            f"got {first_linear.in_features}"
        )

    def test_calendar_mlp_has_two_layers(self) -> None:
        """key-numbers.md: calendar_feature_embedding_layers=2 (§2.3.3).

        The calendar MLP (Equation 3) is a 2-layer MLP:
          sincos(xt) → Linear → GELU → Linear → zt
        Exactly 2 Linear layers — not 1 (projection) or 3.
        """
        encoder = EventEncoder(_CONFIG)

        linear_layers = [
            m for m in encoder.calendar_mlp.modules()
            if isinstance(m, nn.Linear)
        ]
        assert len(linear_layers) == 2, (  # key-numbers.md: calendar_feature_embedding_layers=2
            f"key-numbers.md: calendar MLP must have exactly 2 Linear layers "
            f"(§2.3.3, Equation 3). Got {len(linear_layers)}."
        )


# ---------------------------------------------------------------------------
# TestCalendarPeriodAlignment — DEF-002: sin/cos must use known cycle periods
# ---------------------------------------------------------------------------


class TestCalendarPeriodAlignment:
    """DEF-002 — calendar sincos must normalise by known cycle periods.

    Equation 3 (§2.3.3, key-numbers.md):
        "Periods fixed to known calendar cycles (not learned)."

    Correct normalisation (2π·value/period):
        hour of day   → period=24:  sin(2π·h/24),   cos(2π·h/24)
        day of week   → period=7:   sin(2π·d/7),    cos(2π·d/7)
        day of month  → period=31:  sin(2π·dom/31), cos(2π·dom/31)

    Key property: period-equivalent inputs must produce identical calendar
    embeddings zt (and therefore identical ze, since ze = z'e + zt and
    z'e does not depend on xt).

    Period equivalences used in tests (all mod their cycle period = 0):
        hour=0  ≡ hour=24    (period 24)
        dow=0   ≡ dow=7      (period 7)
        dom=0   ≡ dom=31     (period 31)

    Buggy behaviour (raw sin/cos on integers):
        sin(0)=0, sin(24)≈-0.905 → xt=[0,0,0] and xt=[24,7,31] produce
        different sincos values → different zt → ze_a ≠ ze_b.

    Correct behaviour (period-normalised):
        sin(2π·0/24)=sin(0)=0 and sin(2π·24/24)=sin(2π)=0 → same sincos
        → same zt → ze_a == ze_b.
    """

    def test_hour_period_equivalence(self) -> None:
        """hour=0 and hour=24 are period-equivalent — ze must be identical.

        Period=24 → sin(2π·0/24)=sin(2π·24/24)=0.  (key-numbers.md §2.3.3)
        Buggy code: sin(0)=0 ≠ sin(24)≈-0.905 → ze_a ≠ ze_b (test FAILS).
        Fixed code: sin(0)=sin(2π)=0        → ze_a == ze_b (test PASSES).
        """
        encoder = EventEncoder(_CONFIG).eval()
        torch.manual_seed(42)
        batch, ne, ni = 1, 3, 4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)

        # hour=0 and hour=24 are equivalent modulo 24
        xt_a = torch.tensor([[[0, 0, 0]] * ne], dtype=torch.long)   # hour=0
        xt_b = torch.tensor([[[24, 0, 0]] * ne], dtype=torch.long)  # hour=24

        with torch.no_grad():
            _, ze_a = encoder(xe, xt_a)
            _, ze_b = encoder(xe, xt_b)

        assert torch.allclose(ze_a, ze_b, atol=1e-5), (
            "DEF-002: hour=0 and hour=24 must produce identical ze. "
            "Calendar sincos must normalise by period=24 (2π·h/24). "
            f"Max diff: {(ze_a - ze_b).abs().max().item():.6f}"
        )

    def test_dow_period_equivalence(self) -> None:
        """dow=0 and dow=7 are period-equivalent — ze must be identical.

        Period=7 → sin(2π·0/7)=sin(2π·7/7)=0.  (key-numbers.md §2.3.3)
        Buggy code: sin(0)=0 ≠ sin(7)≈0.657 → ze_a ≠ ze_b (test FAILS).
        Fixed code: sin(0)=sin(2π)=0        → ze_a == ze_b (test PASSES).
        """
        encoder = EventEncoder(_CONFIG).eval()
        torch.manual_seed(42)
        batch, ne, ni = 1, 3, 4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)

        # dow=0 and dow=7 are equivalent modulo 7
        xt_a = torch.tensor([[[0, 0, 0]] * ne], dtype=torch.long)  # dow=0
        xt_b = torch.tensor([[[0, 7, 0]] * ne], dtype=torch.long)  # dow=7

        with torch.no_grad():
            _, ze_a = encoder(xe, xt_a)
            _, ze_b = encoder(xe, xt_b)

        assert torch.allclose(ze_a, ze_b, atol=1e-5), (
            "DEF-002: dow=0 and dow=7 must produce identical ze. "
            "Calendar sincos must normalise by period=7 (2π·d/7). "
            f"Max diff: {(ze_a - ze_b).abs().max().item():.6f}"
        )

    def test_dom_period_equivalence(self) -> None:
        """dom=0 and dom=31 are period-equivalent — ze must be identical.

        Period=31 → sin(2π·0/31)=sin(2π·31/31)=0.  (key-numbers.md §2.3.3)
        Buggy code: sin(0)=0 ≠ sin(31)≈0.404 → ze_a ≠ ze_b (test FAILS).
        Fixed code: sin(0)=sin(2π)=0         → ze_a == ze_b (test PASSES).
        """
        encoder = EventEncoder(_CONFIG).eval()
        torch.manual_seed(42)
        batch, ne, ni = 1, 3, 4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)

        # dom=0 and dom=31 are equivalent modulo 31
        xt_a = torch.tensor([[[0, 0, 0]] * ne], dtype=torch.long)   # dom=0
        xt_b = torch.tensor([[[0, 0, 31]] * ne], dtype=torch.long)  # dom=31

        with torch.no_grad():
            _, ze_a = encoder(xe, xt_a)
            _, ze_b = encoder(xe, xt_b)

        assert torch.allclose(ze_a, ze_b, atol=1e-5), (
            "DEF-002: dom=0 and dom=31 must produce identical ze. "
            "Calendar sincos must normalise by period=31 (2π·dom/31). "
            f"Max diff: {(ze_a - ze_b).abs().max().item():.6f}"
        )

    def test_all_periods_simultaneously(self) -> None:
        """xt=[0,0,0] and xt=[24,7,31] are fully period-equivalent.

        All three dimensions at once: hour+24≡hour, dow+7≡dow, dom+31≡dom.
        This is the combined test that all three periods are correct.
        """
        encoder = EventEncoder(_CONFIG).eval()
        torch.manual_seed(42)
        batch, ne, ni = 1, 3, 4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)

        xt_a = torch.tensor([[[0, 0, 0]] * ne], dtype=torch.long)    # base
        xt_b = torch.tensor([[[24, 7, 31]] * ne], dtype=torch.long)  # +1 full period each

        with torch.no_grad():
            _, ze_a = encoder(xe, xt_a)
            _, ze_b = encoder(xe, xt_b)

        assert torch.allclose(ze_a, ze_b, atol=1e-5), (
            "DEF-002: xt=[0,0,0] and xt=[24,7,31] must produce identical ze. "
            "All three calendar periods must be normalised: "
            "hour/24, dow/7, dom/31 (key-numbers.md §2.3.3). "
            f"Max diff: {(ze_a - ze_b).abs().max().item():.6f}"
        )


# ---------------------------------------------------------------------------
# TestWithinEventPaddingMask — DEF-005a: xe_valid masks padding positions
# ---------------------------------------------------------------------------


class TestWithinEventPaddingMask:
    """DEF-005a — EventEncoder must mask within-event padding positions.

    Each event has ni_max token slots, but real events have fewer real tokens.
    Padding slots (positions n_tok..ni_max-1) contain zero-embedded pad tokens
    and must NOT influence the [EVT] token output at position 0.

    Without masking (current bug): attention allows padding positions to
    contribute to z_hat_e[:,:,0,:] through the softmax, changing [EVT]
    representations when padding content changes (while real tokens stay fixed).

    With xe_valid masking: padding keys are set to -inf before softmax → weight
    is zero → padding positions cannot influence real token outputs.

    Paper: §2.3.3 — each event processed independently; [EVT] at position 0
    summarises the real tokens for that event only.
    """

    def test_padding_tokens_do_not_affect_evt_output(self) -> None:
        """Changing padding slot content must not change z_hat_e for real positions.

        Real tokens: positions 0..n_real-1.
        Padding tokens: positions n_real..ni-1 (different content in base vs modified).

        With xe_valid mask: z_hat_e[:,:,0:n_real,:] is identical between runs.
        Without mask (bug): z_hat_e differs because padding keys attend to real queries.
        """
        encoder = EventEncoder(_CONFIG).eval()
        batch, ne, ni = 1, 2, 6
        n_real = 3  # first 3 token positions are real; positions 3-5 are padding

        torch.manual_seed(42)
        xe_base = torch.randn(batch, ne, ni, _CONFIG.d_model)

        # Modify ONLY the padding slots (positions n_real..ni-1)
        xe_modified = xe_base.clone()
        xe_modified[:, :, n_real:, :] = torch.randn(batch, ne, ni - n_real, _CONFIG.d_model)

        # xe_valid: True = real token, False = padding
        xe_valid = torch.zeros(batch, ne, ni, dtype=torch.bool)
        xe_valid[:, :, :n_real] = True

        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        with torch.no_grad():
            z_hat_e_a, _ = encoder(xe_base, xt, xe_valid=xe_valid)
            z_hat_e_b, _ = encoder(xe_modified, xt, xe_valid=xe_valid)

        assert torch.allclose(
            z_hat_e_a[:, :, :n_real, :],
            z_hat_e_b[:, :, :n_real, :],
            atol=1e-5,
        ), (
            "DEF-005a: real token outputs must not change when padding content changes. "
            "EventEncoder must use xe_valid to mask padding positions in attention. "
            f"Max diff: {(z_hat_e_a[:,:,:n_real,:] - z_hat_e_b[:,:,:n_real,:]).abs().max().item():.6f}"  # noqa: E501
        )

    def test_no_xe_valid_is_backward_compatible(self) -> None:
        """forward(xe, xt) without xe_valid must still work (backward compat)."""
        encoder = EventEncoder(_CONFIG).eval()
        batch, ne, ni = 1, 2, 4
        xe = torch.randn(batch, ne, ni, _CONFIG.d_model)
        xt = torch.zeros(batch, ne, 3, dtype=torch.long)

        with torch.no_grad():
            z_hat_e, ze = encoder(xe, xt)  # no xe_valid — must not raise

        assert z_hat_e.shape == (batch, ne, ni, _CONFIG.d_model)
        assert ze.shape == (batch, ne, _CONFIG.d_model)
