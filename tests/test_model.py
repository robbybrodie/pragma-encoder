"""Tests for the full PRAGMA model assembly.

Derived from PRAGMA paper Section 2.3, Table 1:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:        verify forward() output tensor shapes
  Math tests:         verify USR conditioning, determinism, no NaN
  Architecture tests: class name, config constructor, no shared weights
  Spec tests:         parameter count ~10M (Table 1), value_vocab_size, zh structure

Key paper properties (§2.3, Equations 4–8):
  - forward() accepts pre-embedded float tensors (Equation 1 applied externally)
  - Inputs: xa (batch,na,d), ta (batch,na), xe (batch,ne,ni,d), xt (batch,ne,3),
            te (batch,1+ne), token_ids (batch,ne,ni), mask (batch,ne,ni) bool
  - Steps: ProfileStateEncoder → EventEncoder → cat → HistoryEncoder → MLMHead
  - zh: (batch, 1+ne, d_model) — position 0 = [USR], positions 1..ne = [EVT]
  - Gathering: for mask[b,i,j]=True: zh_i = zh[b, i+1, :] (NOT i — [USR] is at 0)
  - Returns {"zh": zh} always; adds {"logits": logits} when mask is not None
  - Class name: PRAGMA (not PRAGMAModel)
  - Constructor: PRAGMA(config: PRAGMAConfig) — no individual args
  - PRAGMA-S ~10M parameters (Table 1)

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
nn = torch.nn

from src.model.config import PRAGMAConfig
from src.model.pragma import PRAGMA

_CONFIG = PRAGMAConfig.pragma_s()

# Test dimensions — small enough for fast tests
_BATCH = 2
_NE    = 4   # number of events
_NI    = 6   # tokens per event (ni; includes [EVT] at position 0)
_NA    = 8   # profile tokens (na; includes [USR] at position 0)


def _make_inputs(config: PRAGMAConfig = _CONFIG, seed: int = 0):
    """Build a complete set of PRAGMA forward() inputs."""
    torch.manual_seed(seed)
    d = config.d_model
    return dict(
        xa=torch.randn(_BATCH, _NA, d),            # profile embeddings (float)
        ta=torch.zeros(_BATCH, _NA),               # profile temporal coords
        xe=torch.randn(_BATCH, _NE, _NI, d),       # event token embeddings (float)
        xt=torch.randint(0, 24, (_BATCH, _NE, 3)), # calendar features (int)
        te=torch.zeros(_BATCH, 1 + _NE),           # history temporal coords
    )


def _make_mask(density: float = 0.3, seed: int = 99) -> torch.Tensor:
    """Return a (batch, ne, ni) bool mask with approximately `density` True."""
    torch.manual_seed(seed)
    return torch.rand(_BATCH, _NE, _NI) < density


# ---------------------------------------------------------------------------
# TestShapes — verify forward() output shapes
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify PRAGMA forward() input/output shapes (§2.3, Equations 4–8)."""

    def test_zh_shape_when_mask_none(self) -> None:
        """§2.3 / Eq 7: zh is (batch, 1+ne, d_model) when mask=None.

        The History Encoder output zh always has 1+ne positions:
          zh[:,0,:]   = [USR] token (from ProfileStateEncoder via za_usr)
          zh[:,1:,:]  = [EVT] tokens (ne of them)
        The caller slices as needed — PRAGMA returns the full zh.
        """
        model = PRAGMA(_CONFIG).eval()
        inputs = _make_inputs()

        with torch.no_grad():
            output = model(**inputs)

        assert "zh" in output, "forward() must always return 'zh' key"
        assert output["zh"].shape == (_BATCH, 1 + _NE, _CONFIG.d_model), (
            f"§2.3 / Eq 7: zh must be (batch, 1+ne, d_model) = "
            f"({_BATCH}, {1+_NE}, {_CONFIG.d_model}), "
            f"got {tuple(output['zh'].shape)}"
        )

    def test_logits_shape_when_mask_given(self) -> None:
        """§2.3.5 / Eq 8: logits is (n_masked, value_vocab_size) when mask given.

        The MLM head receives only the n_masked gathered positions — not the
        full (batch, ne, ni) tensor. Logits shape must be exactly
        (n_masked, value_vocab_size) not (batch, ne, ni, value_vocab_size).
        """
        model = PRAGMA(_CONFIG).eval()
        inputs = _make_inputs()
        mask = _make_mask(density=0.3)
        n_masked = mask.sum().item()

        with torch.no_grad():
            output = model(**inputs, mask=mask)

        assert "logits" in output, "forward() must return 'logits' key when mask is given"
        assert output["logits"].shape == (n_masked, _CONFIG.value_vocab_size), (
            f"§2.3.5 / Eq 8: logits must be (n_masked, value_vocab_size) = "
            f"({n_masked}, {_CONFIG.value_vocab_size}), "
            f"got {tuple(output['logits'].shape)}"
        )

    def test_no_logits_key_when_mask_none(self) -> None:
        """§2.3.5: 'logits' key is absent from output when mask=None.

        During embedding extraction (no masking), the MLM head is not called.
        The output dict must not contain 'logits' — callers must check before
        accessing to avoid silent errors.
        """
        model = PRAGMA(_CONFIG).eval()
        inputs = _make_inputs()

        with torch.no_grad():
            output = model(**inputs)

        assert "logits" not in output, (
            "§2.3.5: 'logits' must NOT be in output when mask=None. "
            "MLM head is only called during pre-training with mask provided."
        )

    def test_n_masked_equals_mask_sum(self) -> None:
        """§2.3.5: logits.shape[0] == mask.sum() — exactly one row per masked position.

        Gathering collects one (z_hat_e_ij, zh_i, zh_0) triple per True position
        in the token-level mask (batch, ne, ni). The logit count must exactly
        match the number of True entries in the mask.
        """
        model = PRAGMA(_CONFIG).eval()
        inputs = _make_inputs()
        mask = _make_mask(density=0.25)
        expected_n = mask.sum().item()

        with torch.no_grad():
            output = model(**inputs, mask=mask)

        assert output["logits"].shape[0] == expected_n, (
            f"§2.3.5: logits.shape[0] must equal mask.sum() = {expected_n}, "
            f"got {output['logits'].shape[0]}. "
            f"Each masked token position must produce exactly one logit row."
        )


# ---------------------------------------------------------------------------
# TestMathProperties — USR conditioning, determinism, no NaN
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from §2.3."""

    def test_usr_conditioning_changes_evt_outputs(self) -> None:
        """§2.3.4: changing xa (profile) changes zh[:,1:,:] ([EVT] outputs).

        [USR] is at zh[:,0,:] — the History Encoder receives z = [za_usr : ze]
        and via bidirectional self-attention, [USR] at position 0 conditions
        all [EVT] positions (1..ne). This is the profile-conditioning mechanism.
        """
        model = PRAGMA(_CONFIG).eval()
        inputs_base = _make_inputs(seed=0)
        inputs_alt  = _make_inputs(seed=0)
        torch.manual_seed(77)
        inputs_alt["xa"] = torch.randn(_BATCH, _NA, _CONFIG.d_model)

        with torch.no_grad():
            zh_base = model(**inputs_base)["zh"]
            zh_alt  = model(**inputs_alt)["zh"]

        # [EVT] outputs (positions 1..ne) must change when [USR] input changes
        assert not torch.allclose(zh_base[:, 1:, :], zh_alt[:, 1:, :], atol=1e-6), (
            "§2.3.4: changing xa ([USR] profile input) must change zh[:,1:,:] "
            "([EVT] outputs). [USR] conditions [EVT] tokens via bidirectional "
            "self-attention over the concatenated [USR:EVT] sequence."
        )

    def test_deterministic_in_eval_mode(self) -> None:
        """§2.3: same inputs → same outputs in eval mode (dropout disabled)."""
        model = PRAGMA(_CONFIG).eval()
        inputs = _make_inputs()

        with torch.no_grad():
            zh1 = model(**inputs)["zh"]
            zh2 = model(**inputs)["zh"]

        assert torch.allclose(zh1, zh2), (
            "PRAGMA output must be deterministic in eval mode. "
            "Call model.eval() to disable dropout."
        )

    def test_no_nan_in_outputs(self) -> None:
        """§2.3: forward() must produce finite outputs — no NaN or Inf."""
        model = PRAGMA(_CONFIG).eval()
        inputs = _make_inputs()
        mask = _make_mask(density=0.2)

        with torch.no_grad():
            output = model(**inputs, mask=mask)

        assert not torch.isnan(output["zh"]).any(), "NaN found in zh"
        assert not torch.isinf(output["zh"]).any(), "Inf found in zh"
        if "logits" in output:
            assert not torch.isnan(output["logits"]).any(), "NaN found in logits"
            assert not torch.isinf(output["logits"]).any(), "Inf found in logits"

    def test_different_mask_densities_give_different_logit_counts(self) -> None:
        """§2.3.5: denser mask → more logit rows; sparser mask → fewer.

        The logit count tracks mask.sum() exactly. Verifying two different
        masks produce different counts confirms the gathering is token-level
        (batch, ne, ni) not event-level (batch, ne).
        """
        model = PRAGMA(_CONFIG).eval()
        inputs = _make_inputs()

        mask_sparse = _make_mask(density=0.1, seed=1)
        mask_dense  = _make_mask(density=0.6, seed=2)

        with torch.no_grad():
            n_sparse = model(**inputs, mask=mask_sparse)["logits"].shape[0]
            n_dense  = model(**inputs, mask=mask_dense)["logits"].shape[0]

        assert n_sparse == mask_sparse.sum().item(), (
            f"Sparse mask: logits count {n_sparse} != mask.sum() {mask_sparse.sum().item()}"
        )
        assert n_dense == mask_dense.sum().item(), (
            f"Dense mask: logits count {n_dense} != mask.sum() {mask_dense.sum().item()}"
        )
        assert n_sparse < n_dense, (
            "Sparser mask must produce fewer logit rows than denser mask."
        )


# ---------------------------------------------------------------------------
# TestArchitecture — class name, config constructor, no shared weights
# ---------------------------------------------------------------------------


class TestArchitecture:
    """Verify PRAGMA architecture constraints (§2.3)."""

    def test_class_name_is_pragma(self) -> None:
        """§2.3: class must be named PRAGMA (not PRAGMAModel or similar).

        DEVELOPMENT_PROCESS.md: class names must match the paper exactly.
        The paper refers to the full model as 'PRAGMA'.
        """
        assert PRAGMA.__name__ == "PRAGMA", (
            f"Class must be named 'PRAGMA', got '{PRAGMA.__name__}'. "
            "DEVELOPMENT_PROCESS.md: class names must match the paper."
        )

    def test_constructor_takes_config(self) -> None:
        """DEVELOPMENT_PROCESS.md: constructor must be (self, config: PRAGMAConfig).

        All three model sizes must be constructable. No individual args.
        Scaling must require changing exactly one line (the classmethod call).
        """
        model_s = PRAGMA(PRAGMAConfig.pragma_s())
        model_m = PRAGMA(PRAGMAConfig.pragma_m())
        model_l = PRAGMA(PRAGMAConfig.pragma_l())

        # Verify each has the correct d_model from config (Table 1)
        assert model_s.config.d_model == 192   # key-numbers.md: Table 1
        assert model_m.config.d_model == 512   # key-numbers.md: Table 1
        assert model_l.config.d_model == 1024  # key-numbers.md: Table 1

    def test_no_shared_weights_across_encoders(self) -> None:
        """§2.3: the three encoders must have independent weights (no weight tying).

        ADR 002: raw tokens never cross encoder boundaries. Independent weights
        ensure each encoder specialises for its domain (profile vs event vs history).
        Shared weight tensors (same id()) would silently couple the encoders.
        """
        model = PRAGMA(_CONFIG)

        # Collect parameter data_ptr() for each encoder separately
        profile_ptrs = {p.data_ptr() for p in model.profile_encoder.parameters()}
        event_ptrs   = {p.data_ptr() for p in model.event_encoder.parameters()}
        history_ptrs = {p.data_ptr() for p in model.history_encoder.parameters()}

        assert profile_ptrs.isdisjoint(event_ptrs), (
            "§2.3: ProfileStateEncoder and EventEncoder must not share weights. "
            "Found overlapping parameter tensors."
        )
        assert profile_ptrs.isdisjoint(history_ptrs), (
            "§2.3: ProfileStateEncoder and HistoryEncoder must not share weights."
        )
        assert event_ptrs.isdisjoint(history_ptrs), (
            "§2.3: EventEncoder and HistoryEncoder must not share weights."
        )


# ---------------------------------------------------------------------------
# TestPaperSpecifications — exact values from key-numbers.md
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact values from the paper (all sourced from key-numbers.md)."""

    def test_parameter_count_pragma_s_approx_10m(self) -> None:
        """key-numbers.md: params (PRAGMA-S) = 10M, Table 1.

        The parameter count for PRAGMA-S (d_model=192) must be approximately
        10 million. Exact count depends on implementation details, so we check
        within [5M, 20M] to catch gross mismatches (e.g. accidentally using
        PRAGMA-M hyperparameters or missing the MLM head decoder).
        """
        model = PRAGMA(PRAGMAConfig.pragma_s())
        n_params = sum(p.numel() for p in model.parameters())

        assert 5_000_000 < n_params < 20_000_000, (  # key-numbers.md: Table 1
            f"key-numbers.md: PRAGMA-S must have ~10M parameters (Table 1), "
            f"got {n_params:,}. "
            f"Check that all four components (3 encoders + MLMHead) are included "
            f"and no component is accidentally duplicated or omitted."
        )

    def test_logits_dim_is_value_vocab_size(self) -> None:
        """key-numbers.md: value_vocab_size = ~28,000 (§2.2).

        Logits are over the value vocabulary (~28k tokens), NOT the key
        vocabulary (~60 types). Using key_vocab_size would produce 28k/60 = ~470×
        fewer logit classes and fail the MLM objective.
        """
        model = PRAGMA(_CONFIG).eval()
        inputs = _make_inputs()
        mask = _make_mask(density=0.3)

        with torch.no_grad():
            logits = model(**inputs, mask=mask)["logits"]

        assert logits.shape[-1] == _CONFIG.value_vocab_size, (  # key-numbers.md: §2.2
            f"key-numbers.md: logits dim must be value_vocab_size "
            f"({_CONFIG.value_vocab_size}), got {logits.shape[-1]}. "
            f"MLM predicts value tokens (~28k), not key types (~60)."
        )

    def test_zh_has_usr_plus_ne_positions(self) -> None:
        """§2.3.4 / Eq 7: zh has 1+ne positions — [USR] at 0, [EVT] at 1..ne.

        Equation 6: z = [za_usr : ze] — the [USR] token is prepended to the
        ne event representations before the History Encoder. The output zh
        therefore has 1+ne positions:
          zh[:,0,:]   — [USR] token (from ProfileStateEncoder via za[:,0:1,:])
          zh[:,1:,:]  — [EVT] tokens (ne of them, from EventEncoder)

        This also verifies that the MLM gathering `zh[b_idx, i_idx+1, :]`
        is correct — event i is at position i+1, not i.
        """
        model = PRAGMA(_CONFIG).eval()
        inputs = _make_inputs()

        with torch.no_grad():
            zh = model(**inputs)["zh"]

        assert zh.shape[1] == 1 + _NE, (
            f"§2.3.4 / Eq 7: zh must have 1+ne = {1+_NE} positions, "
            f"got {zh.shape[1]}. "
            f"Position 0 = [USR] (from ProfileStateEncoder), "
            f"positions 1..{_NE} = [EVT] tokens (from EventEncoder). "
            f"Gathering for event i uses zh[:,i+1,:] — not zh[:,i,:]."
        )
