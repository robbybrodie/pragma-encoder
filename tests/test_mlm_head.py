"""Tests for MLMHead.

Derived from PRAGMA paper Section 2.3.5:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:        verify forward(z_hat_e_ij, zh_i, zh_0) output shapes
  Architecture tests: verify nn.Module, config constructor, two linears, no LayerNorm
  Math tests:         verify all inputs affect output, determinism, gradients, loss
  Spec tests:         verify mlm_head_input_dim=3×d_model, mlm_head_output_dim=d_model

Key paper properties (§2.3.5, Equation 8):
  - Input: three pre-gathered tensors, each (n_masked, d_model)
      z_hat_e_ij — Event Encoder token output at masked position (local context)
      zh_i       — History Encoder [EVT] output for event i (cross-event context)
      zh_0       — History Encoder [USR] output (user-level context)
  - Concatenation inside head: cat([...], dim=-1) → (n_masked, 3*d_model)
  - Projection: Linear(3*d_model, d_model) → GELU → Linear(d_model, value_vocab_size)
  - Output: logits (n_masked, value_vocab_size)
  - Loss: cross-entropy with label smoothing (§2.3.5) via compute_loss()
  - IS nn.Module — has trainable parameters (two Linear layers)
  - No LayerNorm between the two Linear layers (not in paper)

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
nn = torch.nn
import torch.nn.functional as F

from pragma_encoder.model.config import PRAGMAConfig
from pragma_encoder.model.mlm_head import MLMHead

_CONFIG = PRAGMAConfig.pragma_s()


# ---------------------------------------------------------------------------
# TestShapes — verify forward(z_hat_e_ij, zh_i, zh_0) output shapes
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify MLMHead input/output shapes (§2.3.5, Equation 8)."""

    def test_output_shape(self) -> None:
        """§2.3.5 / Eq 8: forward returns (n_masked, value_vocab_size) logits.

        The MLM head receives flat gathered tensors — not 3D (batch, ne, ni).
        Gathering masked positions from (batch, ne, ni) to (n_masked,) is done
        by the caller (PRAGMA.forward) before calling MLMHead.forward().
        """
        head = MLMHead(_CONFIG)
        n_masked = 37
        z_hat_e_ij = torch.randn(n_masked, _CONFIG.d_model)
        zh_i       = torch.randn(n_masked, _CONFIG.d_model)
        zh_0       = torch.randn(n_masked, _CONFIG.d_model)

        logits = head.forward(z_hat_e_ij, zh_i, zh_0)

        assert logits.shape == (n_masked, _CONFIG.value_vocab_size), (
            f"§2.3.5: logits must be (n_masked, value_vocab_size) = "
            f"({n_masked}, {_CONFIG.value_vocab_size}), got {tuple(logits.shape)}"
        )

    def test_accepts_flat_masked_positions(self) -> None:
        """§2.3.5 / Eq 8: each input is (n_masked, d_model) — flat, not 3D.

        The caller gathers masked positions from (batch, ne, ni, d_model) to
        (n_masked, d_model) BEFORE calling forward. The MLM head does NOT
        receive 3D tensors and does NOT do the gathering itself.
        """
        head = MLMHead(_CONFIG)
        n_masked = 10
        z_hat_e_ij = torch.randn(n_masked, _CONFIG.d_model)
        zh_i       = torch.randn(n_masked, _CONFIG.d_model)
        zh_0       = torch.randn(n_masked, _CONFIG.d_model)

        logits = head.forward(z_hat_e_ij, zh_i, zh_0)

        assert logits.ndim == 2, (
            f"§2.3.5: logits must be 2D (n_masked, value_vocab_size), "
            f"got {logits.ndim}D with shape {tuple(logits.shape)}"
        )

    def test_output_dim_equals_value_vocab_size(self) -> None:
        """§2.3.5: output logit dimension = config.value_vocab_size (~28k).

        The MLM head predicts over the value vocabulary, not the key vocabulary.
        key-numbers.md: value_vocab_size = ~28,000.
        """
        head = MLMHead(_CONFIG)
        n_masked = 5
        z_hat_e_ij = torch.randn(n_masked, _CONFIG.d_model)
        zh_i       = torch.randn(n_masked, _CONFIG.d_model)
        zh_0       = torch.randn(n_masked, _CONFIG.d_model)

        logits = head.forward(z_hat_e_ij, zh_i, zh_0)

        assert logits.shape[-1] == _CONFIG.value_vocab_size, (
            f"§2.3.5: output dim must be config.value_vocab_size "
            f"({_CONFIG.value_vocab_size}), got {logits.shape[-1]}"
        )


# ---------------------------------------------------------------------------
# TestArchitecture — nn.Module, config constructor, two linears, no LayerNorm
# ---------------------------------------------------------------------------


class TestArchitecture:
    """Verify MLMHead architecture (§2.3.5)."""

    def test_is_nn_module(self) -> None:
        """§2.3.5: MLMHead IS nn.Module — it has trainable parameters.

        Unlike MaskingStrategy, the MLM head has two Linear layers with weights
        that are updated during pre-training. It must subclass nn.Module so
        that parameters() and state_dict() work correctly.
        """
        head = MLMHead(_CONFIG)
        assert isinstance(head, nn.Module), (
            "§2.3.5: MLMHead must subclass nn.Module — it has trainable Linear layers."
        )

    def test_constructor_takes_config(self) -> None:
        """DEVELOPMENT_PROCESS.md: constructor must be (self, config: PRAGMAConfig).

        All three model sizes must be constructable. No individual args allowed.
        """
        head_s = MLMHead(PRAGMAConfig.pragma_s())
        head_m = MLMHead(PRAGMAConfig.pragma_m())
        head_l = MLMHead(PRAGMAConfig.pragma_l())

        # value_vocab_size is the same across variants (§2.2)
        assert head_s.value_vocab_size == 28_000   # key-numbers.md: §2.2
        assert head_m.value_vocab_size == 28_000   # key-numbers.md: §2.2
        assert head_l.value_vocab_size == 28_000   # key-numbers.md: §2.2

    def test_first_projection_is_3d_to_d(self) -> None:
        """key-numbers.md: mlm_head_input_dim = 3 × d_model (§2.3.5, Equation 8).

        The first Linear layer must accept 3*d_model input features.
        Equation 8: input = [z_hat_e_ij : zh_i : zh_0] ∈ R^(3d).
        A first linear of (d_model, d_model) is wrong — it misses the 3× factor
        and cannot accept the concatenated input.
        """
        head = MLMHead(_CONFIG)
        expected_in = 3 * _CONFIG.d_model

        linear_layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
        first_linear = linear_layers[0]

        assert first_linear.in_features == expected_in, (
            f"key-numbers.md: first Linear must have in_features = 3 × d_model = "
            f"{expected_in}, got {first_linear.in_features}. "
            f"Equation 8: input is cat([z_hat_e_ij, zh_i, zh_0]) ∈ R^(3d)."
        )

    def test_no_layer_norm(self) -> None:
        """§2.3.5: no LayerNorm between the two Linear layers.

        The paper specifies: Linear(3d → d) → GELU → Linear(d → vocab_size).
        There is no LayerNorm in this pipeline. The stub incorrectly added one
        from the BERT MLM head pattern, but PRAGMA does not include it.
        """
        head = MLMHead(_CONFIG)
        layer_norms = [m for m in head.modules() if isinstance(m, nn.LayerNorm)]
        assert len(layer_norms) == 0, (
            f"§2.3.5: MLMHead must NOT have LayerNorm (not in paper). "
            f"Found {len(layer_norms)} LayerNorm module(s). "
            f"Paper specifies: Linear(3d→d) → GELU → Linear(d→vocab)."
        )

    def test_two_linear_layers(self) -> None:
        """§2.3.5 / Eq 8: exactly two Linear layers.

        Layer 1: Linear(3*d_model, d_model) — projection from concatenated input
        Layer 2: Linear(d_model, value_vocab_size) — projection to vocabulary
        No additional linear layers should be present.
        """
        head = MLMHead(_CONFIG)
        linear_layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2, (
            f"§2.3.5: MLMHead must have exactly 2 Linear layers "
            f"(3d→d and d→vocab_size), got {len(linear_layers)}."
        )
        # Verify the second linear's output dimension
        assert linear_layers[1].out_features == _CONFIG.value_vocab_size, (
            f"§2.3.5: second Linear must have out_features = value_vocab_size "
            f"({_CONFIG.value_vocab_size}), got {linear_layers[1].out_features}."
        )


# ---------------------------------------------------------------------------
# TestMathProperties — inputs affect output, determinism, gradients, loss
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from §2.3.5."""

    def test_all_three_inputs_affect_output(self) -> None:
        """§2.3.5 / Eq 8: all three input tensors contribute to the logits.

        z_hat_e_ij, zh_i, and zh_0 are concatenated and projected together.
        Changing any one of them while holding the others fixed must change the
        output logits. If one input has no effect, the head is not using Equation 8.
        """
        head = MLMHead(_CONFIG).eval()
        n_masked = 8
        torch.manual_seed(42)
        z_base = torch.randn(n_masked, _CONFIG.d_model)
        z_alt  = torch.randn(n_masked, _CONFIG.d_model)

        with torch.no_grad():
            logits_base = head.forward(z_base, z_base, z_base)

            # Change only z_hat_e_ij
            logits_z1 = head.forward(z_alt, z_base, z_base)
            # Change only zh_i
            logits_z2 = head.forward(z_base, z_alt, z_base)
            # Change only zh_0
            logits_z3 = head.forward(z_base, z_base, z_alt)

        assert not torch.allclose(logits_base, logits_z1, atol=1e-6), (
            "§2.3.5: changing z_hat_e_ij must change logits. "
            "All three inputs are concatenated — each must affect the output."
        )
        assert not torch.allclose(logits_base, logits_z2, atol=1e-6), (
            "§2.3.5: changing zh_i must change logits."
        )
        assert not torch.allclose(logits_base, logits_z3, atol=1e-6), (
            "§2.3.5: changing zh_0 must change logits."
        )

    def test_deterministic_in_eval_mode(self) -> None:
        """§2.3.5: same inputs → same logits in eval mode (no dropout stochasticity)."""
        head = MLMHead(_CONFIG).eval()
        n_masked = 6
        z_hat_e_ij = torch.randn(n_masked, _CONFIG.d_model)
        zh_i       = torch.randn(n_masked, _CONFIG.d_model)
        zh_0       = torch.randn(n_masked, _CONFIG.d_model)

        with torch.no_grad():
            logits1 = head.forward(z_hat_e_ij, zh_i, zh_0)
            logits2 = head.forward(z_hat_e_ij, zh_i, zh_0)

        assert torch.allclose(logits1, logits2), (
            "MLMHead output must be deterministic in eval mode."
        )

    def test_gradients_flow(self) -> None:
        """§2.3.5: all trainable parameters receive gradients after backward().

        Both Linear layers must receive gradients when loss.backward() is called.
        Frozen or disconnected parameters would fail silently during training.
        """
        head = MLMHead(_CONFIG)
        n_masked = 12
        z_hat_e_ij = torch.randn(n_masked, _CONFIG.d_model)
        zh_i       = torch.randn(n_masked, _CONFIG.d_model)
        zh_0       = torch.randn(n_masked, _CONFIG.d_model)

        logits = head.forward(z_hat_e_ij, zh_i, zh_0)
        logits.sum().backward()

        for name, param in head.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient — it is not being trained."
            )
            assert not torch.all(param.grad == 0), (
                f"Parameter '{name}' has all-zero gradients."
            )

    def test_loss_is_scalar_and_differentiable(self) -> None:
        """§2.3.5: compute_loss(logits, targets) — cross-entropy with label smoothing.

        The paper specifies "cross-entropy with label smoothing" as the MLM
        objective (§2.3.5). compute_loss() must:
          - Accept logits (n_masked, value_vocab_size) and targets (n_masked,) ints
          - Return a positive scalar loss with requires_grad=True
          - Use label smoothing: loss must differ from vanilla cross-entropy
            (label_smoothing=0.0). If they are identical, label smoothing is off.
        """
        head = MLMHead(_CONFIG)
        n_masked = 20
        torch.manual_seed(7)
        # Use forward() so logits flow through the model's parameters and
        # therefore have requires_grad=True (as in real pre-training)
        z_hat_e_ij = torch.randn(n_masked, _CONFIG.d_model)
        zh_i       = torch.randn(n_masked, _CONFIG.d_model)
        zh_0       = torch.randn(n_masked, _CONFIG.d_model)
        logits  = head.forward(z_hat_e_ij, zh_i, zh_0)
        targets = torch.randint(0, _CONFIG.value_vocab_size, (n_masked,))

        loss = head.compute_loss(logits, targets)

        assert loss.shape == (), (
            f"§2.3.5: compute_loss must return a scalar (shape ()), "
            f"got shape {tuple(loss.shape)}"
        )
        assert loss.item() > 0, (
            f"§2.3.5: cross-entropy loss must be positive, got {loss.item():.4f}"
        )
        assert loss.requires_grad, (
            "§2.3.5: loss must require grad so backward() works during pre-training."
        )

        # Label smoothing check: smoothed loss ≠ vanilla cross-entropy
        vanilla_loss = F.cross_entropy(logits, targets, label_smoothing=0.0)
        assert not torch.isclose(loss, vanilla_loss, atol=1e-6), (
            "§2.3.5: compute_loss must use label smoothing (§2.3.5). "
            "Loss equals vanilla cross-entropy — label smoothing appears to be off."
        )


# ---------------------------------------------------------------------------
# TestPaperSpecifications — exact values from key-numbers.md
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact values from key-numbers.md (§2.3.5)."""

    def test_mlm_head_input_dim_is_3d_model(self) -> None:
        """key-numbers.md: mlm_head_input_dim = 3 × d_model (§2.3.5).

        The first Linear layer's in_features must equal 3 * config.d_model
        for all three model sizes. This is the key structural constraint from
        Equation 8: input = [z_hat_e_ij : zh_i : zh_0] ∈ R^(3d).
        """
        for config in [
            PRAGMAConfig.pragma_s(),
            PRAGMAConfig.pragma_m(),
            PRAGMAConfig.pragma_l(),
        ]:
            head = MLMHead(config)
            linear_layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
            assert linear_layers[0].in_features == 3 * config.d_model, (  # key-numbers.md: §2.3.5
                f"key-numbers.md: mlm_head_input_dim must be 3 × d_model = "
                f"{3 * config.d_model} for {config.model_name}, "
                f"got {linear_layers[0].in_features}"
            )

    def test_mlm_head_output_dim_intermediate_is_d_model(self) -> None:
        """key-numbers.md: mlm_head_output_dim = d_model (§2.3.5).

        The first Linear layer's out_features must equal d_model — the paper
        specifies the intermediate projection as "3d → d → logits" (key-numbers.md).
        The intermediate dimension is d_model, not d_ffn or any other size.
        """
        for config in [
            PRAGMAConfig.pragma_s(),
            PRAGMAConfig.pragma_m(),
            PRAGMAConfig.pragma_l(),
        ]:
            head = MLMHead(config)
            linear_layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
            assert linear_layers[0].out_features == config.d_model, (  # key-numbers.md: §2.3.5
                f"key-numbers.md: mlm_head_output_dim (intermediate) must be d_model = "
                f"{config.d_model} for {config.model_name}, "
                f"got {linear_layers[0].out_features}"
            )
