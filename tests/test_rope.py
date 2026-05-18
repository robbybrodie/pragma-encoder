"""Tests for RoPEEncoding (Rotary Positional Embedding).

Derived from PRAGMA paper Section 2.3.2 and 2.3.4:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:    verify forward() output shapes and float position input
  Math tests:     verify Equation 9 relative-distance property and temporal decay
  Gradient tests: verify no trainable parameters and gradients flow through rotation
  Spec tests:     verify head_dim=64 for all model sizes (key-numbers.md, Table 1)

Key paper property (Equation 9):
  attention(qm, kn) ∝ qm^T kn = q^T R(tm)^T R(tn) k = q^T R(tn − tm) k
  The dot product depends only on relative temporal distance, not absolute position.

Every value asserted here appears in docs/paper/key-numbers.md
with its paper source section.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from src.model.config import PRAGMAConfig
from src.encoders.rope import RoPEEncoding

_CONFIG = PRAGMAConfig.pragma_s()
_HEAD_DIM = _CONFIG.d_model // _CONFIG.n_heads  # 64 — key-numbers.md: head_dimension, Table 1


# ---------------------------------------------------------------------------
# TestShapes — verify output shapes and continuous position input
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify that RoPEEncoding returns correct output shapes (§2.3.2, Eq 9)."""

    def test_output_shape_preserved(self) -> None:
        """§2.3.2 / Eq 9: forward() returns (q_rot, k_rot) with same shapes as input."""
        rope = RoPEEncoding(_CONFIG)
        batch, n_heads, seq_len = 2, _CONFIG.n_heads, 6
        q = torch.randn(batch, n_heads, seq_len, _HEAD_DIM)
        k = torch.randn(batch, n_heads, seq_len, _HEAD_DIM)
        positions = torch.zeros(batch, seq_len)

        q_rot, k_rot = rope(q, k, positions)

        assert q_rot.shape == q.shape, (
            f"q_rot shape {q_rot.shape} must match input q shape {q.shape}"
        )
        assert k_rot.shape == k.shape, (
            f"k_rot shape {k_rot.shape} must match input k shape {k.shape}"
        )

    def test_accepts_continuous_positions(self) -> None:
        """§2.3.2 / Eq 9: positions must be a float tensor (log-seconds).

        PRAGMA temporal coordinates are continuous floats from Equation 2
        (t' = 8·ln(1+t/8)), not sequential integers from torch.arange().
        RoPEEncoding must accept arbitrary non-integer float positions.
        """
        rope = RoPEEncoding(_CONFIG)
        batch, n_heads, seq_len = 1, _CONFIG.n_heads, 4
        q = torch.randn(batch, n_heads, seq_len, _HEAD_DIM)
        k = torch.randn(batch, n_heads, seq_len, _HEAD_DIM)

        # Non-integer temporal coordinates — values from Equation 2: 8·ln(1+t/8)
        # t=0s → 0.0, t=8s → 5.545, t=16s → 11.09, t=40s → 21.97
        positions = torch.tensor([[0.0, 5.545, 11.09, 21.97]])

        q_rot, k_rot = rope(q, k, positions)

        assert q_rot.shape == q.shape, (
            f"RoPEEncoding must accept float positions; got shape {q_rot.shape}"
        )
        assert q_rot.dtype == torch.float32, (
            f"Output must be float32, got {q_rot.dtype}"
        )


# ---------------------------------------------------------------------------
# TestMathProperties — verify Equation 9 relative-distance and decay properties
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from Equation 9 (§2.3.2, Su et al. 2024)."""

    def test_dot_product_depends_only_on_relative_distance(self) -> None:
        """Equation 9: attention(qm, kn) = q^T R(tn − tm) k.

        The dot product between rotated q at position tm and rotated k at
        position tn depends only on the relative distance (tn − tm),
        not on the absolute values of tm and tn.

        Pair A: positions [0.0, 10.0]   — relative distance 10
        Pair B: positions [100.0, 110.0] — relative distance 10
        q_rot_A[0]^T k_rot_A[1] must equal q_rot_B[0]^T k_rot_B[1].
        """
        torch.manual_seed(0)
        rope = RoPEEncoding(_CONFIG)

        q_vec = torch.randn(1, 1, 1, _HEAD_DIM)
        k_vec = torch.randn(1, 1, 1, _HEAD_DIM)
        q_pair = q_vec.expand(-1, -1, 2, -1).clone()  # same vector at both positions
        k_pair = k_vec.expand(-1, -1, 2, -1).clone()

        pos_A = torch.tensor([[0.0, 10.0]])    # relative distance = 10
        pos_B = torch.tensor([[100.0, 110.0]]) # relative distance = 10

        q_rot_A, k_rot_A = rope(q_pair, k_pair, pos_A)
        q_rot_B, k_rot_B = rope(q_pair, k_pair, pos_B)

        # Cross-position dot product: q at seq pos 0, k at seq pos 1
        sim_A = (q_rot_A[0, 0, 0] * k_rot_A[0, 0, 1]).sum().item()
        sim_B = (q_rot_B[0, 0, 0] * k_rot_B[0, 0, 1]).sum().item()

        assert abs(sim_A - sim_B) < 1e-4, (
            f"Eq 9: same relative distance must yield same dot product. "
            f"sim_A={sim_A:.6f}, sim_B={sim_B:.6f}, |diff|={abs(sim_A - sim_B):.2e}"
        )

    def test_closer_temporal_positions_higher_similarity(self) -> None:
        """Equation 9 / §2.3.2: closer temporal positions → higher similarity.

        For the same query vector at position 0:
          key at t=1.0  (near) → higher dot product similarity
          key at t=100.0 (far) → lower dot product similarity

        Reference: DEVELOPMENT_PROCESS.md §2b math property example.
        key-numbers.md: RoPE encodes temporal proximity (§2.3.2).
        """
        torch.manual_seed(42)
        rope = RoPEEncoding(_CONFIG)

        # Use the same vector at both sequence positions (self-similarity test)
        q_vec = torch.randn(1, 1, 1, _HEAD_DIM)
        q_pair = q_vec.expand(-1, -1, 2, -1).clone()

        pos_near = torch.tensor([[0.0, 1.0]])    # 1 log-second apart
        pos_far  = torch.tensor([[0.0, 100.0]])  # 100 log-seconds apart

        q_rot_near, k_rot_near = rope(q_pair, q_pair, pos_near)
        q_rot_far,  k_rot_far  = rope(q_pair, q_pair, pos_far)

        # Cross-position dot product: query at pos 0, key at pos 1
        sim_near = (q_rot_near[0, 0, 0] * k_rot_near[0, 0, 1]).sum().item()
        sim_far  = (q_rot_far[0, 0, 0]  * k_rot_far[0, 0, 1]).sum().item()

        assert sim_near > sim_far, (
            f"Eq 9: closer temporal positions must have higher similarity. "
            f"sim_near={sim_near:.4f}, sim_far={sim_far:.4f}"
        )

    def test_zero_relative_distance_is_unrotated(self) -> None:
        """Equation 9: when tm == tn, R(tm)^T R(tn) = R(0) = I.

        The dot product of q at position t with k at the SAME position t
        must equal the unrotated q^T k, regardless of what t is.

        Proof: q_rot[m]^T k_rot[m] = (R(t)*q)^T (R(t)*k)
                                    = q^T R(t)^T R(t) k
                                    = q^T I k = q^T k
        """
        torch.manual_seed(7)
        rope = RoPEEncoding(_CONFIG)

        q_vec = torch.randn(1, 1, 1, _HEAD_DIM)
        k_vec = torch.randn(1, 1, 1, _HEAD_DIM)
        q_pair = q_vec.expand(-1, -1, 2, -1).clone()
        k_pair = k_vec.expand(-1, -1, 2, -1).clone()

        # Both positions at t=5.0 — relative distance = 0
        positions = torch.tensor([[5.0, 5.0]])

        q_rot, k_rot = rope(q_pair, k_pair, positions)

        sim_rotated   = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum().item()
        sim_unrotated = (q_vec[0, 0, 0] * k_vec[0, 0, 0]).sum().item()

        assert abs(sim_rotated - sim_unrotated) < 1e-4, (
            f"Eq 9: same position (delta=0) must equal unrotated dot product. "
            f"rotated={sim_rotated:.6f}, unrotated={sim_unrotated:.6f}, "
            f"|diff|={abs(sim_rotated - sim_unrotated):.2e}"
        )


# ---------------------------------------------------------------------------
# TestGradientFlow — no trainable parameters; gradients flow through rotation
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify RoPE gradient behaviour (§2.3.2)."""

    def test_no_trainable_parameters(self) -> None:
        """§2.3.2: RoPEEncoding is a pure rotation — no learned parameters.

        inv_freq is a fixed buffer (not a Parameter). Confirming this ensures
        the rotation is not accidentally treated as trainable weights, which
        would waste parameter budget and distort the rotation semantics.
        """
        rope = RoPEEncoding(_CONFIG)
        params = list(rope.parameters())
        assert len(params) == 0, (
            f"RoPEEncoding must have 0 trainable parameters, got {len(params)}: "
            f"{[n for n, _ in rope.named_parameters()]}"
        )

    def test_gradients_flow_through_rotation(self) -> None:
        """§2.3.2: Gradients must flow through the RoPE rotation to q and k.

        The ProfileStateEncoder and HistoryEncoder train their Q and K projection
        weights. If gradients do not flow through RoPE, those projections receive
        no gradient — a silent failure that would only be caught at evaluation.
        """
        rope = RoPEEncoding(_CONFIG)
        q = torch.randn(1, _CONFIG.n_heads, 4, _HEAD_DIM, requires_grad=True)
        k = torch.randn(1, _CONFIG.n_heads, 4, _HEAD_DIM, requires_grad=True)
        positions = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

        q_rot, k_rot = rope(q, k, positions)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()

        assert q.grad is not None, (
            "Gradients must flow through RoPE to q — "
            "Q projection weights in encoders require this"
        )
        assert k.grad is not None, (
            "Gradients must flow through RoPE to k — "
            "K projection weights in encoders require this"
        )
        assert not torch.all(q.grad == 0), "q gradients must be non-zero"
        assert not torch.all(k.grad == 0), "k gradients must be non-zero"


# ---------------------------------------------------------------------------
# TestPaperSpecifications — verify exact values from key-numbers.md
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact values from the paper (all sourced from key-numbers.md)."""

    def test_head_dim_is_64_from_config(self) -> None:
        """key-numbers.md: head_dimension = 64, Table 1.

        Derived invariant: d_model / n_heads = 64 for all three model sizes.
        RoPEEncoding must derive head_dim from config, not hardcode 64.

        PRAGMA-S: 192 / 3  = 64
        PRAGMA-M: 512 / 8  = 64
        PRAGMA-L: 1024 / 16 = 64
        """
        for config in [
            PRAGMAConfig.pragma_s(),
            PRAGMAConfig.pragma_m(),
            PRAGMAConfig.pragma_l(),
        ]:
            head_dim = config.d_model // config.n_heads
            assert head_dim == 64, (  # key-numbers.md: head_dimension = 64, Table 1
                f"key-numbers.md: head_dim must be 64 for {config.model_name}, "
                f"got {head_dim}"
            )
            rope = RoPEEncoding(config)
            assert rope.head_dim == 64, (
                f"RoPEEncoding.head_dim must be 64 for {config.model_name}, "
                f"got {rope.head_dim}"
            )
