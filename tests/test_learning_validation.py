"""Tests for PRAGMA local learning validation (Sections 2.3, 2.3.5, 2.4).

Derived from PRAGMA paper Sections 2.3, 2.3.5, and 2.4:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
    Math tests:       verify loss is finite, positive, and decreases with training
    Gradient tests:   verify key model modules receive non-zero gradients
    Spec tests:       verify checkpoint save/reload and readiness report fields
    Safety tests:     verify no cluster side-effects (S3, KFP, oc)

These tests use only synthetic data. No real dataset, no S3, no KFP, no OpenShift.
All tests run locally with no cluster access required.

Test classes:
    TestLossComputation     — §2.3.5: MLM loss is finite, positive, and trainable
    TestGradientFlow        — §2.3:   key modules receive non-zero gradients
    TestOptimizerStep       — §2.4:   optimizer step changes at least one parameter
    TestCheckpoint          — §2.4:   save/reload preserves model state exactly
    TestMaskingTargets      — §2.3.5: masking produces at least one supervised target
    TestNoSideEffects       — local:  training loop does not import cluster APIs
    TestReadinessReport     — local:  readiness utility exposes correct fields
"""

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pragma_encoder.masking import MaskingStrategy
from pragma_encoder.model import PRAGMA, PRAGMAConfig
from pragma_encoder.model.assembler import EmbeddingAssembler
from pragma_encoder.tokenizer.vocabulary import VocabularySpec
from pragma_encoder.training.readiness import make_readiness_report

# ---------------------------------------------------------------------------
# Tiny config for convergence tests — fast, deterministic, no GPU needed
# ---------------------------------------------------------------------------

def _tiny_config() -> PRAGMAConfig:
    """A minimal config for convergence tests.

    Uses a tiny vocabulary (100 values) so cross-entropy converges in <30 steps.
    Uses no dropout so the loss trajectory is deterministic (no stochastic noise
    from dropout corrupting the convergence signal).

    Dimensions are deliberately non-standard — this config is NOT paper-spec.
    It exists only to make the convergence test fast and reliable.
    """
    return PRAGMAConfig(
        model_name="pragma-s-tiny-test",
        d_model=32,
        d_ffn=64,
        n_heads=2,
        profile_encoder_layers=1,
        event_encoder_layers=1,
        history_encoder_layers=1,
        dropout=0.0,           # no stochastic noise — deterministic convergence
        max_event_tokens=8,
        max_profile_tokens=8,
        max_events=10,
        key_vocab_size=10,
        value_vocab_size=100,  # tiny vocab — easy to memorise a fixed batch
        token_mask_prob=0.15,
        event_mask_prob=0.10,
        key_mask_prob=0.10,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_vocab_spec(config: PRAGMAConfig) -> VocabularySpec:
    """Build a minimal VocabularySpec consistent with config vocab sizes."""
    return VocabularySpec(
        special_tokens={"PAD": 0, "MASK": 1, "USR": 2, "EVT": 3, "UNK": 4},
        key_start=5,
        key_size=config.key_vocab_size,
        value_start=5 + config.key_vocab_size,
        value_size=config.value_vocab_size,
        total_embedding_vocab_size=5 + config.key_vocab_size + config.value_vocab_size,
        field_key_ids={},
        field_value_ranges={},
    )


def _make_synthetic_batch(
    config: PRAGMAConfig,
    vocab_spec: VocabularySpec,
    batch: int = 2,
    ne: int = 5,
    ni: int = 8,
    na: int = 6,
) -> dict:
    """Return a dictionary of synthetic batch tensors in valid ID ranges.

    All IDs are drawn uniformly from the valid vocab ranges defined by
    vocab_spec. No real transaction data is used.
    """
    xe_val_ids = torch.randint(
        vocab_spec.value_start,
        vocab_spec.value_start + config.value_vocab_size,
        (batch, ne, ni),
    )
    xe_key_ids = torch.randint(
        vocab_spec.key_start,
        vocab_spec.key_start + config.key_vocab_size,
        (batch, ne, ni),
    )
    xe_pos_ids = torch.arange(ni).unsqueeze(0).unsqueeze(0).expand(batch, ne, -1)
    xa_val_ids = torch.randint(
        vocab_spec.value_start,
        vocab_spec.value_start + config.value_vocab_size,
        (batch, na),
    )
    xa_key_ids = torch.randint(
        vocab_spec.key_start,
        vocab_spec.key_start + config.key_vocab_size,
        (batch, na),
    )
    xa_pos_ids = torch.arange(na).unsqueeze(0).expand(batch, -1)
    ta = torch.zeros(batch, na)
    te = torch.cat([torch.zeros(batch, 1), torch.rand(batch, ne) * 100.0], dim=1)
    xt = torch.stack([
        torch.randint(0, 24, (batch, ne)),
        torch.randint(0, 7,  (batch, ne)),
        torch.randint(1, 32, (batch, ne)),
    ], dim=-1)
    return dict(
        xe_val_ids=xe_val_ids,
        xe_key_ids=xe_key_ids,
        xe_pos_ids=xe_pos_ids,
        xa_val_ids=xa_val_ids,
        xa_key_ids=xa_key_ids,
        xa_pos_ids=xa_pos_ids,
        ta=ta, te=te, xt=xt,
    )


def _assemble_and_forward(
    config: PRAGMAConfig,
    vocab_spec: VocabularySpec,
    model: PRAGMA,
    assembler: EmbeddingAssembler,
    masker: MaskingStrategy,
    batch: dict | None = None,
) -> torch.Tensor:
    """Run one masked forward pass and return the scalar MLM loss.

    Guarantees at least one [MASK] position so the loss always has a
    supervision signal (prevents degenerate zero-masked batches on tiny dims).

    Args:
        batch: If None, a fresh synthetic batch is generated.

    Returns:
        Scalar MLM cross-entropy loss (differentiable — has grad_fn when
        model is in train mode).
    """
    if batch is None:
        batch = _make_synthetic_batch(config, vocab_spec)

    masked_val_ids, _, mlm_mask = masker.forward(
        batch["xe_val_ids"], batch["xe_key_ids"]
    )

    # Guarantee at least one [MASK] position so the loss is never empty.
    if not mlm_mask.any():
        mlm_mask = mlm_mask.clone()
        mlm_mask[0, 0, 0] = True
        masked_val_ids = masked_val_ids.clone()
        masked_val_ids[0, 0, 0] = 1  # MASK_ID = 1

    assembled = assembler.forward(
        xa_key_ids=batch["xa_key_ids"],
        xa_val_ids=batch["xa_val_ids"],
        xa_pos_ids=batch["xa_pos_ids"],
        ta=batch["ta"],
        xe_key_ids=batch["xe_key_ids"],
        xe_val_ids=masked_val_ids,
        xe_pos_ids=batch["xe_pos_ids"],
        xt=batch["xt"],
        te=batch["te"],
        target_ids=batch["xe_val_ids"],
        mask=mlm_mask,
    )

    output = model.forward(
        xa=assembled.xa,
        ta=assembled.ta,
        xe=assembled.xe,
        xt=assembled.xt,
        te=assembled.te,
        mask=assembled.mlm_mask,
    )

    logits = output["logits"]                    # (n_masked, value_vocab_size)
    valid_targets = assembled.targets[assembled.mlm_mask]  # (n_masked,)
    return model.mlm_head.compute_loss(logits, valid_targets)


# ---------------------------------------------------------------------------
# TestLossComputation — §2.3.5: MLM loss is finite, positive, and trainable
# ---------------------------------------------------------------------------


class TestLossComputation:
    """§2.3.5: MLM loss properties on synthetic data.

    Tests that the masked event modelling loss is numerically sound
    and that the model can learn (loss decreases) on a fixed batch.
    """

    def test_loss_is_finite(self) -> None:
        """Section 2.3.5: MLM loss must be finite (not NaN, not Inf).

        NaN or Inf loss indicates numerical instability:
          - NaN: degenerate embeddings, log(0), division by zero
          - Inf: overflow in softmax (logits too large), zero masked positions

        Uses PRAGMA-S (paper-spec config) on synthetic data.
        No real dataset required.
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        model = PRAGMA(config)
        assembler = EmbeddingAssembler(vocab_spec, config)
        masker = MaskingStrategy(config)
        model.eval()
        assembler.eval()

        with torch.no_grad():
            loss = _assemble_and_forward(config, vocab_spec, model, assembler, masker)

        assert not torch.isnan(loss), (
            "MLM loss is NaN. "
            "Numerical instability in embeddings or logits."
        )
        assert not torch.isinf(loss), (
            "MLM loss is Inf. "
            "Overflow in cross-entropy — logits may be too large."
        )

    def test_loss_is_positive(self) -> None:
        """Section 2.3.5: MLM loss must be positive on random initialisation.

        Cross-entropy with label smoothing over value_vocab_size=28k classes
        on a randomly-initialised model should produce a positive loss close
        to log(28000) ≈ 10.2 nats. A zero or negative loss on random data
        indicates a bug in target indexing or loss computation.
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        model = PRAGMA(config)
        assembler = EmbeddingAssembler(vocab_spec, config)
        masker = MaskingStrategy(config)
        model.eval()
        assembler.eval()

        with torch.no_grad():
            loss = _assemble_and_forward(config, vocab_spec, model, assembler, masker)

        assert loss.item() > 0.0, (
            f"MLM loss is not positive: {loss.item():.6f}. "
            "Cross-entropy on a random initialisation must be positive."
        )

    def test_loss_decreases_over_steps_on_fixed_batch(self) -> None:
        """Section 2.3.5 / §2.4: loss decreases when training on a fixed batch.

        A tiny model (d_model=32, value_vocab_size=100) trained with Adam
        on the same batch for 30 steps must show a net loss decrease.
        This proves that:
          1. Gradients flow through the full architecture (§2.3)
          2. The optimizer updates parameters in the right direction (§2.4)
          3. The model can memorise a small batch (sanity baseline for real training)

        Uses a tiny non-paper-spec config so the test runs in <3 seconds.
        Loss decrease on real data (PRAGMA-S, IBM TabFormer) is verified by
        the manual smoke script: examples/workbench/06_local_learning_validation.py

        key-numbers.md: token_mask_prob=0.15, Adam optimizer (§2.4)
        """
        config = _tiny_config()
        vocab_spec = _make_vocab_spec(config)
        model = PRAGMA(config)
        assembler = EmbeddingAssembler(vocab_spec, config)
        masker = MaskingStrategy(config)
        model.train()
        assembler.train()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(assembler.parameters()),
            lr=1e-3,
        )

        # Fixed batch — same data repeated 30 times → model should memorise
        torch.manual_seed(42)
        fixed_batch = _make_synthetic_batch(config, vocab_spec, batch=4, ne=4, ni=8, na=6)

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = _assemble_and_forward(
                config, vocab_spec, model, assembler, masker, batch=fixed_batch
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        first_loss = losses[0]
        last_loss  = losses[-1]

        assert last_loss < first_loss, (
            f"Loss did not decrease over 30 steps on a fixed batch. "
            f"Initial loss: {first_loss:.4f}, final loss: {last_loss:.4f}. "
            "This means gradients are not flowing or the optimizer is misconfigured. "
            "Check for detached tensors or frozen parameters."
        )


# ---------------------------------------------------------------------------
# TestGradientFlow — §2.3: key modules receive non-zero gradients
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """§2.3: gradient flow through the three-encoder PRAGMA architecture.

    Derived from DEVELOPMENT_PROCESS.md §2c — gradient flow test pattern:
    'Catches frozen parameters, detached tensors, wrong loss computation.'
    """

    def test_key_module_gradients(self) -> None:
        """Section 2.3: the four named modules must each have non-zero gradients.

        After one backward pass, at least one parameter in each of:
          - profile_encoder (§2.3.2)
          - event_encoder   (§2.3.3)
          - history_encoder (§2.3.4)
          - mlm_head        (§2.3.5)

        must have a non-zero gradient. A zero-gradient module is not being trained.
        This test detects architectural breaks (detached tensors, wrong loss path).
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        model = PRAGMA(config)
        assembler = EmbeddingAssembler(vocab_spec, config)
        masker = MaskingStrategy(config)
        model.train()
        assembler.train()

        loss = _assemble_and_forward(config, vocab_spec, model, assembler, masker)
        loss.backward()

        key_modules = {
            "profile_encoder": model.profile_encoder,
            "event_encoder":   model.event_encoder,
            "history_encoder": model.history_encoder,
            "mlm_head":        model.mlm_head,
        }
        for module_name, module in key_modules.items():
            has_nonzero_grad = any(
                p.grad is not None and p.grad.abs().sum().item() > 0.0
                for p in module.parameters()
            )
            assert has_nonzero_grad, (
                f"Module '{module_name}' has no non-zero gradient after backward(). "
                "This module is not being trained. "
                "Check for detached tensors or an incorrect loss path."
            )

    def test_embedding_assembler_receives_gradient(self) -> None:
        """Equation 1 (§2.3.1): the shared embedding table E must receive a gradient.

        EmbeddingAssembler.E is the shared lookup table for key and value IDs
        (Equation 1). If it does not receive a gradient, token embeddings are
        not updated and the model cannot adapt to the training data distribution.
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        model = PRAGMA(config)
        assembler = EmbeddingAssembler(vocab_spec, config)
        masker = MaskingStrategy(config)
        model.train()
        assembler.train()

        loss = _assemble_and_forward(config, vocab_spec, model, assembler, masker)
        loss.backward()

        assert assembler.E.weight.grad is not None, (
            "EmbeddingAssembler.E.weight has no gradient. "
            "The shared embedding table (Equation 1) is not being updated."
        )
        assert assembler.E.weight.grad.abs().sum().item() > 0.0, (
            "EmbeddingAssembler.E.weight gradient is all-zero. "
            "Token embeddings are not being trained."
        )

    def test_no_model_parameters_have_null_gradient(self) -> None:
        """Section 2.3: every requires_grad parameter must receive a gradient.

        Frozen or disconnected parameters fail silently. This test detects
        any parameter that is part of the forward graph but receives no gradient.
        The full three-encoder architecture requires all parameters to participate.
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        model = PRAGMA(config)
        assembler = EmbeddingAssembler(vocab_spec, config)
        masker = MaskingStrategy(config)
        model.train()
        assembler.train()

        loss = _assemble_and_forward(config, vocab_spec, model, assembler, masker)
        loss.backward()

        null_grad_params = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        assert not null_grad_params, (
            f"The following model parameters received no gradient: {null_grad_params}. "
            "These parameters are frozen or the computation graph is broken."
        )


# ---------------------------------------------------------------------------
# TestOptimizerStep — §2.4: optimizer step changes parameters
# ---------------------------------------------------------------------------


class TestOptimizerStep:
    """§2.4: one optimizer step must change at least one model parameter."""

    def test_optimizer_step_changes_at_least_one_parameter(self) -> None:
        """Section 2.4: optimizer.step() must change at least one parameter.

        If no parameters change after a step, either:
          - gradients are zero (not flowing)
          - the optimizer is misconfigured (zero learning rate)
          - parameters are frozen

        Snapshots parameter values before and after one Adam step.
        At least one parameter tensor must differ from the snapshot.
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        model = PRAGMA(config)
        assembler = EmbeddingAssembler(vocab_spec, config)
        masker = MaskingStrategy(config)
        model.train()
        assembler.train()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(assembler.parameters()),
            lr=1e-3,
        )

        # Snapshot parameters before step
        params_before = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        optimizer.zero_grad()
        loss = _assemble_and_forward(config, vocab_spec, model, assembler, masker)
        loss.backward()
        optimizer.step()

        changed = [
            name
            for name, param in model.named_parameters()
            if not torch.allclose(param.data, params_before[name])
        ]
        assert changed, (
            "No model parameters changed after optimizer.step(). "
            "Gradients may not be flowing or the optimizer is misconfigured."
        )

    def test_multiple_optimizer_steps_change_different_parameters(self) -> None:
        """Section 2.4: repeated steps must keep changing parameters.

        Two consecutive Adam steps must each produce a different parameter state.
        If parameters stop changing after the first step, the optimizer has stalled
        (e.g. learning rate annealed to zero by a scheduler misconfiguration).
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        model = PRAGMA(config)
        assembler = EmbeddingAssembler(vocab_spec, config)
        masker = MaskingStrategy(config)
        model.train()
        assembler.train()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(assembler.parameters()),
            lr=1e-3,
        )

        # Step 1
        optimizer.zero_grad()
        _assemble_and_forward(config, vocab_spec, model, assembler, masker).backward()
        optimizer.step()

        snapshot_after_step1 = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        # Step 2
        optimizer.zero_grad()
        _assemble_and_forward(config, vocab_spec, model, assembler, masker).backward()
        optimizer.step()

        changed_step2 = [
            name
            for name, param in model.named_parameters()
            if not torch.allclose(param.data, snapshot_after_step1[name])
        ]
        assert changed_step2, (
            "No parameters changed on the second optimizer step. "
            "The optimizer may have stalled."
        )


# ---------------------------------------------------------------------------
# TestCheckpoint — §2.4: save/reload preserves model state exactly
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """§2.4: checkpoint save and reload must preserve model state exactly."""

    def test_checkpoint_save_creates_nonempty_file(self, tmp_path: pathlib.Path) -> None:
        """Section 2.4: torch.save creates the expected checkpoint file.

        The checkpoint must exist and be non-empty after save.
        An empty file indicates torch.save silently failed (disk full, etc.).
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)
        checkpoint_path = tmp_path / "pragma-s-step-000.pt"

        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": 0,
        }, checkpoint_path)

        assert checkpoint_path.exists(), (
            f"Checkpoint file was not created at {checkpoint_path}"
        )
        assert checkpoint_path.stat().st_size > 0, (
            "Checkpoint file exists but is empty — torch.save may have failed silently."
        )

    def test_checkpoint_save_includes_required_keys(self, tmp_path: pathlib.Path) -> None:
        """Section 2.4: checkpoint dict must include model_state_dict, config, step.

        A checkpoint missing these keys cannot be reliably reloaded for
        continued training or embedding extraction.
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)
        checkpoint_path = tmp_path / "pragma-s-keys-test.pt"

        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": 0,
        }, checkpoint_path)

        ckpt = torch.load(checkpoint_path, weights_only=False)
        for key in ("model_state_dict", "config", "step"):
            assert key in ckpt, (
                f"Checkpoint is missing required key '{key}'. "
                "Checkpoints must include model_state_dict, config, and step."
            )

    def test_checkpoint_reload_restores_state_dict(self, tmp_path: pathlib.Path) -> None:
        """Section 2.4: model loaded from checkpoint has identical state_dict.

        Every parameter tensor in the reloaded model must be numerically
        identical (allclose) to the original. A mismatch indicates a
        serialisation or deserialisation error.
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)
        checkpoint_path = tmp_path / "pragma-s-reload.pt"

        torch.save({"model_state_dict": model.state_dict(), "config": config, "step": 0},
                   checkpoint_path)

        model_reloaded = PRAGMA(config)
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model_reloaded.load_state_dict(ckpt["model_state_dict"])

        for name, param in model.named_parameters():
            assert torch.allclose(param.data, model_reloaded.state_dict()[name]), (
                f"Parameter '{name}' does not match after checkpoint reload."
            )

    def test_checkpoint_reload_gives_identical_forward_output(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Section 2.4: reloaded model must produce identical zh output.

        Given the same inputs, the original model and the checkpoint-reloaded
        model must produce bit-identical zh (history encoder output). This
        proves that the full model state — not just named parameters, but
        also registered buffers such as sinusoidal position tables — is
        correctly serialised and deserialised.
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        model = PRAGMA(config)
        assembler = EmbeddingAssembler(vocab_spec, config)
        model.eval()
        assembler.eval()

        checkpoint_path = tmp_path / "pragma-s-output-test.pt"
        torch.save({"model_state_dict": model.state_dict(), "step": 0}, checkpoint_path)

        # Generate fixed batch and forward pass for original model
        torch.manual_seed(0)
        batch = _make_synthetic_batch(config, vocab_spec)
        masker = MaskingStrategy(config)
        masked_val_ids, _, _ = masker.forward(batch["xe_val_ids"], batch["xe_key_ids"])

        assembled = assembler.forward(
            xa_key_ids=batch["xa_key_ids"], xa_val_ids=batch["xa_val_ids"],
            xa_pos_ids=batch["xa_pos_ids"], ta=batch["ta"],
            xe_key_ids=batch["xe_key_ids"], xe_val_ids=masked_val_ids,
            xe_pos_ids=batch["xe_pos_ids"], xt=batch["xt"], te=batch["te"],
        )

        with torch.no_grad():
            out_original = model.forward(
                xa=assembled.xa, ta=assembled.ta,
                xe=assembled.xe, xt=assembled.xt, te=assembled.te,
            )

        # Reload and run with the same inputs
        model_reloaded = PRAGMA(config)
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model_reloaded.load_state_dict(ckpt["model_state_dict"])
        model_reloaded.eval()

        with torch.no_grad():
            out_reloaded = model_reloaded.forward(
                xa=assembled.xa, ta=assembled.ta,
                xe=assembled.xe, xt=assembled.xt, te=assembled.te,
            )

        assert torch.allclose(out_original["zh"], out_reloaded["zh"]), (
            "Reloaded model produces different zh output — "
            "full model state (including buffers) was not preserved by checkpoint."
        )


# ---------------------------------------------------------------------------
# TestMaskingTargets — §2.3.5: masking produces supervised targets
# ---------------------------------------------------------------------------


class TestMaskingTargets:
    """§2.3.5: masking must produce at least one [MASK] position per batch.

    Reference: Ostroukhov et al. (2026), Section 2.3.5 — three masking strategies.
    """

    def test_masking_produces_mask_positions_across_batches(self) -> None:
        """Section 2.3.5: at least one [MASK] position across 10 random batches.

        With token_mask_prob=0.15 and 8 token positions per event, the
        probability of zero masked positions in a single batch of 2×5 events
        is approximately (0.85^8 × 0.90^5 × 0.90^5)^2 ≈ 0.003. Over 10
        independent batches this probability is negligible.

        key-numbers.md: token_mask_prob=0.15, event_mask_prob=0.10, key_mask_prob=0.10
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        masker = MaskingStrategy(config)

        torch.manual_seed(99)
        found_masked = False
        for _ in range(10):
            batch = _make_synthetic_batch(config, vocab_spec, batch=2, ne=5, ni=8, na=6)
            _, _, mlm_mask = masker.forward(batch["xe_val_ids"], batch["xe_key_ids"])
            if mlm_mask.any():
                found_masked = True
                break

        assert found_masked, (
            "MaskingStrategy produced zero masked positions across 10 independent batches. "
            "token_mask_prob, event_mask_prob, or key_mask_prob may be zero. "
            f"config.token_mask_prob={config.token_mask_prob}"
        )

    def test_mask_output_is_boolean_dtype(self) -> None:
        """Section 2.3.5: the mask tensor must have dtype torch.bool.

        PRAGMA.forward() calls mask.nonzero(as_tuple=True) to gather masked
        positions. This requires boolean dtype. An integer mask (e.g. uint8)
        would treat all non-zero IDs as masked, causing silent over-masking.
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        masker = MaskingStrategy(config)

        batch = _make_synthetic_batch(config, vocab_spec)
        _, _, mlm_mask = masker.forward(batch["xe_val_ids"], batch["xe_key_ids"])

        assert mlm_mask.dtype == torch.bool, (
            f"mask must be torch.bool dtype, got {mlm_mask.dtype}. "
            "PRAGMA.forward() uses mask.nonzero() which requires bool dtype to "
            "correctly distinguish [MASK] positions from [UNK] positions."
        )

    def test_masked_ids_contains_mask_token_at_mask_positions(self) -> None:
        """Section 2.3.5: [MASK] token (ID=1) appears at all mask=True positions.

        The MaskingStrategy contract: positions where mask=True must have
        masked_ids == MASK_ID (1). This is the corruption that the model
        learns to reverse during pretraining.
        """
        config = PRAGMAConfig.pragma_s()
        vocab_spec = _make_vocab_spec(config)
        masker = MaskingStrategy(config)

        torch.manual_seed(7)
        # Run until we find a batch with at least one masked position
        for _ in range(20):
            batch = _make_synthetic_batch(config, vocab_spec)
            masked_ids, _, mlm_mask = masker.forward(
                batch["xe_val_ids"], batch["xe_key_ids"]
            )
            if mlm_mask.any():
                break

        # All mask=True positions must contain MASK_ID=1
        mask_token_id = 1
        masked_positions = masked_ids[mlm_mask]
        assert (masked_positions == mask_token_id).all(), (
            f"Some mask=True positions do not contain MASK_ID={mask_token_id}. "
            "MaskingStrategy.forward() must write MASK_ID at all mask=True positions."
        )


# ---------------------------------------------------------------------------
# TestNoSideEffects — local training must not import cluster APIs
# ---------------------------------------------------------------------------


class TestNoSideEffects:
    """Local learning validation must not call S3, KFP, or Kubernetes APIs.

    These tests check that the training readiness module does not import
    cluster-facing libraries as a side effect. Importing boto3 or kfp
    would cause training to fail in environments without cluster credentials.
    """

    def test_readiness_module_does_not_import_boto3(self) -> None:
        """Local training readiness module must not import boto3 (S3 client).

        The readiness utility is a pure local diagnostic tool. Importing boto3
        as a side effect would make local training fail when S3 credentials
        are not configured — even when no S3 access is needed.
        """
        # Snapshot sys.modules before (boto3 may already be present if another
        # test imported it — we skip if it was already there before our check)
        boto3_was_present_before = "boto3" in sys.modules

        import importlib
        importlib.import_module("pragma_encoder.training.readiness")

        if not boto3_was_present_before:
            assert "boto3" not in sys.modules, (
                "boto3 was imported as a side effect of importing src.training.readiness. "
                "Local training must not require S3 credentials."
            )

    def test_readiness_module_does_not_import_kfp(self) -> None:
        """Local training readiness module must not import kfp (KFP SDK).

        kfp is an optional dependency. Its absence must not prevent local
        training or readiness reporting from running.
        """
        kfp_was_present_before = "kfp" in sys.modules

        import importlib
        importlib.import_module("pragma_encoder.training.readiness")

        if not kfp_was_present_before:
            assert "kfp" not in sys.modules, (
                "kfp was imported as a side effect of importing src.training.readiness. "
                "Local training must not require the KFP SDK."
            )

    def test_readiness_module_does_not_import_openshift_client(self) -> None:
        """Local training readiness module must not import kubernetes or oc.

        The readiness utility must run without cluster access.
        """
        import importlib
        oc_was_present_before = "kubernetes" in sys.modules

        importlib.import_module("pragma_encoder.training.readiness")

        if not oc_was_present_before:
            assert "kubernetes" not in sys.modules, (
                "kubernetes client was imported by src.training.readiness. "
                "Local training must not require cluster access."
            )


# ---------------------------------------------------------------------------
# TestReadinessReport — readiness utility exposes required fields
# ---------------------------------------------------------------------------


class TestReadinessReport:
    """Readiness report must expose all required fields with correct values.

    The TrainingReadinessReport is a diagnostic tool that summarises the
    dataset and model configuration before training starts.
    """

    def test_readiness_report_has_all_required_fields(self) -> None:
        """Readiness report must expose all fields specified in the user contract.

        Required fields:
            dataset_name, dataset_path, n_sequences, n_events,
            key_vocab_size, value_vocab_size, max_event_tokens,
            max_events, mask_rate, model_variant, trainable_params
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)

        report = make_readiness_report(
            config=config,
            model=model,
            dataset_name="ibm-tabformer-synthetic",
            n_sequences=10,
            n_events=50,
        )

        required_fields = [
            "dataset_name", "dataset_path", "n_sequences", "n_events",
            "key_vocab_size", "value_vocab_size", "max_event_tokens",
            "max_events", "mask_rate", "model_variant", "trainable_params",
        ]
        for field in required_fields:
            assert hasattr(report, field), (
                f"TrainingReadinessReport is missing required field '{field}'."
            )

    def test_readiness_report_vocab_sizes_match_config(self) -> None:
        """Readiness report vocab sizes must exactly match PRAGMAConfig values.

        key-numbers.md: key_vocab_size=60, value_vocab_size=28000 (§2.2)
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)

        report = make_readiness_report(
            config=config, model=model, dataset_name="syn", n_sequences=1, n_events=1,
        )

        assert report.key_vocab_size == config.key_vocab_size, (
            f"key_vocab_size mismatch: report={report.key_vocab_size}, "
            f"config={config.key_vocab_size}"
        )
        assert report.value_vocab_size == config.value_vocab_size, (
            f"value_vocab_size mismatch: report={report.value_vocab_size}, "
            f"config={config.value_vocab_size}"
        )

    def test_readiness_report_max_event_tokens_matches_config(self) -> None:
        """Readiness report max_event_tokens must match PRAGMAConfig.

        key-numbers.md: max_event_tokens=24 (§2.4)
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)

        report = make_readiness_report(
            config=config, model=model, dataset_name="syn", n_sequences=1, n_events=1,
        )

        assert report.max_event_tokens == config.max_event_tokens, (
            f"max_event_tokens mismatch: report={report.max_event_tokens}, "
            f"config={config.max_event_tokens}"
        )
        assert report.max_events == config.max_events, (
            f"max_events mismatch: report={report.max_events}, "
            f"config={config.max_events}"
        )

    def test_readiness_report_trainable_params_matches_model(self) -> None:
        """Readiness report trainable_params must match model.parameters() sum.

        key-numbers.md: PRAGMA-S has approximately 10M trainable parameters.
        The exact count is verified by test_model.py; here we verify that
        the readiness report uses the same counting convention.
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)

        report = make_readiness_report(
            config=config, model=model, dataset_name="syn", n_sequences=1, n_events=1,
        )

        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert report.trainable_params == expected, (
            f"trainable_params mismatch: report={report.trainable_params:,}, "
            f"expected={expected:,}"
        )

    def test_readiness_report_model_variant_matches_config(self) -> None:
        """Readiness report model_variant must match config.model_name.

        Ensures the report correctly identifies which PRAGMA size is being trained.
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)

        report = make_readiness_report(
            config=config, model=model, dataset_name="syn", n_sequences=1, n_events=1,
        )

        assert report.model_variant == config.model_name, (
            f"model_variant mismatch: report={report.model_variant!r}, "
            f"config.model_name={config.model_name!r}"
        )

    def test_readiness_report_mask_rate_in_reasonable_range(self) -> None:
        """Readiness report mask_rate must be in the expected range for §2.3.5.

        The token masking rate is 0.15 (§2.3.5). The reported mask_rate
        (derived from config.token_mask_prob) must be in [0.05, 0.50].
        A value outside this range indicates a bug in rate calculation.

        key-numbers.md: token_mask_prob=0.15 (§2.3.5)
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)

        report = make_readiness_report(
            config=config, model=model, dataset_name="syn", n_sequences=1, n_events=1,
        )

        assert 0.05 <= report.mask_rate <= 0.50, (
            f"mask_rate={report.mask_rate} is outside expected range [0.05, 0.50]. "
            f"Expected a value reflecting token_mask_prob={config.token_mask_prob}."
        )

    def test_readiness_report_dataset_fields_match_inputs(self) -> None:
        """Readiness report dataset_name, n_sequences, n_events must match inputs."""
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)

        report = make_readiness_report(
            config=config,
            model=model,
            dataset_name="ibm-tabformer",
            n_sequences=42,
            n_events=1234,
            dataset_path="/data/tabformer/card_transaction.v1.csv",
        )

        assert report.dataset_name == "ibm-tabformer"
        assert report.n_sequences == 42
        assert report.n_events == 1234
        assert report.dataset_path == "/data/tabformer/card_transaction.v1.csv"

    def test_readiness_report_print_does_not_raise(self) -> None:
        """Readiness report print_report() must not raise an exception.

        The report must be printable without crashing, even when dataset_path
        is None (not yet prepared).
        """
        config = PRAGMAConfig.pragma_s()
        model = PRAGMA(config)

        report = make_readiness_report(
            config=config, model=model, dataset_name="ibm-tabformer",
            n_sequences=10, n_events=500, dataset_path=None,
        )

        # Must not raise
        report.print_report()
