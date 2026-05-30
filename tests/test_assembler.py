"""Tests for EmbeddingAssembler.

Derived from PRAGMA paper Section 2.3.1, Equation 1:
"PRAGMA: Revolut Foundation Model"
Ostroukhov et al. (2026), arXiv:2604.08649v1

Test types:
  Shape tests:    verify output tensor shapes from forward()
  Math tests:     verify Equation 1, target localisation, path independence
  Gradient tests: verify all trainable parameters receive gradients
  Spec tests:     verify sinusoidal (non-learnable) PosEmb, embedding table size,
                  and ADR 002 invariant

Key paper properties (§2.3.1, Equation 1):
  x_ij = PosEmb_j( E(k_i) + E(v_ij) )
  E     — single shared embedding table for both key and value IDs
  PosEmb_j — sinusoidal, NOT learnable (Vaswani et al. 2017)

ADR 002 key invariant:
  targets[targets != IGNORE_INDEX] must be in [0, value_vocab_size)
  (value-vocab-local IDs, not global token IDs)

Test fixture vocabulary (matches test_vocabulary.py):
  Fields: "amount" (NumericalTokenizer, n_buckets=10) and "currency" (CategoricalTokenizer)
  key_start=5, key_size=2, value_start=7, value_size=17, total=24

Every value asserted here appears in docs/paper/key-numbers.md
or is derived from it.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from pragma_encoder.model.assembled_batch import AssembledBatch
from pragma_encoder.model.assembler import EmbeddingAssembler
from pragma_encoder.model.config import PRAGMAConfig
from pragma_encoder.tokenizer.categorical import CategoricalTokenizer
from pragma_encoder.tokenizer.numerical import NumericalTokenizer
from pragma_encoder.tokenizer.pipeline import TokenizerPipeline
from pragma_encoder.tokenizer.vocabulary import VocabularySpec

# ---------------------------------------------------------------------------
# Constants and fixtures
# ---------------------------------------------------------------------------

_CONFIG = PRAGMAConfig.pragma_s()

# Small test dimensions — fast forward passes
_BATCH = 2
_NA    = 4   # profile tokens
_NE    = 3   # events
_NI    = 5   # tokens per event
_D     = _CONFIG.d_model  # 192

# Vocabulary constants (derived from the two-field fixture)
_KEY_START  = 5
_KEY_SIZE   = 2
_VAL_START  = 7
_VAL_SIZE   = 17
_TOTAL_VOCAB = 24

_MASK_ID = 1  # TokenizerPipeline.MASK_ID — [MASK] token for corruption


def _make_assembler():
    """Return (VocabularySpec, EmbeddingAssembler) from the two-field fixture."""
    amount = NumericalTokenizer(n_buckets=10)
    amount.fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    currency = CategoricalTokenizer()
    currency.fit(["GBP", "EUR", "USD"])

    pipeline = TokenizerPipeline(
        field_tokenizers={"amount": amount, "currency": currency}
    )
    spec = pipeline.vocabulary_spec()
    return spec, EmbeddingAssembler(spec, _CONFIG)


def _make_inputs(spec: VocabularySpec, with_mask: bool = True, seed: int = 0) -> dict:
    """Build a complete set of EmbeddingAssembler.forward() inputs.

    Args:
        spec:      VocabularySpec from _make_assembler().
        with_mask: If True, include target_ids and mask (training mode).
                   If False, omit them (inference / embedding-extraction mode).
        seed:      Manual seed for reproducibility.

    Returns:
        Dict keyed to EmbeddingAssembler.forward() parameter names.
    """
    torch.manual_seed(seed)
    val_lo, val_hi = spec.value_start, spec.value_start + spec.value_size
    key_lo, key_hi = spec.key_start, spec.key_start + spec.key_size

    d = dict(
        xa_key_ids = torch.randint(key_lo, key_hi, (_BATCH, _NA)),
        xa_val_ids = torch.randint(val_lo, val_hi, (_BATCH, _NA)),
        xa_pos_ids = torch.zeros(_BATCH, _NA, dtype=torch.long),
        ta         = torch.zeros(_BATCH, _NA),
        xe_key_ids = torch.randint(key_lo, key_hi, (_BATCH, _NE, _NI)),
        xe_val_ids = torch.randint(val_lo, val_hi, (_BATCH, _NE, _NI)),
        xe_pos_ids = torch.zeros(_BATCH, _NE, _NI, dtype=torch.long),
        xt         = torch.randint(0, 24, (_BATCH, _NE, 3)),
        te         = torch.zeros(_BATCH, 1 + _NE),
    )

    if with_mask:
        mask = torch.zeros(_BATCH, _NE, _NI, dtype=torch.bool)
        mask[:, :, 1:] = True          # mask positions 1..(NI-1) per event
        target_ids = d["xe_val_ids"].clone()  # original global value IDs
        d["mask"] = mask
        d["target_ids"] = target_ids

    return d


# ---------------------------------------------------------------------------
# TestShapes — verify forward() output shapes
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify EmbeddingAssembler output shapes (§2.3.1, Equation 1)."""

    def test_xa_shape(self) -> None:
        """§2.3.1 / Eq 1: xa must be (batch, na, d_model) after embedding."""
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=False)

        with torch.no_grad():
            batch = asm(**inputs)

        assert batch.xa.shape == (_BATCH, _NA, _D), (
            f"§2.3.1 / Eq 1: xa must be ({_BATCH}, {_NA}, {_D}), "
            f"got {tuple(batch.xa.shape)}"
        )

    def test_xe_shape(self) -> None:
        """§2.3.1 / Eq 1: xe must be (batch, ne, ni, d_model) after embedding."""
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=False)

        with torch.no_grad():
            batch = asm(**inputs)

        assert batch.xe.shape == (_BATCH, _NE, _NI, _D), (
            f"§2.3.1 / Eq 1: xe must be ({_BATCH}, {_NE}, {_NI}, {_D}), "
            f"got {tuple(batch.xe.shape)}"
        )

    def test_assembled_batch_field_shapes(self) -> None:
        """AssembledBatch fields must have correct shapes and dtypes (ADR 002)."""
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=True)

        with torch.no_grad():
            batch = asm(**inputs)

        assert batch.xt.shape == (_BATCH, _NE, 3),         "xt must be (batch, ne, 3)"
        assert batch.te.shape == (_BATCH, 1 + _NE),        "te must be (batch, 1+ne)"
        assert batch.ta.shape == (_BATCH, _NA),             "ta must be (batch, na)"
        assert batch.mlm_mask.shape == (_BATCH, _NE, _NI), "mlm_mask must be (batch, ne, ni)"
        assert batch.targets.shape == (_BATCH, _NE, _NI),  "targets must be (batch, ne, ni)"
        assert batch.mlm_mask.dtype == torch.bool,  "mlm_mask must be bool"
        assert batch.targets.dtype == torch.long,   "targets must be long"

    def test_inference_mode_no_mask(self) -> None:
        """§2.3.1: mask=None → all targets are IGNORE_INDEX, mlm_mask all False.

        Inference mode (embedding extraction): EmbeddingAssembler embeds
        token IDs and returns the batch with no MLM-specific information.
        """
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=False)

        with torch.no_grad():
            batch = asm(**inputs)

        assert (batch.targets == AssembledBatch.IGNORE_INDEX).all(), (
            "mask=None: all targets must be IGNORE_INDEX (-100)"
        )
        assert not batch.mlm_mask.any(), (
            "mask=None: mlm_mask must be all False"
        )


# ---------------------------------------------------------------------------
# TestMathProperties — Equation 1, target localisation, path independence
# ---------------------------------------------------------------------------


class TestMathProperties:
    """Verify mathematical properties from §2.3.1 and ADR 002."""

    def test_equation_1_uses_both_key_and_value(self) -> None:
        """§2.3.1 / Eq 1: E(k) + E(v) — changing v must change xe.

        If the value embedding were ignored, changing xe_val_ids would have
        no effect. This test verifies the value path is live.
        """
        spec, asm = _make_assembler()
        asm.eval()
        inputs_a = _make_inputs(spec, with_mask=False, seed=0)
        inputs_b = dict(inputs_a)
        inputs_b["xe_val_ids"] = torch.randint(
            spec.value_start, spec.value_start + spec.value_size,
            (_BATCH, _NE, _NI)
        )  # different value IDs, same key IDs

        with torch.no_grad():
            xe_a = asm(**inputs_a).xe
            xe_b = asm(**inputs_b).xe

        assert not torch.allclose(xe_a, xe_b), (
            "§2.3.1 / Eq 1: changing xe_val_ids must change xe output. "
            "E(k) + E(v) — value path must be active."
        )

    def test_position_zero_differs_from_position_one(self) -> None:
        """§2.3.1 / Eq 1: PosEmb(0) ≠ PosEmb(1) — positions are distinguishable.

        Sinusoidal positional embeddings must produce different vectors for
        different within-field positions. If PosEmb were constant, the model
        could not distinguish which token appears first in a multi-token field.
        """
        spec, asm = _make_assembler()
        asm.eval()

        # Two otherwise identical inputs — only pos_ids differ
        inputs_p0 = _make_inputs(spec, with_mask=False, seed=7)
        inputs_p1 = dict(inputs_p0)
        inputs_p1["xe_pos_ids"] = torch.ones(_BATCH, _NE, _NI, dtype=torch.long)

        with torch.no_grad():
            xe_p0 = asm(**inputs_p0).xe
            xe_p1 = asm(**inputs_p1).xe

        assert not torch.allclose(xe_p0, xe_p1), (
            "§2.3.1 / Eq 1: PosEmb(0) must differ from PosEmb(1). "
            "Sinusoidal positions must be distinguishable."
        )

    def test_different_values_produce_different_embeddings(self) -> None:
        """§2.3.1 / Eq 1: E(k)+E(v1) ≠ E(k)+E(v2) for v1 ≠ v2.

        Distinct value tokens must produce distinct embeddings.
        """
        spec, asm = _make_assembler()
        asm.eval()

        # Fixed key IDs; two distinct value IDs at same position
        key_id = torch.full((_BATCH, _NE, _NI), spec.key_start, dtype=torch.long)
        pos_id = torch.zeros(_BATCH, _NE, _NI, dtype=torch.long)

        # Value A: all = value_start (first value token)
        val_a = torch.full((_BATCH, _NE, _NI), spec.value_start, dtype=torch.long)
        # Value B: all = value_start + 1 (second value token)
        val_b = torch.full((_BATCH, _NE, _NI), spec.value_start + 1, dtype=torch.long)

        inputs_a = _make_inputs(spec, with_mask=False)
        inputs_a["xe_key_ids"] = key_id
        inputs_a["xe_val_ids"] = val_a
        inputs_a["xe_pos_ids"] = pos_id

        inputs_b = dict(inputs_a)
        inputs_b["xe_val_ids"] = val_b

        with torch.no_grad():
            xe_a = asm(**inputs_a).xe
            xe_b = asm(**inputs_b).xe

        assert not torch.allclose(xe_a, xe_b), (
            "§2.3.1 / Eq 1: E(k)+E(v1) must differ from E(k)+E(v2) for v1≠v2"
        )

    def test_targets_are_local_ids_at_masked_positions(self) -> None:
        """ADR 002: targets[mask] must be value-vocab-local IDs in [0, value_vocab_size).

        local_id = global_id - value_start. For our fixture, value_start=7
        and value_size=17, so local IDs are in [0, 17) ⊂ [0, value_vocab_size=28000).
        """
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=True)

        with torch.no_grad():
            batch = asm(**inputs)

        mask = inputs["mask"]
        local_targets = batch.targets[mask]

        assert local_targets.numel() > 0, "Need at least one masked position"
        assert (local_targets >= 0).all(), (
            "ADR 002: all non-ignore targets must be >= 0"
        )
        assert (local_targets < _CONFIG.value_vocab_size).all(), (
            f"ADR 002: targets[mask] must be < value_vocab_size={_CONFIG.value_vocab_size}"
        )

    def test_unmasked_targets_are_ignore_index(self) -> None:
        """ADR 002: targets at unmasked positions must be IGNORE_INDEX (-100)."""
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=True)

        with torch.no_grad():
            batch = asm(**inputs)

        mask = inputs["mask"]
        non_masked_targets = batch.targets[~mask]

        assert (non_masked_targets == AssembledBatch.IGNORE_INDEX).all(), (
            f"ADR 002: unmasked positions must have IGNORE_INDEX ({AssembledBatch.IGNORE_INDEX}), "
            f"found {non_masked_targets.unique().tolist()}"
        )

    def test_masked_input_uses_mask_token_embedding(self) -> None:
        """§2.3.5: EmbeddingAssembler must faithfully embed MASK_ID=1 at masked positions.

        MaskingStrategy replaces selected value IDs with MASK_ID=1 before passing
        to EmbeddingAssembler. The assembler embeds whatever ID it receives — E(1)
        when MASK_ID is present. This test verifies that substituting MASK_ID
        changes the embedding at that position, and the result matches E(k)+E(1)+pos.
        """
        spec, asm = _make_assembler()
        asm.eval()

        inputs_orig = _make_inputs(spec, with_mask=False, seed=3)
        inputs_masked = dict(inputs_orig)
        inputs_masked["xe_val_ids"] = inputs_orig["xe_val_ids"].clone()
        inputs_masked["xe_val_ids"][0, 0, 0] = _MASK_ID  # inject MASK_ID at [0,0,0]

        with torch.no_grad():
            batch_orig   = asm(**inputs_orig)
            batch_masked = asm(**inputs_masked)

        # Embedding at [0, 0, 0] must differ when MASK_ID is substituted
        assert not torch.allclose(batch_orig.xe[0, 0, 0], batch_masked.xe[0, 0, 0]), (
            "§2.3.5: substituting MASK_ID at a position must change its embedding"
        )

        # Manual verification: E(key_id) + E(MASK_ID) + pos_emb(pos_id)
        k = inputs_orig["xe_key_ids"][0, 0, 0]
        p = inputs_orig["xe_pos_ids"][0, 0, 0]
        with torch.no_grad():
            expected = (
                asm.E(k.unsqueeze(0))
                + asm.E(torch.tensor([_MASK_ID]))
                + asm.pos_emb(p.unsqueeze(0))
            ).squeeze(0)

        assert torch.allclose(batch_masked.xe[0, 0, 0], expected, atol=1e-6), (
            "§2.3.5: masked position embedding must equal E(key_id)+E(MASK_ID)+pos_emb(pos)"
        )

    def test_xa_is_independent_of_xe(self) -> None:
        """§2.3.1: xa and xe are independent paths through the shared embedding table E.

        Despite sharing embedding table E, changing xa inputs must not affect xe,
        and changing xe inputs must not affect xa. The two paths are computed
        independently in a single forward pass.

        This verifies: TD-003 note — xa path is genuinely live, not a placeholder
        reusing xe data.
        """
        spec, asm = _make_assembler()
        asm.eval()

        inputs_a = _make_inputs(spec, with_mask=False, seed=10)

        # Change only xa inputs — xe inputs stay identical
        inputs_b = dict(inputs_a)
        torch.manual_seed(99)
        inputs_b["xa_key_ids"] = torch.randint(
            spec.key_start, spec.key_start + spec.key_size, (_BATCH, _NA)
        )
        inputs_b["xa_val_ids"] = torch.randint(
            spec.value_start, spec.value_start + spec.value_size, (_BATCH, _NA)
        )

        with torch.no_grad():
            batch_a = asm(**inputs_a)
            batch_b = asm(**inputs_b)

        # xa must change (different xa inputs)
        assert not torch.allclose(batch_a.xa, batch_b.xa), (
            "§2.3.1: changing xa_key_ids/xa_val_ids must change xa output"
        )

        # xe must NOT change (identical xe inputs, shared E but independent lookup)
        assert torch.allclose(batch_a.xe, batch_b.xe), (
            "§2.3.1: changing xa inputs must NOT affect xe. "
            "xe = E(xe_key_ids)+E(xe_val_ids)+pos_emb — no dependency on xa inputs."
        )


# ---------------------------------------------------------------------------
# TestGradientFlow — verify all trainable parameters receive gradients
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradient flow through EmbeddingAssembler (§2.3.1)."""

    def test_gradient_flows_through_embedding_table(self) -> None:
        """§2.3.1: embedding table E.weight must receive non-zero gradients.

        E is the only trainable component in EmbeddingAssembler. A zero or
        missing gradient means embeddings are frozen and cannot be learned.
        """
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=False)

        batch = asm(**inputs)
        loss = batch.xa.sum() + batch.xe.sum()
        loss.backward()

        assert asm.E.weight.grad is not None, (
            "E.weight must have a gradient after backward()"
        )
        assert not torch.all(asm.E.weight.grad == 0), (
            "E.weight gradient must be non-zero — embedding table must be trainable"
        )

    def test_gradient_flows_through_xa_path(self) -> None:
        """§2.3.1: loss from xa alone must flow back to E.weight.

        The xa path (profile tokens) must be fully differentiable end-to-end.
        """
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=False)

        batch = asm(**inputs)
        loss = batch.xa.sum()   # only xa in loss
        loss.backward()

        assert asm.E.weight.grad is not None, "xa path: E.weight must have grad"
        assert not torch.all(asm.E.weight.grad == 0), (
            "xa path: E.weight gradient must be non-zero"
        )

    def test_gradient_flows_through_xe_path(self) -> None:
        """§2.3.1: loss from xe alone must flow back to E.weight.

        The xe path (event tokens) must be fully differentiable end-to-end.
        """
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=False)

        batch = asm(**inputs)
        loss = batch.xe.sum()   # only xe in loss
        loss.backward()

        assert asm.E.weight.grad is not None, "xe path: E.weight must have grad"
        assert not torch.all(asm.E.weight.grad == 0), (
            "xe path: E.weight gradient must be non-zero"
        )


# ---------------------------------------------------------------------------
# TestPaperSpecifications — exact values and structural constraints
# ---------------------------------------------------------------------------


class TestPaperSpecifications:
    """Verify exact paper values and ADR 002 structural constraints."""

    def test_position_embedding_is_not_a_parameter(self) -> None:
        """§2.3.1 / Vaswani et al.: sinusoidal PosEmb must have no trainable parameters.

        Equation 1 uses sinusoidal positional encoding (fixed, not learned).
        A learnable positional embedding would add parameters not in the paper.
        """
        _, asm = _make_assembler()
        n_pos_params = sum(p.numel() for p in asm.pos_emb.parameters())
        assert n_pos_params == 0, (
            f"§2.3.1: pos_emb must have 0 trainable parameters (sinusoidal, fixed). "
            f"Got {n_pos_params}. Use register_buffer(), not nn.Parameter or nn.Embedding."
        )

    def test_embedding_table_size_matches_vocab_spec(self) -> None:
        """§2.2 / ADR 002: E.weight must have shape (total_embedding_vocab_size, d_model).

        total_embedding_vocab_size = N_SPECIAL + key_size + value_size = 24 (test fixture).
        d_model = 192 (PRAGMA-S, Table 1).
        """
        spec, asm = _make_assembler()
        assert asm.E.weight.shape[0] == spec.total_embedding_vocab_size, (
            f"§2.2: E.weight must have {spec.total_embedding_vocab_size} rows "
            f"(total_embedding_vocab_size), got {asm.E.weight.shape[0]}"
        )
        assert asm.E.weight.shape[1] == _CONFIG.d_model, (
            f"Table 1: E.weight must have {_CONFIG.d_model} cols (d_model), "
            f"got {asm.E.weight.shape[1]}"
        )

    def test_validate_passes_for_assembler_output(self) -> None:
        """ADR 002: AssembledBatch returned by EmbeddingAssembler must pass validate().

        EmbeddingAssembler calls validate() internally before returning.
        This test verifies the invariant holds for a real forward pass.
        """
        spec, asm = _make_assembler()
        inputs = _make_inputs(spec, with_mask=True)

        with torch.no_grad():
            batch = asm(**inputs)

        # validate() is called inside forward(); explicit call must also pass
        batch.validate(_CONFIG)

    def test_pos_emb_covers_profile_positions(self) -> None:
        """§2.4: sinusoidal table must cover max_profile_tokens (200),
        not just max_event_tokens (24).

        Profile tokens (xa path) may have position IDs up to max_profile_tokens - 1.
        If the sinusoidal table is sized to max_event_tokens only, positions >= 24
        cause an index-out-of-bounds error at runtime.
        """
        spec, asm = _make_assembler()
        asm.eval()

        # xa_pos_ids at max_profile_tokens - 1 (the highest valid profile position)
        max_prof = _CONFIG.max_profile_tokens  # 200
        inputs = _make_inputs(spec, with_mask=False)
        inputs["xa_pos_ids"] = torch.full(
            (_BATCH, _NA), max_prof - 1, dtype=torch.long
        )

        with torch.no_grad():
            batch = asm(**inputs)   # must not raise IndexError

        assert batch.xa.shape == (_BATCH, _NA, _D), (
            f"§2.4: xa shape must be ({_BATCH}, {_NA}, {_D}) "
            f"even with pos_id={max_prof - 1}"
        )

    def test_target_localisation_is_global_minus_value_start(self) -> None:
        """ADR 002: local_id = global_id - value_start (VocabularyMap arithmetic).

        For our fixture: value_start = 7. If a global value ID is 10, the local
        target ID must be 10 - 7 = 3.
        """
        spec, asm = _make_assembler()
        asm.eval()

        inputs = _make_inputs(spec, with_mask=False)
        # Manually set one xe_val_id to a known value, then mask it
        known_global = spec.value_start + 3   # = 10 for our fixture
        inputs["xe_val_ids"] = torch.full((_BATCH, _NE, _NI), known_global, dtype=torch.long)
        inputs["target_ids"] = inputs["xe_val_ids"].clone()
        inputs["mask"] = torch.ones(_BATCH, _NE, _NI, dtype=torch.bool)

        with torch.no_grad():
            batch = asm(**inputs)

        expected_local = known_global - spec.value_start   # = 3
        assert (batch.targets[batch.mlm_mask] == expected_local).all(), (
            f"ADR 002: global_id={known_global} → local_id={expected_local}. "
            f"Got {batch.targets[batch.mlm_mask].unique().tolist()}"
        )
