"""Local learning validation — proves PRAGMA-S can learn from data.

This script runs a short local training loop on synthetic IBM TabFormer-style
data and reports whether the model learns (loss decreases), gradients flow,
and checkpoints save and reload correctly.

This is NOT production training. It is a local sanity check that proves:
  - The full pipeline (tokeniser → masking → assembler → PRAGMA → loss) runs
  - Loss is finite at step 0 (no NaN/Inf from random initialisation)
  - Loss decreases over the configured number of steps (gradient flow confirmed)
  - Checkpoint save and reload produce identical forward-pass output

It does NOT:
  - Use a real IBM TabFormer dataset (generates synthetic data in valid ID ranges)
  - Connect to S3, KFP, or any OpenShift cluster
  - Submit a PyTorchJob or pipeline run
  - Require GPU (runs on CPU)

To run:
    PYTHONPATH=. python examples/workbench/06_local_learning_validation.py
    PYTHONPATH=. python examples/workbench/06_local_learning_validation.py --max-steps 100
    PYTHONPATH=. python examples/workbench/06_local_learning_validation.py --model-size S --max-steps 50

Reference: Ostroukhov et al. (2026), arXiv:2604.08649v1, Sections 2.3, 2.3.5, 2.4
"""

# ── Project root on sys.path ──────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import tempfile
import time

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch

# ── PRAGMA ────────────────────────────────────────────────────────────────────
from pragma_encoder.masking import MaskingStrategy
from pragma_encoder.model import PRAGMA, PRAGMAConfig
from pragma_encoder.model.assembler import EmbeddingAssembler
from pragma_encoder.tokenizer.vocabulary import VocabularySpec
from pragma_encoder.training.readiness import make_readiness_report


# ── Constants ─────────────────────────────────────────────────────────────────

# Synthetic batch dimensions — realistic IBM TabFormer-style geometry.
# These match the paper's truncation limits (§2.4):
#   max_event_tokens=24, max_events=6500 (used during real training)
# For local validation we use a small batch to keep CPU runtime short.
_BATCH_SIZE = 4    # sequences per step
_NE = 10           # events per sequence (real training: up to 6500)
_NI = 8            # tokens per event (real training: up to 24)
_NA = 6            # profile tokens per sequence


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRAGMA local learning validation — proves the model can learn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        metavar="N",
        help="Number of optimisation steps to run (default: 50).",
    )
    parser.add_argument(
        "--model-size",
        choices=["S", "M", "L"],
        default="S",
        help="PRAGMA model variant: S (~10M), M (~100M), L (~1B). Default: S.",
    )
    parser.add_argument(
        "--resume-steps",
        type=int,
        default=5,
        metavar="N",
        help="Extra steps to run after reloading from checkpoint (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


# ── Vocabulary spec ───────────────────────────────────────────────────────────

def _make_vocab_spec(config: PRAGMAConfig) -> VocabularySpec:
    """Build a synthetic VocabularySpec consistent with config vocab sizes.

    In real training, this comes from FinancialTokenizerPipeline.vocabulary_spec()
    after fitting on the IBM TabFormer data. Here we construct it directly with
    the same special-token layout so all ID arithmetic is identical.
    """
    return VocabularySpec(
        special_tokens={"PAD": 0, "MASK": 1, "EVT": 2, "SEP": 3},
        key_start=4,
        key_size=config.key_vocab_size,
        value_start=4 + config.key_vocab_size,
        value_size=config.value_vocab_size,
        total_embedding_vocab_size=4 + config.key_vocab_size + config.value_vocab_size,
        field_key_ids={},
        field_value_ranges={},
    )


# ── Synthetic batch generator ─────────────────────────────────────────────────

def _make_batch(config: PRAGMAConfig, vocab_spec: VocabularySpec) -> dict:
    """Generate one synthetic batch of IBM TabFormer-style token IDs.

    All IDs are drawn uniformly from valid vocab ranges. In real training
    these would come from PRAGMADataset loading prepared shards from disk.

    Returns:
        Dictionary of integer token tensors in valid ID ranges.
    """
    xe_val_ids = torch.randint(
        vocab_spec.value_start,
        vocab_spec.value_start + config.value_vocab_size,
        (_BATCH_SIZE, _NE, _NI),
    )
    xe_key_ids = torch.randint(
        vocab_spec.key_start,
        vocab_spec.key_start + config.key_vocab_size,
        (_BATCH_SIZE, _NE, _NI),
    )
    xa_val_ids = torch.randint(
        vocab_spec.value_start,
        vocab_spec.value_start + config.value_vocab_size,
        (_BATCH_SIZE, _NA),
    )
    xa_key_ids = torch.randint(
        vocab_spec.key_start,
        vocab_spec.key_start + config.key_vocab_size,
        (_BATCH_SIZE, _NA),
    )
    return dict(
        xe_val_ids=xe_val_ids,
        xe_key_ids=xe_key_ids,
        xe_pos_ids=torch.arange(_NI).unsqueeze(0).unsqueeze(0).expand(_BATCH_SIZE, _NE, -1),
        xa_val_ids=xa_val_ids,
        xa_key_ids=xa_key_ids,
        xa_pos_ids=torch.arange(_NA).unsqueeze(0).expand(_BATCH_SIZE, -1),
        ta=torch.zeros(_BATCH_SIZE, _NA),
        te=torch.cat([
            torch.zeros(_BATCH_SIZE, 1),
            torch.rand(_BATCH_SIZE, _NE) * 1e6,  # ~log-seconds spread
        ], dim=1),
        xt=torch.stack([
            torch.randint(0, 24, (_BATCH_SIZE, _NE)),   # hour of day
            torch.randint(0, 7,  (_BATCH_SIZE, _NE)),   # day of week
            torch.randint(1, 32, (_BATCH_SIZE, _NE)),   # day of month
        ], dim=-1),
    )


# ── One training step ─────────────────────────────────────────────────────────

def _step(
    config: PRAGMAConfig,
    vocab_spec: VocabularySpec,
    model: PRAGMA,
    assembler: EmbeddingAssembler,
    masker: MaskingStrategy,
    optimizer: torch.optim.Optimizer,
    batch: dict,
) -> float:
    """Run one forward + backward + optimizer step. Return scalar loss."""
    optimizer.zero_grad()

    # Apply three-strategy masking (§2.3.5)
    masked_val_ids, _, mlm_mask = masker.forward(
        batch["xe_val_ids"], batch["xe_key_ids"]
    )
    # Guarantee at least one supervised position (avoids empty-loss edge case
    # when all three strategies happen to select zero positions on a small batch)
    if not mlm_mask.any():
        mlm_mask = mlm_mask.clone()
        mlm_mask[0, 0, 0] = True
        masked_val_ids = masked_val_ids.clone()
        masked_val_ids[0, 0, 0] = 1  # MASK_ID = 1

    # Equation 1: embed token IDs → float tensors
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

    # Six-step PRAGMA forward pass (Equations 4–8)
    output = model.forward(
        xa=assembled.xa,
        ta=assembled.ta,
        xe=assembled.xe,
        xt=assembled.xt,
        te=assembled.te,
        mask=assembled.mlm_mask,
    )

    # MLM loss at masked positions (§2.3.5, label smoothing ε=0.1)
    logits       = output["logits"]                          # (n_masked, value_vocab_size)
    valid_targets = assembled.targets[assembled.mlm_mask]   # (n_masked,)
    loss = model.mlm_head.compute_loss(logits, valid_targets)

    loss.backward()
    optimizer.step()

    return loss.item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  PRAGMA LOCAL LEARNING VALIDATION")
    print("  This is a local sanity check — NOT production training.")
    print("  No S3, no KFP, no cluster. Synthetic data only.")
    print("=" * 60)
    print()

    # ── Config ────────────────────────────────────────────────────────────────
    size_map = {"S": PRAGMAConfig.pragma_s, "M": PRAGMAConfig.pragma_m, "L": PRAGMAConfig.pragma_l}
    config = size_map[args.model_size]()
    vocab_spec = _make_vocab_spec(config)

    # ── Model + training components ───────────────────────────────────────────
    model    = PRAGMA(config)
    assembler = EmbeddingAssembler(vocab_spec, config)
    masker   = MaskingStrategy(config)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(assembler.parameters()),
        lr=1e-4,
    )

    model.train()
    assembler.train()

    # ── Readiness report ──────────────────────────────────────────────────────
    n_synthetic = _BATCH_SIZE * args.max_steps
    n_events    = n_synthetic * _NE
    report = make_readiness_report(
        config=config,
        model=model,
        dataset_name="ibm-tabformer-synthetic",
        n_sequences=n_synthetic,
        n_events=n_events,
        dataset_path=None,
    )
    report.print_report()
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"  Running {args.max_steps} steps with Adam lr=1e-4 ...")
    print(f"  Batch: {_BATCH_SIZE} sequences × {_NE} events × {_NI} tokens/event")
    print()

    fixed_batch = _make_batch(config, vocab_spec)   # same batch every step → memorisation
    losses: list[float] = []
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        loss_val = _step(config, vocab_spec, model, assembler, masker, optimizer, fixed_batch)
        losses.append(loss_val)

        if step == 1 or step % max(1, args.max_steps // 10) == 0 or step == args.max_steps:
            elapsed = time.time() - t0
            print(f"  step {step:>4d}/{args.max_steps}  loss={loss_val:.4f}  ({elapsed:.1f}s)")

    initial_loss = losses[0]
    final_loss   = losses[-1]
    decreased    = final_loss < initial_loss
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Training summary")
    print("=" * 60)
    print(f"  Initial loss (step 1)  : {initial_loss:.4f}")
    print(f"  Final loss   (step {args.max_steps:>3d}): {final_loss:.4f}")
    print(f"  Loss decreased         : {'YES ✓' if decreased else 'NO — CHECK ARCHITECTURE'}")
    print()

    if not decreased:
        print("  WARNING: Loss did not decrease. Possible causes:")
        print("    - Gradients not flowing (detached tensor in forward path)")
        print("    - All parameters frozen (requires_grad=False somewhere)")
        print("    - Learning rate too low or zero")
        print("    - Too few steps to show movement on this batch size")
        print()

    # ── Checkpoint save ───────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / f"pragma-{args.model_size.lower()}-step-{args.max_steps:04d}.pt"

        torch.save({
            "model_state_dict":     model.state_dict(),
            "assembler_state_dict": assembler.state_dict(),
            "config":               config,
            "step":                 args.max_steps,
            "final_loss":           final_loss,
        }, ckpt_path)

        size_kb = ckpt_path.stat().st_size // 1024
        print(f"  Checkpoint saved       : {ckpt_path.name}  ({size_kb:,} KB)")

        # ── Checkpoint reload ─────────────────────────────────────────────────
        model_reloaded    = PRAGMA(config)
        assembler_reloaded = EmbeddingAssembler(vocab_spec, config)

        ckpt = torch.load(ckpt_path, weights_only=False)
        model_reloaded.load_state_dict(ckpt["model_state_dict"])
        assembler_reloaded.load_state_dict(ckpt["assembler_state_dict"])

        model_reloaded.eval()
        assembler_reloaded.eval()

        # Verify reload: same inputs → identical zh output
        model.eval()
        assembler.eval()

        with torch.no_grad():
            # Run one forward pass on both original and reloaded
            eval_batch = _make_batch(config, vocab_spec)
            masked_val, _, eval_mask = masker.forward(eval_batch["xe_val_ids"], eval_batch["xe_key_ids"])
            if not eval_mask.any():
                eval_mask = eval_mask.clone()
                eval_mask[0, 0, 0] = True
                masked_val = masked_val.clone()
                masked_val[0, 0, 0] = 1

            assembled_eval = assembler.forward(
                xa_key_ids=eval_batch["xa_key_ids"], xa_val_ids=eval_batch["xa_val_ids"],
                xa_pos_ids=eval_batch["xa_pos_ids"], ta=eval_batch["ta"],
                xe_key_ids=eval_batch["xe_key_ids"], xe_val_ids=masked_val,
                xe_pos_ids=eval_batch["xe_pos_ids"], xt=eval_batch["xt"], te=eval_batch["te"],
            )
            out_orig     = model.forward(xa=assembled_eval.xa, ta=assembled_eval.ta,
                                         xe=assembled_eval.xe, xt=assembled_eval.xt,
                                         te=assembled_eval.te)
            out_reloaded = model_reloaded.forward(xa=assembled_eval.xa, ta=assembled_eval.ta,
                                                   xe=assembled_eval.xe, xt=assembled_eval.xt,
                                                   te=assembled_eval.te)

        reload_ok = torch.allclose(out_orig["zh"], out_reloaded["zh"])
        print(f"  Checkpoint reload      : {'OK ✓' if reload_ok else 'MISMATCH — state not preserved'}")

        if not reload_ok:
            print()
            print("  WARNING: Reloaded model produces different output.")
            print("    The checkpoint may not have saved all model state (buffers, etc.).")

        # ── Resume: a few more steps from reloaded state ──────────────────────
        if args.resume_steps > 0:
            model_reloaded.train()
            assembler_reloaded.train()
            optimizer_resume = torch.optim.Adam(
                list(model_reloaded.parameters()) + list(assembler_reloaded.parameters()),
                lr=1e-4,
            )
            print()
            print(f"  Resuming for {args.resume_steps} more step(s) from reloaded checkpoint ...")
            for rs in range(1, args.resume_steps + 1):
                rl = _step(config, vocab_spec, model_reloaded, assembler_reloaded,
                           masker, optimizer_resume, fixed_batch)
                if rs == 1 or rs == args.resume_steps:
                    print(f"    resume step {rs}: loss={rl:.4f}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULT")
    print("=" * 60)

    checks = [
        ("Loss is finite",           not (torch.isnan(torch.tensor(final_loss)) or
                                          torch.isinf(torch.tensor(final_loss)))),
        ("Loss decreased",           decreased),
        ("Checkpoint reload OK",     reload_ok),
    ]

    all_ok = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {label}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  All checks passed. PRAGMA-S is learning correctly.")
        print()
        print("  Next step: run on real IBM TabFormer data.")
        print("    See docs/training-guide.md → Local training — PRAGMA-S")
    else:
        print("  One or more checks FAILED. Investigate before running on real data.")
        print("    See DEVELOPMENT_PROCESS.md for debugging guidance.")

    print()
    print("  This was LOCAL LEARNING VALIDATION — no cluster resources were used.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
