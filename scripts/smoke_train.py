#!/usr/bin/env python3
"""Smoke training script — proves PRAGMA is trainable.

Runs 20 steps on synthetic data. Loss must decrease.
No GPU required. Uses PRAGMA-S (10M params).

Usage:
    python scripts/smoke_train.py

Expected output:
    Step 0:  loss ~10.0 (random initialisation)
    Step 19: loss < step 0 loss (model is learning)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import PRAGMA, PRAGMAConfig
from src.model.assembler import EmbeddingAssembler
from src.model.assembled_batch import AssembledBatch
from src.masking import MaskingStrategy
from src.tokenizer.vocabulary import VocabularySpec

def make_synthetic_batch(config, vocab_spec, batch=2, ne=5, ni=8, na=6):
    """Generate a synthetic training batch."""
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
    te = torch.cat([
        torch.zeros(batch, 1),
        torch.rand(batch, ne) * 100.0,
    ], dim=1)
    xt = torch.stack([
        torch.randint(0, 24, (batch, ne)),
        torch.randint(0, 7,  (batch, ne)),
        torch.randint(1, 32, (batch, ne)),
    ], dim=-1)
    return (xe_val_ids, xe_key_ids, xe_pos_ids,
            xa_val_ids, xa_key_ids, xa_pos_ids,
            ta, te, xt)

def main():
    torch.manual_seed(42)
    config = PRAGMAConfig.pragma_s()

    vocab_spec = VocabularySpec(
        special_tokens={"PAD": 0, "MASK": 1, "EVT": 2, "SEP": 3},
        key_start=4,
        key_size=config.key_vocab_size,
        value_start=4 + config.key_vocab_size,
        value_size=config.value_vocab_size,
        total_embedding_vocab_size=4 + config.key_vocab_size
                                     + config.value_vocab_size,
        field_key_ids={},
        field_value_ranges={},
    )

    assembler = EmbeddingAssembler(vocab_spec, config)
    model = PRAGMA(config)
    masker = MaskingStrategy(config)

    optimiser = torch.optim.AdamW(
        list(model.parameters()) + list(assembler.parameters()),
        lr=1e-4,
    )

    model.train()
    assembler.train()

    losses = []
    print("PRAGMA smoke training — 20 steps on synthetic data")
    print(f"Model: PRAGMA-S ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Assembler: ({sum(p.numel() for p in assembler.parameters()):,} params)")
    print()

    for step in range(20):
        optimiser.zero_grad()

        (xe_val_ids, xe_key_ids, xe_pos_ids,
         xa_val_ids, xa_key_ids, xa_pos_ids,
         ta, te, xt) = make_synthetic_batch(config, vocab_spec)

        masked_val_ids, target_ids, mlm_mask = masker.forward(
            xe_val_ids, xe_key_ids
        )

        if not mlm_mask.any():
            continue

        assembled = assembler.forward(
            xa_key_ids=xa_key_ids,
            xa_val_ids=xa_val_ids,
            xa_pos_ids=xa_pos_ids,
            ta=ta,
            xe_key_ids=xe_key_ids,
            xe_val_ids=masked_val_ids,
            xe_pos_ids=xe_pos_ids,
            xt=xt,
            te=te,
            target_ids=xe_val_ids,
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

        if "logits" not in output:
            print(f"Step {step:2d}: no masked positions — skipping")
            continue

        valid_targets = assembled.targets[assembled.mlm_mask]
        loss = model.mlm_head.compute_loss(output["logits"], valid_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(assembler.parameters()),
            max_norm=1.0,
        )
        optimiser.step()

        losses.append(loss.item())
        print(f"Step {step:2d}: loss = {loss.item():.4f}")

    print()
    if len(losses) >= 2:
        if losses[-1] < losses[0]:
            print(f"✓ Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
            print("PRAGMA is trainable on synthetic data.")
        else:
            print(f"✗ Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}")
            print("Check the training loop — something may be wrong.")
            sys.exit(1)
    else:
        print("Not enough steps with masked positions to evaluate.")
        sys.exit(1)

if __name__ == "__main__":
    main()
