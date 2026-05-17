"""PRAGMA Encoder — Open Source Implementation.

An independent implementation of the PRAGMA foundation model architecture
described in:

    Ostroukhov, M. et al. (2026). PRAGMA: Revolut Foundation Model.
    arXiv:2604.08649v1

This implementation is not affiliated with Revolut or NVIDIA.
No weights, training data, or proprietary code from Revolut are included.
Architecture is implemented from the paper description only.

Top-level package structure:

    src.tokenizer     — Key-value-time tokenisation (Section 2.2)
    src.encoders      — Three-encoder architecture (Sections 2.3.2–2.3.4)
    src.masking       — Three-strategy MLM masking (Section 2.3.5)
    src.model         — Full PRAGMA model assembly (Section 2.3)
    src.adaptation    — LoRA fine-tuning and linear probes (Section 3.1)
    src.training      — Training objective and data utilities (Section 2.3.5)
    src.evaluation    — Downstream task evaluation (Section 3)
"""
