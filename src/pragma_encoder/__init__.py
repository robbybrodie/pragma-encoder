"""PRAGMA Encoder — Open Source Implementation.

An independent implementation of the PRAGMA foundation model architecture
described in:

    Ostroukhov, M. et al. (2026). PRAGMA: Revolut Foundation Model.
    arXiv:2604.08649v1

This implementation is not affiliated with Revolut or NVIDIA.
No weights, training data, or proprietary code from Revolut are included.
Architecture is implemented from the paper description only.

Top-level package structure:

    pragma_encoder.tokenizer     — Key-value-time tokenisation (Section 2.2)
    pragma_encoder.encoders      — Three-encoder architecture (Sections 2.3.2–2.3.4)
    pragma_encoder.masking       — Three-strategy MLM masking (Section 2.3.5)
    pragma_encoder.model         — Full PRAGMA model assembly (Section 2.3)
    pragma_encoder.adaptation    — LoRA fine-tuning and linear probes (Section 3.1)
    pragma_encoder.training      — Training objective and data utilities (Section 2.3.5)
    pragma_encoder.evaluation    — Downstream task evaluation (Section 3)
    pragma_encoder.workbench     — Workbench training API and pipeline utilities
    pragma_encoder.data          — Dataset adapters and manifest (Section 2.4)
"""
