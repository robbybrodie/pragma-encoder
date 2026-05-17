"""Test suite for the PRAGMA encoder implementation.

Tests cover:
    test_tokenizer.py  — Tokenisation correctness (Section 2.2)
    test_encoders.py   — Encoder forward passes and output shapes (Section 2.3)
    test_model.py      — Full PRAGMA model integration (Section 2.3)
    test_masking.py    — Three masking strategy correctness (Section 2.3.5)

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=html
"""
