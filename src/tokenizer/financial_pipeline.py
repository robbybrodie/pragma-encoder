"""Financial-domain tokenisation pipeline.

A pre-configured TokenizerPipeline for standard financial transaction fields
as described in the PRAGMA paper (Section 2.2) and applicable to retail
banking transaction histories.

This pipeline pre-registers tokenisers for the core transaction fields used
in the PRAGMA pretraining corpus:
    - amount_local        (NumericalTokenizer)
    - amount_usd          (NumericalTokenizer)
    - merchant_category   (CategoricalTokenizer)
    - currency            (CategoricalTokenizer)
    - country             (CategoricalTokenizer)
    - transaction_type    (CategoricalTokenizer)
    - channel             (CategoricalTokenizer)
    - merchant_name       (TextualTokenizer — BPE)
    - description         (TextualTokenizer — BPE)
    - timestamp           (TemporalTokenizer)

Callers fit the pipeline on their training corpus before use.

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from .categorical import CategoricalTokenizer
from .numerical import NumericalTokenizer
from .pipeline import TokenizerPipeline
from .temporal import TemporalTokenizer
from .textual import TextualTokenizer


def build_financial_pipeline(
    n_amount_buckets: int = 100,
    bpe_vocab_size: int = 8_000,
    bpe_max_length: int = 16,
    n_log_buckets: int = 128,
) -> TokenizerPipeline:
    """Construct a TokenizerPipeline pre-configured for financial transactions.

    All tokenisers are unfitted. Call pipeline.field_tokenizers[field].fit(data)
    for each field before use, or use a training harness to fit all at once.

    Args:
        n_amount_buckets: Percentile buckets for amount fields. Default: 100.
        bpe_vocab_size: BPE vocabulary size for text fields. Default: 8_000.
        bpe_max_length: Max subword tokens per text field. Default: 16.
        n_log_buckets: Log-seconds buckets for temporal tokeniser. Default: 128.

    Returns:
        A configured (but unfitted) TokenizerPipeline.
    """
    field_tokenizers = {
        "amount_local": NumericalTokenizer(n_buckets=n_amount_buckets),
        "amount_usd": NumericalTokenizer(n_buckets=n_amount_buckets),
        "merchant_category": CategoricalTokenizer(min_freq=5),
        "currency": CategoricalTokenizer(),
        "country": CategoricalTokenizer(),
        "transaction_type": CategoricalTokenizer(),
        "channel": CategoricalTokenizer(),
        "merchant_name": TextualTokenizer(
            vocab_size=bpe_vocab_size, max_length=bpe_max_length
        ),
        "description": TextualTokenizer(
            vocab_size=bpe_vocab_size, max_length=bpe_max_length
        ),
    }

    temporal_tokenizer = TemporalTokenizer(n_log_buckets=n_log_buckets)

    return TokenizerPipeline(
        field_tokenizers=field_tokenizers,
        temporal_tokenizer=temporal_tokenizer,
    )


# Convenience alias — matches the PRAGMA paper naming
FinancialTokenizerPipeline = build_financial_pipeline
