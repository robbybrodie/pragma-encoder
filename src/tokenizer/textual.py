"""Textual field tokeniser — BPE subword encoding.

Implements the textual tokenisation strategy from PRAGMA paper Section 2.2.

Free-text fields (e.g. merchant name, transaction description) are tokenised
using Byte-Pair Encoding (BPE) via the Hugging Face tokenizers library.
This produces a variable-length sequence of subword token IDs per field value.

PRAGMA uses BPE for textual fields to handle:
    - Out-of-vocabulary merchant names
    - Multi-language descriptions
    - Abbreviations and concatenated words common in banking data

The BPE vocabulary is trained on the financial text corpus to capture
domain-specific subwords (e.g. "PAYPAL", "AMZN", "CONTACTLESS").

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from typing import Any, List, Union

from .base import BaseTokenizer


class TextualTokenizer(BaseTokenizer):
    """BPE subword tokeniser for free-text fields.

    Wraps a Hugging Face BPE tokenizer trained on the financial
    text corpus. Produces a variable-length list of token IDs.

    Args:
        vocab_size: BPE vocabulary size. Default: 8_000.
        max_length: Maximum number of tokens per field value.
                    Values are truncated to this length. Default: 16.
    """

    def __init__(self, vocab_size: int = 8_000, max_length: int = 16):
        self._vocab_size = vocab_size
        self.max_length = max_length
        self._tokenizer = None
        self._fitted = False

    def fit(self, data: List[Any]) -> "TextualTokenizer":
        """Train a BPE tokeniser on the provided text corpus.

        Args:
            data: List of raw text strings from the training corpus.

        Returns:
            Self.
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self._vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[MISSING]"],
        )
        texts = [str(v) for v in data if v is not None]
        tokenizer.train_from_iterator(texts, trainer=trainer)

        self._tokenizer = tokenizer
        self._fitted = True
        return self

    def encode(self, value: Any) -> List[int]:
        """Encode a text value to a list of subword token IDs.

        Args:
            value: Raw text string.

        Returns:
            List of token IDs, truncated to max_length.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")
        if value is None:
            return [self._tokenizer.token_to_id("[MISSING]")]
        encoding = self._tokenizer.encode(str(value))
        return encoding.ids[: self.max_length]

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        """Decode token IDs back to a text string.

        Args:
            token_ids: List of subword token IDs.

        Returns:
            Decoded text string.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before decode().")
        ids = [token_ids] if isinstance(token_ids, int) else token_ids
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        """BPE vocabulary size."""
        return self._vocab_size
