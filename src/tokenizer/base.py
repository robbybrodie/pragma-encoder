"""Base tokeniser abstract class.

Defines the interface that all PRAGMA field tokenisers must implement.

PRAGMA uses a heterogeneous tokenisation strategy (Section 2.2) where
each field type is tokenised differently to preserve its statistical
properties. This abstract class enforces a consistent interface across
all tokeniser types (numerical, categorical, textual, temporal).

Key contract:
    - fit(data): Learn vocabulary or bucket boundaries from training data.
    - encode(value): Map a raw value to a token ID or sequence of IDs.
    - decode(token_ids): Invert encode for interpretability.
    - vocab_size: The number of distinct tokens this tokeniser can produce.

Reference: Ostroukhov et al. (2026), Section 2.2
"""

from abc import ABC, abstractmethod
from typing import Any, List, Union


class BaseTokenizer(ABC):
    """Abstract base class for all PRAGMA field tokenisers.

    Each subclass handles one field type (numerical, categorical,
    textual, or temporal) and is responsible for mapping raw field
    values to token IDs within a shared vocabulary segment.
    """

    @abstractmethod
    def fit(self, data: List[Any]) -> "BaseTokenizer":
        """Learn vocabulary or bucket boundaries from training data.

        Args:
            data: List of raw field values from the training corpus.

        Returns:
            Self, to allow method chaining.
        """
        ...

    @abstractmethod
    def encode(self, value: Any) -> Union[int, List[int]]:
        """Map a raw value to one or more token IDs.

        Args:
            value: A single raw field value.

        Returns:
            A single token ID (int) or a list of token IDs for
            multi-token fields (e.g. subword-tokenised text).
        """
        ...

    @abstractmethod
    def decode(self, token_ids: Union[int, List[int]]) -> Any:
        """Invert encode for interpretability and debugging.

        Args:
            token_ids: Token ID(s) produced by encode().

        Returns:
            The reconstructed raw value (may be approximate for
            numerical fields that use bucket midpoints).
        """
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of distinct tokens this tokeniser produces."""
        ...
