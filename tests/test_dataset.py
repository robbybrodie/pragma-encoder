"""Tests for PragmaDataset temporal coordinate computation.

DEF-003: te must be log-seconds to most recent event (Equation 2, §2.2),
not the inter-event gap (the current bug).

Paper statement (§2.3.4, Equation 2):
    t = elapsed time since most recent event, in seconds
    t' = 8 · ln(1 + t/8)
    te[i] = log-seconds from event i to the most recent event

With events sorted oldest-first (e0 < e1 < e2, where e2 is most recent):
    te[USR] = 0.0                           (always, position 0)
    te[e0]  = T(t_last - t0)  oldest  → largest t'
    te[e1]  = T(t_last - t1)
    te[e2]  = T(0)             most recent → t' = 0.0

Buggy behaviour (inter-event gap):
    te[e0] = T(0) = 0.0           ← wrong (no previous event)
    te[e1] = T(t1 - t0)
    te[e2] = T(t2 - t1)           ← wrong (≠ 0 when t2 > t1)

Reference: Ostroukhov et al. (2026), §2.2 Equation 2, §2.3.4
Every expected value derives from docs/paper/key-numbers.md:
    temporal_transform = 8 · ln(1 + t/8)   §2.2
"""

import math
import pickle

import pytest

from pragma_encoder.data.pragma_dataset import PragmaDataset
from pragma_encoder.tokenizer.categorical import CategoricalTokenizer
from pragma_encoder.tokenizer.pipeline import TokenizerPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pipeline() -> TokenizerPipeline:
    """Minimal fitted pipeline with one categorical field (currency)."""
    currency = CategoricalTokenizer()
    currency.fit(["USD"])
    return TokenizerPipeline(field_tokenizers={"currency": currency})


def _make_csv(path) -> None:
    """Write a minimal TabFormer-format CSV with one customer and 3 events.

    Events are spaced exactly 1 day apart (86400 s):
        e0: 2024-01-01 00:00  (oldest, 2 days before most recent)
        e1: 2024-01-02 00:00  (middle, 1 day before most recent)
        e2: 2024-01-03 00:00  (most recent, t=0)

    Only the columns read by TabFormerAdapter are required.
    """
    rows = [
        "User,Card,Year,Month,Day,Time,Amount,Use Chip,"
        "Merchant Name,Merchant City,Merchant State,MCC,Errors?,Is Fraud?",
        "0,1,2024,1,1,00:00,$1.00,Chip Transaction,"
        "Merchant A,City A,CA,1234,,No",
        "0,1,2024,1,2,00:00,$2.00,Chip Transaction,"
        "Merchant B,City B,CA,1234,,No",
        "0,1,2024,1,3,00:00,$3.00,Chip Transaction,"
        "Merchant C,City C,CA,1234,,No",
    ]
    path.write_text("\n".join(rows))


@pytest.fixture()
def dataset(tmp_path):
    """PragmaDataset with one customer (3 events, 1-day gaps). val split."""
    pipeline = _make_pipeline()
    vocab_path = tmp_path / "vocab.pkl"
    vocab_path.write_bytes(pickle.dumps(pipeline))

    csv_path = tmp_path / "transactions.csv"
    _make_csv(csv_path)

    # With 1 customer: int(1*0.8)=0, so train is empty, val has the customer.
    return PragmaDataset(csv_path, vocab_path, split="val", ne_max=5)


# ---------------------------------------------------------------------------
# Constants from key-numbers.md / Equation 2
# ---------------------------------------------------------------------------

_ONE_DAY_S = 86_400.0  # seconds in one day — used to derive expected te values


def _eq2(t: float) -> float:
    """Equation 2 (key-numbers.md §2.2): t' = 8·ln(1 + t/8)."""
    return 8.0 * math.log(1.0 + t / 8.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTemporalCoordinates:
    """DEF-003 — PragmaDataset must emit te as time-to-most-recent.

    Paper: §2.2 Equation 2, §2.3.4 History Encoder te specification.
    """

    def test_usr_position_is_always_zero(self, dataset):
        """[USR] position (te[0]) is always 0.0 — §2.3.4.

        The [USR] token has no event timestamp; its temporal coordinate is
        defined as 0 throughout the paper.
        """
        te = dataset[0]["te"]
        assert te[0].item() == pytest.approx(0.0), (
            "[USR] te[0] must be 0.0 (§2.3.4)"
        )

    def test_most_recent_event_has_zero_temporal_coord(self, dataset):
        """Most recent event (e2, te[3]) must have te ≈ 0.0 (Equation 2).

        t = ts_last - ts_last = 0  →  t' = 8·ln(1+0/8) = 0.0

        Buggy (inter-event gap): te[3] = T(t2 - t1) = T(86400) ≈ 74.2  ≠ 0.
        key-numbers.md: temporal_transform = 8 · ln(1 + t/8) at t=0 → 0.0
        """
        te = dataset[0]["te"]
        # te[3] = position of e2 ([USR] at 0 shifts event indices by +1)
        assert te[3].item() == pytest.approx(0.0, abs=1e-6), (
            "Most recent event te must be 0.0 — t=ts_last-ts_last=0 → T(0)=0 (Eq 2). "
            f"Got {te[3].item():.6f}"
        )

    def test_older_events_have_larger_temporal_coords(self, dataset):
        """te must decrease monotonically from oldest event to most recent.

        Oldest event is furthest from most recent → largest t → largest t'.
        With e0 < e1 < e2 (oldest-to-newest): te[e0] > te[e1] > te[e2] ≈ 0.

        Buggy (inter-event gap): te[e0]=0 < te[e1] — monotonicity inverted.
        """
        te = dataset[0]["te"]
        # te[1]=e0 (oldest), te[2]=e1 (middle), te[3]=e2 (most recent)
        assert te[1].item() > te[2].item() > te[3].item(), (
            "te must decrease monotonically oldest→newest (§2.3.4). "
            f"Got te[e0]={te[1].item():.4f}, te[e1]={te[2].item():.4f}, "
            f"te[e2]={te[3].item():.4f}"
        )

    def test_equation_2_at_known_gap(self, dataset):
        """te[e0] = 8·ln(1 + 172800/8) for e0 that is 2 days before most recent.

        t = 2 * 86400 = 172800 s → t' = 8 * ln(1 + 172800/8) = 8 * ln(21601)
        key-numbers.md: temporal_transform = 8 · ln(1 + t/8)  §2.2
        """
        te = dataset[0]["te"]
        t_e0 = 2.0 * _ONE_DAY_S           # e0 is 2 days before most recent
        expected = _eq2(t_e0)              # Equation 2 from key-numbers.md
        assert te[1].item() == pytest.approx(expected, rel=1e-5), (
            f"te[e0] must equal T(2 days)=8·ln(1+172800/8)={expected:.4f}. "
            f"Got {te[1].item():.4f}"
        )

    def test_equation_2_at_one_day_gap(self, dataset):
        """te[e1] = 8·ln(1 + 86400/8) for e1 that is 1 day before most recent.

        t = 86400 s → t' = 8 * ln(1 + 86400/8) = 8 * ln(10801)
        key-numbers.md: temporal_transform = 8 · ln(1 + t/8)  §2.2
        """
        te = dataset[0]["te"]
        t_e1 = 1.0 * _ONE_DAY_S           # e1 is 1 day before most recent
        expected = _eq2(t_e1)
        assert te[2].item() == pytest.approx(expected, rel=1e-5), (
            f"te[e1] must equal T(1 day)=8·ln(1+86400/8)={expected:.4f}. "
            f"Got {te[2].item():.4f}"
        )
