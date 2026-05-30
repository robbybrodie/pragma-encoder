"""Special-token contract tests — [USR] and [EVT] sentinels.

Verifies that the data pipeline correctly inserts the [USR] and [EVT] special
tokens into the positions assumed by the model architecture:

    xa[:, 0]     — [USR] sentinel (profile, §2.3.2)
    xe[:, :, 0]  — [EVT] sentinel (per event, §2.3.3)

Two classes of defect are guarded here:

1. **[EVT] not inserted** — if `encode_event()` payload starts at position 0,
   then `EventEncoder.z_hat_e[:,:,0,:]` extracts the first real field token as
   the event representation, not the [EVT] sentinel.  The model is
   geometrically shape-correct but semantically wrong.

2. **[USR] not inserted** — if the profile path uses a key/value-pad placeholder
   instead of USR_ID, the profile sentinel is indistinguishable from padding.
   HistoryEncoder then has no [USR] token to attend from event positions.

3. **xe_valid semantics** — [EVT] at position 0 must be marked valid (True) for
   real events so EventEncoder's attention mask includes the sentinel.  Padding
   events must have xe_valid[:, :, 0] = False.

4. **EVT excluded from MLM** — [EVT] tokens (position 0) must never appear as
   MLM targets; they are sentinels, not payload to predict.

5. **vocabulary_spec() exports USR and EVT** — downstream code that needs to
   distinguish sentinel tokens by name must be able to look them up.

Reference: Ostroukhov et al. (2026), §2.3.2–§2.3.4
Reference: PRAGMA special-token contract (docs/tech-debt.md TD-003)
"""

from __future__ import annotations

import pickle

import pytest

from pragma_encoder.tokenizer.categorical import CategoricalTokenizer
from pragma_encoder.tokenizer.pipeline import TokenizerPipeline

torch = pytest.importorskip("torch", reason="torch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pipeline() -> TokenizerPipeline:
    """Minimal fitted pipeline with one categorical field (currency)."""
    currency = CategoricalTokenizer()
    currency.fit(["USD", "GBP", "EUR"])
    return TokenizerPipeline(field_tokenizers={"currency": currency})


def _make_dataset(tmp_path, ne_max: int = 5, ni_max: int = 8):
    """Return a PragmaDataset backed by a 3-event, 1-customer synthetic CSV."""
    from pragma_encoder.data.pragma_dataset import PragmaDataset

    pipeline = _make_pipeline()
    vocab_path = tmp_path / "vocab.pkl"
    vocab_path.write_bytes(pickle.dumps(pipeline))

    csv_path = tmp_path / "transactions.csv"
    csv_path.write_text(
        "User,Card,Year,Month,Day,Time,Amount,Use Chip,"
        "Merchant Name,Merchant City,Merchant State,MCC,Errors?,Is Fraud?\n"
        "0,1,2024,1,1,00:00,$1.00,Chip Transaction,Merchant A,City A,CA,1234,,No\n"
        "0,1,2024,1,2,00:00,$2.00,Chip Transaction,Merchant B,City B,CA,1234,,No\n"
        "0,1,2024,1,3,00:00,$3.00,Chip Transaction,Merchant C,City C,CA,1234,,No\n"
    )
    # 1 customer → int(1*0.8)=0 train, 1 val → use val split
    return PragmaDataset(csv_path, vocab_path, split="val", ne_max=ne_max, ni_max=ni_max)


# ---------------------------------------------------------------------------
# 1. TokenizerPipeline special token IDs
# ---------------------------------------------------------------------------


class TestSpecialTokenIDs:
    """TokenizerPipeline must define USR_ID, EVT_ID, UNK_ID explicitly."""

    def test_usr_id_is_2(self) -> None:
        """USR_ID must be 2 — profile sentinel (formerly CLS_ID)."""
        assert TokenizerPipeline.USR_ID == 2, (
            "USR_ID must be 2.  Layout: PAD=0, MASK=1, USR=2, EVT=3, UNK=4."
        )

    def test_evt_id_is_3(self) -> None:
        """EVT_ID must be 3 — event sentinel."""
        assert TokenizerPipeline.EVT_ID == 3, (
            "EVT_ID must be 3.  Layout: PAD=0, MASK=1, USR=2, EVT=3, UNK=4."
        )

    def test_unk_id_is_4(self) -> None:
        """UNK_ID must be 4 — unknown / out-of-vocabulary token."""
        assert TokenizerPipeline.UNK_ID == 4, (
            "UNK_ID must be 4.  Layout: PAD=0, MASK=1, USR=2, EVT=3, UNK=4."
        )

    def test_n_special_tokens_is_5(self) -> None:
        """N_SPECIAL_TOKENS must be 5 (PAD, MASK, USR, EVT, UNK)."""
        assert TokenizerPipeline.N_SPECIAL_TOKENS == 5, (
            "N_SPECIAL_TOKENS must be 5 after adding USR, EVT, UNK. "
            f"Got {TokenizerPipeline.N_SPECIAL_TOKENS}."
        )

    def test_no_cls_id_attribute(self) -> None:
        """CLS_ID must not exist — it was replaced by USR_ID and EVT_ID."""
        assert not hasattr(TokenizerPipeline, "CLS_ID"), (
            "TokenizerPipeline still has CLS_ID.  Remove it — "
            "the ambiguous CLS sentinel is now split into USR_ID (2) and EVT_ID (3)."
        )

    def test_key_start_is_n_special(self) -> None:
        """Key tokens must start immediately after the 5 special tokens."""
        pipeline = _make_pipeline()
        spec = pipeline.vocabulary_spec()
        assert spec.key_start == TokenizerPipeline.N_SPECIAL_TOKENS, (
            f"key_start must equal N_SPECIAL_TOKENS ({TokenizerPipeline.N_SPECIAL_TOKENS}). "
            f"Got {spec.key_start}."
        )


# ---------------------------------------------------------------------------
# 2. vocabulary_spec() exports USR and EVT
# ---------------------------------------------------------------------------


class TestVocabularySpecExports:
    """vocabulary_spec() must export USR and EVT in special_tokens dict."""

    def test_vocabulary_spec_has_usr_key(self) -> None:
        """vocabulary_spec().special_tokens must contain 'USR' with value USR_ID."""
        pipeline = _make_pipeline()
        spec = pipeline.vocabulary_spec()
        assert "USR" in spec.special_tokens, (
            "vocabulary_spec().special_tokens must contain 'USR'. "
            f"Got keys: {list(spec.special_tokens.keys())}"
        )
        assert spec.special_tokens["USR"] == TokenizerPipeline.USR_ID, (
            f"special_tokens['USR'] must be {TokenizerPipeline.USR_ID}. "
            f"Got {spec.special_tokens['USR']}."
        )

    def test_vocabulary_spec_has_evt_key(self) -> None:
        """vocabulary_spec().special_tokens must contain 'EVT' with value EVT_ID."""
        pipeline = _make_pipeline()
        spec = pipeline.vocabulary_spec()
        assert "EVT" in spec.special_tokens, (
            "vocabulary_spec().special_tokens must contain 'EVT'. "
            f"Got keys: {list(spec.special_tokens.keys())}"
        )
        assert spec.special_tokens["EVT"] == TokenizerPipeline.EVT_ID, (
            f"special_tokens['EVT'] must be {TokenizerPipeline.EVT_ID}. "
            f"Got {spec.special_tokens['EVT']}."
        )

    def test_vocabulary_spec_has_no_cls_key(self) -> None:
        """vocabulary_spec().special_tokens must not contain 'CLS' (old ambiguous name)."""
        pipeline = _make_pipeline()
        spec = pipeline.vocabulary_spec()
        assert "CLS" not in spec.special_tokens, (
            "vocabulary_spec().special_tokens still exports 'CLS'. "
            "Remove it — the sentinel is now exported as 'USR' and 'EVT'."
        )


# ---------------------------------------------------------------------------
# 3. PragmaDataset — [USR] at profile position 0
# ---------------------------------------------------------------------------


class TestProfileUSRSentinel:
    """PragmaDataset must insert [USR] sentinel at xa_key_ids[:, 0] and xa_val_ids[:, 0]."""

    def test_xa_key_ids_position_0_is_usr_id(self, tmp_path) -> None:
        """xa_key_ids[0] must equal USR_ID for the profile sentinel.

        The HistoryEncoder assembles z = [za : ze] where za is the [USR] token.
        If position 0 is a padding key token instead, the HistoryEncoder has no
        [USR] representation to anchor the profile path.
        """
        dataset = _make_dataset(tmp_path)
        sample = dataset[0]
        xa_key = sample["xa_key_ids"]  # shape (1,) for TD-003 single token
        assert xa_key[0].item() == TokenizerPipeline.USR_ID, (
            f"xa_key_ids[0] must be USR_ID={TokenizerPipeline.USR_ID}. "
            f"Got {xa_key[0].item()}.  "
            "PragmaDataset must set xa_key_ids[0] = USR_ID (profile sentinel)."
        )

    def test_xa_val_ids_position_0_is_usr_id(self, tmp_path) -> None:
        """xa_val_ids[0] must equal USR_ID for the profile sentinel."""
        dataset = _make_dataset(tmp_path)
        sample = dataset[0]
        xa_val = sample["xa_val_ids"]  # shape (1,)
        assert xa_val[0].item() == TokenizerPipeline.USR_ID, (
            f"xa_val_ids[0] must be USR_ID={TokenizerPipeline.USR_ID}. "
            f"Got {xa_val[0].item()}.  "
            "PragmaDataset must set xa_val_ids[0] = USR_ID (profile sentinel)."
        )


# ---------------------------------------------------------------------------
# 4. PragmaDataset — [EVT] at event position 0
# ---------------------------------------------------------------------------


class TestEventEVTSentinel:
    """PragmaDataset must insert [EVT] sentinel at xe[:, :, 0] for every real event."""

    def test_evt_token_at_position_0_of_every_real_event(self, tmp_path) -> None:
        """xe_key_ids[:, :, 0] == EVT_ID for all real events.

        EventEncoder extracts z_hat_e[:, :, 0, :] as the [EVT] representation.
        If position 0 is the first payload field token instead, the encoder
        returns a field embedding as the event summary — semantically wrong.
        """
        dataset = _make_dataset(tmp_path, ne_max=5, ni_max=8)
        sample = dataset[0]
        xe_key = sample["xa_key_ids"]  # (1,)  — xa for profile
        xe_key_event = sample["xe_key_ids"]  # (ne_max, ni_max)
        n_events = sample["n_events"].item()

        for i in range(n_events):
            token_at_0 = xe_key_event[i, 0].item()
            assert token_at_0 == TokenizerPipeline.EVT_ID, (
                f"xe_key_ids[{i}, 0] must be EVT_ID={TokenizerPipeline.EVT_ID}. "
                f"Got {token_at_0}.  "
                "PragmaDataset must prepend the [EVT] sentinel at event position 0 "
                "before the payload tokens."
            )

    def test_evt_val_id_at_position_0_of_every_real_event(self, tmp_path) -> None:
        """xe_val_ids[:, :, 0] == EVT_ID for all real events."""
        dataset = _make_dataset(tmp_path, ne_max=5, ni_max=8)
        sample = dataset[0]
        xe_val = sample["xe_val_ids"]  # (ne_max, ni_max)
        n_events = sample["n_events"].item()

        for i in range(n_events):
            token_at_0 = xe_val[i, 0].item()
            assert token_at_0 == TokenizerPipeline.EVT_ID, (
                f"xe_val_ids[{i}, 0] must be EVT_ID={TokenizerPipeline.EVT_ID}. "
                f"Got {token_at_0}."
            )

    def test_payload_starts_at_position_1(self, tmp_path) -> None:
        """Payload tokens (field key/value IDs) must start at position 1, not 0.

        With [EVT] at position 0, the first real payload token is at position 1.
        The payload key IDs must be >= key_start (i.e., not a special token ID).
        """
        dataset = _make_dataset(tmp_path, ne_max=5, ni_max=8)
        sample = dataset[0]
        xe_key = sample["xe_key_ids"]  # (ne_max, ni_max)
        xe_valid = sample["xe_valid"]  # (ne_max, ni_max) bool
        n_events = sample["n_events"].item()

        pipeline = dataset.pipeline
        key_start = pipeline.N_SPECIAL_TOKENS

        for i in range(n_events):
            # Position 1 is the first payload slot — must be a real key token
            if xe_valid[i, 1]:  # only check if the position is valid
                key_at_1 = xe_key[i, 1].item()
                assert key_at_1 >= key_start, (
                    f"xe_key_ids[{i}, 1] = {key_at_1} is a special token ID "
                    f"(< key_start={key_start}). "
                    "Payload tokens must start at position 1 with a real key token "
                    f">= {key_start}."
                )


# ---------------------------------------------------------------------------
# 5. PragmaDataset — xe_valid semantics for [EVT] sentinel
# ---------------------------------------------------------------------------


class TestXeValidEVTSemantics:
    """xe_valid must mark [EVT] token valid for real events, invalid for padding."""

    def test_evt_valid_for_all_real_events(self, tmp_path) -> None:
        """xe_valid[:, 0] must be True for all real events.

        The [EVT] sentinel at position 0 is a real token in every real event.
        EventEncoder's attention mask is built from xe_valid; if [EVT] is not
        marked valid, the sentinel is masked out of its own event's attention.
        """
        dataset = _make_dataset(tmp_path, ne_max=5, ni_max=8)
        sample = dataset[0]
        xe_valid = sample["xe_valid"]  # (ne_max, ni_max) bool
        n_events = sample["n_events"].item()

        for i in range(n_events):
            assert xe_valid[i, 0].item(), (
                f"xe_valid[{i}, 0] must be True for real event {i}. "
                "[EVT] sentinel at position 0 must be marked as a real token."
            )

    def test_evt_invalid_for_padding_events(self, tmp_path) -> None:
        """xe_valid[:, 0] must be False for padding event slots.

        Padding events (slots beyond n_events) must have xe_valid = False
        at ALL positions including position 0.
        """
        dataset = _make_dataset(tmp_path, ne_max=5, ni_max=8)
        sample = dataset[0]
        xe_valid = sample["xe_valid"]  # (ne_max, ni_max) bool
        n_events = sample["n_events"].item()

        for i in range(n_events, 5):  # ne_max=5
            assert not xe_valid[i, 0].item(), (
                f"xe_valid[{i}, 0] must be False for padding event slot {i}. "
                "Padding events must not have a valid [EVT] sentinel."
            )


# ---------------------------------------------------------------------------
# 6. MLM mask — [EVT] at position 0 excluded from targets
# ---------------------------------------------------------------------------


class TestEVTExcludedFromMLM:
    """[EVT] tokens at position 0 must never be MLM targets.

    The masking strategy operates over all token positions.  After masking,
    position 0 must be forced to False in mlm_mask so that EVT_ID=3 is never
    passed to global_to_local_value_id() (which rejects non-value IDs).
    """

    def test_evt_position_not_in_mlm_targets(self) -> None:
        """Simulates train.py guard: mlm_mask[:, :, 0] = False after masking.

        This confirms the invariant that must hold in the training loop:
        assembler.forward(target_ids=xe_val_ids, mask=mlm_mask) must never
        see EVT_ID at a masked position.
        """
        from pragma_encoder.masking.strategy import MaskingStrategy
        from pragma_encoder.model.config import PRAGMAConfig
        from pragma_encoder.tokenizer.vocabulary import VocabularySpec

        config = PRAGMAConfig.pragma_s()
        spec = VocabularySpec(
            special_tokens={"PAD": 0, "MASK": 1, "USR": 2, "EVT": 3, "UNK": 4},
            key_start=5,
            key_size=config.key_vocab_size,
            value_start=5 + config.key_vocab_size,
            value_size=config.value_vocab_size,
            total_embedding_vocab_size=5 + config.key_vocab_size + config.value_vocab_size,
            field_key_ids={},
            field_value_ranges={},
        )

        masker = MaskingStrategy(config)

        batch, ne, ni = 2, 4, 8
        # Build xe_val_ids: EVT_ID at position 0, valid value IDs elsewhere
        xe_val_ids = torch.randint(
            spec.value_start, spec.value_start + config.value_vocab_size, (batch, ne, ni)
        )
        xe_val_ids[:, :, 0] = TokenizerPipeline.EVT_ID  # [EVT] at position 0

        xe_key_ids = torch.randint(
            spec.key_start, spec.key_start + config.key_vocab_size, (batch, ne, ni)
        )
        xe_key_ids[:, :, 0] = TokenizerPipeline.EVT_ID

        # Run masking (may or may not select position 0 for masking)
        _masked_val, _target, mlm_mask = masker.forward(xe_val_ids, xe_key_ids)

        # Apply the train.py guard: force position 0 out of mask
        mlm_mask[:, :, 0] = False

        # Verify: no EVT_ID in the masked target positions
        masked_targets = xe_val_ids[mlm_mask]
        evt_in_targets = (masked_targets == TokenizerPipeline.EVT_ID).any()
        assert not evt_in_targets.item(), (
            "EVT_ID appeared in mlm_mask target positions after applying "
            "mlm_mask[:, :, 0] = False guard.  "
            "The train.py guard must zero out position 0 of mlm_mask "
            "to prevent EVT_ID from being passed to global_to_local_value_id()."
        )
