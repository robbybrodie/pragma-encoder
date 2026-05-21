"""Import and dependency graph tests.

Verifies that all src/ modules import correctly and
that the dependency graph is respected. No circular imports.

These tests catch import errors immediately rather than
discovering them mid-implementation when they are
harder to fix.
"""

import importlib
import inspect

import pytest


class TestModuleImports:
    """Verify all src/ modules import without errors."""

    def test_config_imports(self) -> None:
        """src/pragma_encoder/model/config.py must import cleanly."""
        import pragma_encoder.model.config  # noqa: F401

    def test_tokenizer_imports(self) -> None:
        """src/pragma_encoder/tokenizer/ must import cleanly."""
        import pragma_encoder.tokenizer  # noqa: F401

    def test_encoders_imports(self) -> None:
        """src/pragma_encoder/encoders/ must import cleanly."""
        import pragma_encoder.encoders  # noqa: F401

    def test_masking_imports(self) -> None:
        """src/pragma_encoder/masking/ must import cleanly."""
        import pragma_encoder.masking  # noqa: F401

    def test_model_imports(self) -> None:
        """src/pragma_encoder/model/ must import cleanly."""
        import pragma_encoder.model  # noqa: F401

    def test_adaptation_imports(self) -> None:
        """src/pragma_encoder/adaptation/ must import cleanly."""
        import pragma_encoder.adaptation  # noqa: F401

    def test_training_imports(self) -> None:
        """src/pragma_encoder/training/ must import cleanly."""
        import pragma_encoder.training  # noqa: F401

    def test_evaluation_imports(self) -> None:
        """src/pragma_encoder/evaluation/ must import cleanly."""
        import pragma_encoder.evaluation  # noqa: F401


class TestDependencyGraph:
    """Verify the dependency graph is respected.

    Dependencies must flow in one direction only:
      config → tokenizer → encoders → model → adaptation

    No module may import from a module that depends on it.
    """

    def test_config_has_no_src_dependencies(self) -> None:
        """config.py must not import from any other src/ module.

        It is the root of the dependency graph.
        """
        import pragma_encoder.model.config as config_module

        source = inspect.getsource(config_module)

        forbidden = [
            "from pragma_encoder.tokenizer",
            "from pragma_encoder.encoders",
            "from pragma_encoder.masking",
            "import pragma_encoder.tokenizer",
            "import pragma_encoder.encoders",
            "import pragma_encoder.masking",
        ]
        for pattern in forbidden:
            assert pattern not in source, (
                f"config.py must not import from pragma_encoder.*: found '{pattern}'"
            )

    def test_tokenizer_does_not_import_encoders(self) -> None:
        """src/pragma_encoder/tokenizer/ must not import from src/pragma_encoder/encoders/."""
        import pragma_encoder.tokenizer as tok_module

        source = inspect.getsource(tok_module)

        forbidden = [
            "from pragma_encoder.encoders",
            "import pragma_encoder.encoders",
            "from pragma_encoder.model",
            "import pragma_encoder.model",
        ]
        for pattern in forbidden:
            assert pattern not in source, (
                f"tokenizer must not import from encoders or model: found '{pattern}'"
            )

    def test_encoders_do_not_import_model(self) -> None:
        """src/pragma_encoder/encoders/ must not import from src/pragma_encoder/model/pragma.py.

        Encoders depend on config and rope only.
        They must not create circular dependencies with the model.
        """
        import pragma_encoder.encoders as enc_module

        source = inspect.getsource(enc_module)

        forbidden = [
            "from pragma_encoder.model.pragma",
            "import pragma_encoder.model.pragma",
            "from pragma_encoder.adaptation",
            "import pragma_encoder.adaptation",
        ]
        for pattern in forbidden:
            assert pattern not in source, (
                f"encoders must not import from model or adaptation: found '{pattern}'"
            )

    def test_no_circular_imports(self) -> None:
        """Full import chain must not create circular dependencies.

        Import all modules in dependency order.
        If any raises ImportError due to circular dependency
        this test fails with a clear message.
        """
        modules_in_order = [
            "pragma_encoder.model.config",
            "pragma_encoder.tokenizer",
            "pragma_encoder.encoders.rope",
            "pragma_encoder.encoders",
            "pragma_encoder.masking",
            "pragma_encoder.model",
            "pragma_encoder.adaptation",
            "pragma_encoder.training",
            "pragma_encoder.evaluation",
        ]

        for module_path in modules_in_order:
            try:
                importlib.import_module(module_path)
            except ImportError as e:
                if "circular" in str(e).lower() or "cannot import" in str(e).lower():
                    pytest.fail(
                        f"Circular import detected in {module_path}: {e}\n"
                        f"Fix the dependency graph. See DEVELOPMENT_PROCESS.md."
                    )


class TestNamingConventions:
    """Verify key classes exist with the correct names.

    Names must match DEVELOPMENT_PROCESS.md naming conventions.
    Wrong names here mean the naming conventions were not followed.
    These tests will initially fail (nothing implemented yet).
    They define what success looks like.
    """

    def test_pragma_config_class_exists(self) -> None:
        """PRAGMAConfig must exist in src/model/config.py."""
        from pragma_encoder.model.config import PRAGMAConfig  # noqa: F401

    def test_pragma_config_has_three_variants(self) -> None:
        """PRAGMAConfig must have pragma_s, pragma_m, pragma_l classmethods."""
        from pragma_encoder.model.config import PRAGMAConfig

        assert hasattr(PRAGMAConfig, "pragma_s"), (
            "PRAGMAConfig must have a pragma_s() classmethod"
        )
        assert hasattr(PRAGMAConfig, "pragma_m"), (
            "PRAGMAConfig must have a pragma_m() classmethod"
        )
        assert hasattr(PRAGMAConfig, "pragma_l"), (
            "PRAGMAConfig must have a pragma_l() classmethod"
        )

    def test_profile_state_encoder_class_name(self) -> None:
        """Class must be named ProfileStateEncoder exactly."""
        from pragma_encoder.encoders.profile_state_encoder import ProfileStateEncoder  # noqa: F401

    def test_event_encoder_class_name(self) -> None:
        """Class must be named EventEncoder exactly."""
        from pragma_encoder.encoders.event_encoder import EventEncoder  # noqa: F401

    def test_history_encoder_class_name(self) -> None:
        """Class must be named HistoryEncoder exactly."""
        from pragma_encoder.encoders.history_encoder import HistoryEncoder  # noqa: F401

    def test_rope_class_name(self) -> None:
        """Class must be named RoPEEncoding exactly."""
        from pragma_encoder.encoders.rope import RoPEEncoding  # noqa: F401

    def test_masking_strategy_class_name(self) -> None:
        """Class must be named MaskingStrategy exactly."""
        from pragma_encoder.masking.strategy import MaskingStrategy  # noqa: F401

    def test_pragma_model_class_name(self) -> None:
        """Full model must be named PRAGMA exactly."""
        from pragma_encoder.model.pragma import PRAGMA  # noqa: F401

    def test_lora_adapter_class_name(self) -> None:
        """LoRA adapter must be named LoRAAdapter exactly."""
        from pragma_encoder.adaptation.lora import LoRAAdapter  # noqa: F401

    def test_embedding_probe_class_name(self) -> None:
        """Embedding probe must be named EmbeddingProbe exactly."""
        from pragma_encoder.adaptation.probe import EmbeddingProbe  # noqa: F401
