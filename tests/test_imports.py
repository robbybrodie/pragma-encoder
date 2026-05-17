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
        """src/model/config.py must import cleanly."""
        import src.model.config  # noqa: F401

    def test_tokenizer_imports(self) -> None:
        """src/tokenizer/ must import cleanly."""
        import src.tokenizer  # noqa: F401

    def test_encoders_imports(self) -> None:
        """src/encoders/ must import cleanly."""
        import src.encoders  # noqa: F401

    def test_masking_imports(self) -> None:
        """src/masking/ must import cleanly."""
        import src.masking  # noqa: F401

    def test_model_imports(self) -> None:
        """src/model/ must import cleanly."""
        import src.model  # noqa: F401

    def test_adaptation_imports(self) -> None:
        """src/adaptation/ must import cleanly."""
        import src.adaptation  # noqa: F401

    def test_training_imports(self) -> None:
        """src/training/ must import cleanly."""
        import src.training  # noqa: F401

    def test_evaluation_imports(self) -> None:
        """src/evaluation/ must import cleanly."""
        import src.evaluation  # noqa: F401


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
        import src.model.config as config_module

        source = inspect.getsource(config_module)

        forbidden = [
            "from src.tokenizer",
            "from src.encoders",
            "from src.masking",
            "import src.tokenizer",
            "import src.encoders",
            "import src.masking",
        ]
        for pattern in forbidden:
            assert pattern not in source, (
                f"config.py must not import from src/: found '{pattern}'"
            )

    def test_tokenizer_does_not_import_encoders(self) -> None:
        """src/tokenizer/ must not import from src/encoders/."""
        import src.tokenizer as tok_module

        source = inspect.getsource(tok_module)

        forbidden = [
            "from src.encoders",
            "import src.encoders",
            "from src.model",
            "import src.model",
        ]
        for pattern in forbidden:
            assert pattern not in source, (
                f"tokenizer must not import from encoders or model: found '{pattern}'"
            )

    def test_encoders_do_not_import_model(self) -> None:
        """src/encoders/ must not import from src/model/pragma.py.

        Encoders depend on config and rope only.
        They must not create circular dependencies with the model.
        """
        import src.encoders as enc_module

        source = inspect.getsource(enc_module)

        forbidden = [
            "from src.model.pragma",
            "import src.model.pragma",
            "from src.adaptation",
            "import src.adaptation",
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
            "src.model.config",
            "src.tokenizer",
            "src.encoders.rope",
            "src.encoders",
            "src.masking",
            "src.model",
            "src.adaptation",
            "src.training",
            "src.evaluation",
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
        from src.model.config import PRAGMAConfig  # noqa: F401

    def test_pragma_config_has_three_variants(self) -> None:
        """PRAGMAConfig must have pragma_s, pragma_m, pragma_l classmethods."""
        from src.model.config import PRAGMAConfig

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
        from src.encoders.profile_state_encoder import ProfileStateEncoder  # noqa: F401

    def test_event_encoder_class_name(self) -> None:
        """Class must be named EventEncoder exactly."""
        from src.encoders.event_encoder import EventEncoder  # noqa: F401

    def test_history_encoder_class_name(self) -> None:
        """Class must be named HistoryEncoder exactly."""
        from src.encoders.history_encoder import HistoryEncoder  # noqa: F401

    def test_rope_class_name(self) -> None:
        """Class must be named RoPEEncoding exactly."""
        from src.encoders.rope import RoPEEncoding  # noqa: F401

    def test_masking_strategy_class_name(self) -> None:
        """Class must be named MaskingStrategy exactly."""
        from src.masking.strategy import MaskingStrategy  # noqa: F401

    def test_pragma_model_class_name(self) -> None:
        """Full model must be named PRAGMA exactly."""
        from src.model.pragma import PRAGMA  # noqa: F401

    def test_lora_adapter_class_name(self) -> None:
        """LoRA adapter must be named LoRAAdapter exactly."""
        from src.adaptation.lora import LoRAAdapter  # noqa: F401

    def test_embedding_probe_class_name(self) -> None:
        """Embedding probe must be named EmbeddingProbe exactly."""
        from src.adaptation.probe import EmbeddingProbe  # noqa: F401
