"""Compatibility wrapper — training entrypoint moved to pragma_encoder.training.train.

The canonical entrypoint is now the installed console script::

    pragma-encoder-train --help

Or via module execution::

    python -m pragma_encoder.training.train --help

This wrapper exists for one release so that existing invocations of
``python scripts/train_pragma.py`` continue to work without changes.

Reference: pragma_encoder.training.train (package module)
"""

from pragma_encoder.training.train import main

raise SystemExit(main())
