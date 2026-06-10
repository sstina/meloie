"""PyInstaller entry point for the frozen Meloie exe.

Uses ABSOLUTE imports (unlike ``src/ui/__main__.py``, whose relative imports fail
when PyInstaller runs the entry script as ``__main__``). Sets the frozen cache
redirection BEFORE importing the app (so no heavy import can touch C:), then hands
off to the same ``app.main()`` the source launch uses.
"""

import sys

from src.app_paths import setup_frozen_cache_env

setup_frozen_cache_env()

from src.ui.app import main  # noqa: E402  (must follow the cache-env setup)

if __name__ == "__main__":
    sys.exit(main())
