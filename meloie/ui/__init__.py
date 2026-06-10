"""PySide6 + QML desktop GUI for the RVC realtime voice changer.

This package is the ONLY Qt-dependent code in the project. The engine, the
control facade (``meloie/control``), and the pure pytest suite never import Qt — so
keep all ``PySide6`` imports inside this package's modules (``backend`` / ``app``),
not at this package's top level. ``config_assembly`` here is deliberately Qt-free.

Run it with::

    . .\\setup_env_applio.ps1
    python -m meloie.ui
"""
