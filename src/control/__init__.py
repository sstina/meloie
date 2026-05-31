"""Qt-free realtime control layer for the RVC voice changer.

``RealtimeSession`` owns the engine + the stream thread + the live metrics and
exposes a small lifecycle (load/start/stop/reload), live INPUT-side setters, and
a metrics snapshot — so a GUI (PySide6 / QML, a later phase) can bind to it
without touching the engine / audio internals. Importing this package is cheap:
torch / sounddevice / the engine are imported lazily inside the methods.
"""

from .session import RealtimeSession, SessionError, SessionState

__all__ = ["RealtimeSession", "SessionError", "SessionState"]
