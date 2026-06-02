"""QApplication + QQmlApplicationEngine bootstrap for the RVC GUI."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonType
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from .backend import Backend
from .tray import TrayController, load_app_icon

_UI_DIR = os.path.dirname(os.path.abspath(__file__))
_QML_DIR = os.path.join(_UI_DIR, "qml")
RVC_ROOT = os.path.dirname(os.path.dirname(_UI_DIR))   # .../RVC


def _apply_dark_titlebar(window) -> None:
    """Recolor the native Windows title bar to match the dark theme via DWM.

    Per-window and cosmetic only — keeps every native behaviour (minimise /
    maximise / close, drag, snap, double-click-maximise); it just makes the
    caption, its text, and the thin frame dark so the bar blends into the app
    instead of a light-grey native strip. Colors mirror Theme.qml (bgBase /
    textPrimary / bgElevated). Best-effort: a no-op on non-Windows or pre-Win11
    builds where the attributes are unsupported (DwmSetWindowAttribute then just
    returns an error HRESULT — never raises), so it can never break startup."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        dwm = ctypes.windll.dwmapi
        hwnd = ctypes.c_void_p(int(window.winId()))      # forces native handle creation

        def _set(attr: int, value: int) -> None:
            v = ctypes.c_int(value)
            dwm.DwmSetWindowAttribute(hwnd, ctypes.c_uint(attr),
                                      ctypes.byref(v), ctypes.sizeof(v))

        def _colorref(r: int, g: int, b: int) -> int:
            return (b << 16) | (g << 8) | r              # DWM COLORREF = 0x00BBGGRR

        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        DWMWA_BORDER_COLOR = 34
        DWMWA_CAPTION_COLOR = 35
        DWMWA_TEXT_COLOR = 36
        _set(DWMWA_USE_IMMERSIVE_DARK_MODE, 1)                   # light glyphs on a dark bar
        _set(DWMWA_CAPTION_COLOR, _colorref(0x1A, 0x21, 0x29))   # Theme.bgSurface (a touch lighter than bgBase)
        _set(DWMWA_TEXT_COLOR,    _colorref(0xF1, 0xF5, 0xF9))   # Theme.textPrimary
        _set(DWMWA_BORDER_COLOR,  _colorref(0x23, 0x2D, 0x38))   # Theme.bgElevated (faint edge)
    except Exception:
        pass     # unsupported build / API absent -> keep the default native bar


def main() -> int:
    # Resolve model/profile/index relative paths against the project root.
    if os.path.abspath(os.getcwd()) != os.path.abspath(RVC_ROOT):
        os.chdir(RVC_ROOT)

    # Basic style so our hand-styled controls (accent slider/switch/button/combo)
    # fully apply, instead of the Windows-native style clashing with the dark glass.
    QQuickStyle.setStyle("Basic")

    app = QApplication(sys.argv)
    app.setApplicationName("Meloie")

    # App / taskbar / tray icon (rendered from icon.svg; multi-size for crispness).
    icon = load_app_icon(os.path.join(RVC_ROOT, "icon.svg"))
    if not icon.isNull():
        app.setWindowIcon(icon)

    # Theme tokens + glass switches live in a QML singleton under the "App" URI.
    # qmlRegisterSingletonType(url, ...) avoids the local-dir module-name matching
    # friction of a qmldir module; same-dir components still resolve by filename.
    qmlRegisterSingletonType(
        QUrl.fromLocalFile(os.path.join(_QML_DIR, "Theme.qml")), "App", 1, 0, "Theme"
    )

    # Minimize-to-tray only when a tray actually exists; otherwise close = quit so
    # the app can never become unquittable. The QML close handler reads this gate.
    tray_available = QSystemTrayIcon.isSystemTrayAvailable()

    engine = QQmlApplicationEngine()
    backend = Backend()
    backend.set_tray_active(tray_available)
    engine.rootContext().setContextProperty("backend", backend)
    engine.addImportPath(_QML_DIR)
    engine.load(QUrl.fromLocalFile(os.path.join(_QML_DIR, "Main.qml")))
    if not engine.rootObjects():
        print("ERROR: failed to load Main.qml", file=sys.stderr)
        return 1

    # Recolor the native Windows title bar to match the dark theme (best-effort).
    _apply_dark_titlebar(engine.rootObjects()[0])

    tray = None
    if tray_available:
        # closing/hiding windows must NOT auto-quit — only the tray 退出 (or an
        # explicit app.quit) exits, so the stream survives a window close.
        app.setQuitOnLastWindowClosed(False)
        tray = TrayController(app, backend, icon)
        tray.attach_window(engine.rootObjects()[0])
        tray.show()

    app.aboutToQuit.connect(backend.shutdown)
    return app.exec()
