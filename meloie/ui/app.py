"""QApplication + QQmlApplicationEngine bootstrap for the RVC GUI."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonType
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from ..app_paths import app_base_dir, setup_frozen_cache_env
from .backend import Backend
from .tray import TrayController, load_app_icon

# QML is bundled INSIDE the build. Frozen: it sits at <_MEIPASS>/meloie/ui/qml (see
# meloie.spec datas) — resolve via _MEIPASS, not __file__ (which points into the PYZ).
# Source: resolve next to this module.
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
if getattr(sys, "frozen", False):
    _QML_DIR = os.path.join(getattr(sys, "_MEIPASS", _UI_DIR), "meloie", "ui", "qml")
else:
    _QML_DIR = os.path.join(_UI_DIR, "qml")
# External data (icon.svg, models/ incl. predictors+embedders, config/) lives
# next to the .exe when frozen, else at the source root.
RVC_ROOT = app_base_dir()


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


def _run_selftest(engine, deep: bool = False) -> int:
    """Frozen-build completeness probe (set MELOIE_SELFTEST=1, ideally with
    QT_QPA_PLATFORM=offscreen): verify the QML graph loaded AND the heavy lazy stack
    is importable in the bundle — torch + ``meloie.core.pipeline`` (the exact import
    the engine does at Start) + the key native deps — WITHOUT touching audio. With
    ``deep`` (MELOIE_SELFTEST=2) it ALSO loads the first discovered model and runs
    one ``process_block`` on synthetic audio — true end-to-end, no audio hardware.
    Prints one ``SELFTEST OK|FAIL`` line and returns 0/1."""
    results: list[str] = []
    ok = True

    def probe(name: str, fn) -> None:
        nonlocal ok
        try:
            fn()
            results.append(f"{name}=OK")
        except Exception as exc:
            ok = False
            results.append(f"{name}=FAIL({type(exc).__name__}: {exc})")

    probe("qml_root", lambda: None if engine.rootObjects() else (_ for _ in ()).throw(RuntimeError("no root object")))
    probe("torch", lambda: __import__("torch"))

    probe("meloie.core.pipeline",
          lambda: __import__("importlib").import_module("meloie.core.pipeline"))
    for mod in ("faiss", "torchaudio", "torchfcpe", "transformers", "sounddevice", "librosa"):
        probe(mod, lambda m=mod: __import__(m))

    if deep:
        def _engine_probe() -> str:
            import numpy as np
            from .config_assembly import build_configs_for_model, list_model_files
            from ..engine.streaming_engine import StreamingRvcEngine
            models = list_model_files()
            if not models:
                return "SKIP (no models staged next to exe)"
            scfg, _ = build_configs_for_model(models[0]["path"], None, None)
            eng = StreamingRvcEngine(scfg)
            eng.load()                                  # full pipeline: v2 guard, embedder, predictors, faiss
            x = (np.random.default_rng(0).standard_normal(eng.block_frame).astype(np.float32) * 0.1)
            out = eng.process_block(x, eng.stream_sr)   # one real inference block (no audio device)
            finite = bool(np.all(np.isfinite(out)))
            return f"OK (dev={eng.resolved_device}, model={models[0]['name']}, out_finite={finite}, n={int(out.shape[0])})"
        try:
            results.append("engine_load+block=" + _engine_probe())
        except Exception as exc:
            ok = False
            results.append(f"engine_load+block=FAIL({type(exc).__name__}: {exc})")

    print("SELFTEST " + ("OK" if ok else "FAIL") + " :: " + " | ".join(results), flush=True)
    return 0 if ok else 1


def main() -> int:
    # Frozen exe: redirect caches next to the .exe (the launch ps1 won't have run).
    setup_frozen_cache_env()

    # Containment (zero C: writes): Qt caches to %LOCALAPPDATA%/<AppName>/cache on
    # Windows via QStandardPaths, ignoring our HF/XDG redirects. TWO separate caches
    # land there: the QML disk cache (compiled .qmlc) and the RHI graphics PIPELINE
    # cache (qtpipelinecache). Disable BOTH (tiny recompile cost) so the .exe never
    # creates anything on C:. MUST precede any QApplication/QQmlEngine init.
    os.environ.setdefault("QML_DISABLE_DISK_CACHE", "1")
    os.environ.setdefault("QSG_RHI_DISABLE_DISK_CACHE", "1")

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

    # Frozen-build / CI completeness probe: verify QML + the heavy stack, then exit
    # WITHOUT entering the event loop (no window, no audio). Guarded by env -> normal
    # launch is unaffected.
    _selftest = os.environ.get("MELOIE_SELFTEST")
    if _selftest in ("1", "2"):
        return _run_selftest(engine, deep=(_selftest == "2"))

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
