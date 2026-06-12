"""Headless QML load + glass-composite contrast check (dev verification tool).

Mirrors meloie/ui/app.py setup under the offscreen platform: registers the Theme
singleton, forces Basic style, loads Main.qml, and FAILS on any QML warning or a
missing root object. Then grabs SEVERAL rendered frames across the animated
flowing-light drift and measures the WORST-CASE WCAG contrast of
textPrimary / textSecond / textMuted against the worst (brightest)
*actually composited* background pixel — catching the case where a drifting
blob lifts panel luminance and hurts text. Text anti-aliasing edges are
excluded by requiring a locally-uniform neighborhood around sampled pixels.

(Rebuilt 2026-06-10 — the original .tmp incarnation was lost; behavior and
output format follow rvc.md §3.9. Lives in tools/ so it can't get lost again.)

Run in .venv-applio:
    . .\\setup_env_applio.ps1
    python tools\\check_qml.py

NOTE (glass / vibrancy): the offscreen platform cannot render ShaderEffectSource
/ MultiEffect mask textures, so with the vibrancy glass enabled the contrast
number here UNDERESTIMATES the real composited brightness. This tool remains
the authoritative LOAD + 0-warning gate; the authoritative CONTRAST gate on
glass is tools/real_shoot.py --probe (real GPU, blob-parked worst case).

Exit codes: 0 OK / 1 load failure or warnings / 2 contrast below 4.5 for an
essential text role.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

RVC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RVC)
os.chdir(RVC)

from PySide6.QtCore import QUrl, QTimer                      # noqa: E402
from PySide6.QtGui import QColor                             # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonType  # noqa: E402
from PySide6.QtQuickControls2 import QQuickStyle             # noqa: E402
from PySide6.QtWidgets import QApplication                   # noqa: E402

from meloie.ui.backend import Backend                        # noqa: E402

QML_DIR = os.path.join(RVC, "meloie", "ui", "qml")


def theme_color(token: str) -> QColor:
    """Read a color token straight out of Theme.qml — a hardcoded copy here once
    drifted (#B7C2CE vs the real textSecond #94A3B8) and silently inflated the
    contrast gate, so the source file is the only allowed source of truth."""
    import re
    with open(os.path.join(QML_DIR, "Theme.qml"), encoding="utf-8") as fh:
        m = re.search(rf'property color {token}:\s*"(#[0-9A-Fa-f]{{6,8}})"', fh.read())
    if not m:
        raise RuntimeError(f"Theme.qml token not found: {token}")
    return QColor(m.group(1))


TEXT_PRIMARY = theme_color("textPrimary")
TEXT_SECOND = theme_color("textSecond")
TEXT_MUTED = theme_color("textMuted")

N_FRAMES = 5            # spread across the flowing-light drift
FRAME_SPACING_MS = 450
PIXEL_STEP = 3          # sample every 3rd pixel (speed)


def _lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _lum(qc: QColor) -> float:
    return 0.2126 * _lin(qc.red()) + 0.7152 * _lin(qc.green()) + 0.0722 * _lin(qc.blue())


def _contrast(fg: QColor, bg: QColor) -> float:
    l1, l2 = sorted((_lum(fg), _lum(bg)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def _near(c: QColor, t: QColor, delta: int = 48) -> bool:
    return (abs(c.red() - t.red()) + abs(c.green() - t.green())
            + abs(c.blue() - t.blue())) <= delta


def _worst_bg_pixel(img):
    """Brightest BACKGROUND-LIKE pixel. Text pixels are excluded by COLOR
    PROXIMITY to the three text tokens (a plain luminance cutoff would also
    skip a vibrancy-brightened card background — exactly the pixels under
    test); very bright pixels (accents / level bars / pills) are excluded by a
    generous luminance ceiling; AA edges by local uniformity."""
    w, h = img.width(), img.height()
    worst = None
    worst_lum = -1.0
    for y in range(2, h - 2, PIXEL_STEP):
        for x in range(2, w - 2, PIXEL_STEP):
            c = img.pixelColor(x, y)
            lum = _lum(c)
            if lum > 0.30:                       # accent fills / level bars / glyph cores
                continue
            if _near(c, TEXT_PRIMARY) or _near(c, TEXT_SECOND) or _near(c, TEXT_MUTED):
                continue                          # text pixel, not background
            # local uniformity: all 4 neighbors within a small delta
            uniform = True
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                n = img.pixelColor(x + dx, y + dy)
                if (abs(n.red() - c.red()) + abs(n.green() - c.green())
                        + abs(n.blue() - c.blue())) > 18:
                    uniform = False
                    break
            if uniform and lum > worst_lum:
                worst_lum = lum
                worst = c
    return worst


def main() -> int:
    app = QApplication(sys.argv)
    QQuickStyle.setStyle("Basic")
    qmlRegisterSingletonType(
        QUrl.fromLocalFile(os.path.join(QML_DIR, "Theme.qml")), "App", 1, 0, "Theme"
    )
    engine = QQmlApplicationEngine()
    warnings = []
    engine.warnings.connect(lambda errs: warnings.extend(str(e) for e in errs))
    backend = Backend()
    engine.rootContext().setContextProperty("backend", backend)
    engine.addImportPath(QML_DIR)
    engine.load(QUrl.fromLocalFile(os.path.join(QML_DIR, "Main.qml")))

    if not engine.rootObjects():
        print("LOAD FAILED: no root object")
        for wmsg in warnings:
            print("  warning:", wmsg)
        return 1
    win = engine.rootObjects()[0]
    print(f"devices: {len(backend.devices)}  models: {len(backend.models)}")

    state = {"frame": 0, "worst": None, "worst_lum": -1.0, "rc": 0}

    def grab_frame():
        img = win.grabWindow()
        bg = _worst_bg_pixel(img)
        if bg is not None and _lum(bg) > state["worst_lum"]:
            state["worst_lum"] = _lum(bg)
            state["worst"] = bg
        state["frame"] += 1
        if state["frame"] < N_FRAMES:
            QTimer.singleShot(FRAME_SPACING_MS, grab_frame)
        else:
            finish()

    def finish():
        bg = state["worst"]
        if bg is None:
            print("no background pixel found (unexpected)")
            state["rc"] = 1
        else:
            cp = _contrast(TEXT_PRIMARY, bg)
            cs = _contrast(TEXT_SECOND, bg)
            cm = _contrast(TEXT_MUTED, bg)
            print(f"worst composited bg px  = {bg.name().upper()}  "
                  f"[worst of {N_FRAMES} frames]")
            print(f"  textPrimary contrast = {cp:.2f}:1 "
                  f"({'OK' if cp >= 4.5 else 'FAIL'})")
            print(f"  textSecond  contrast = {cs:.2f}:1 "
                  f"({'OK' if cs >= 4.5 else 'FAIL'})")
            print(f"  textMuted   contrast = {cm:.2f}:1 "
                  f"({'OK' if cm >= 4.5 else 'sub-4.5, non-essential by design'})")
            if cp < 4.5 or cs < 4.5:
                state["rc"] = 2
        print(f"warnings: {len(warnings)}")
        for wmsg in warnings:
            print("  warning:", wmsg)
        if warnings and state["rc"] == 0:
            state["rc"] = 1
        print("LOAD OK" if state["rc"] == 0 else "CHECK FAILED")
        backend.shutdown()
        app.quit()

    QTimer.singleShot(1200, grab_frame)        # let layout + animation settle
    app.exec()
    return state["rc"]


if __name__ == "__main__":
    sys.exit(main())
