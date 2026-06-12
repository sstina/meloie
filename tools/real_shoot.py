"""Real-GPU window capture + (with --probe) the authoritative glass contrast gate.

Unlike tools/check_qml.py (offscreen), this forces the REAL windows platform
plugin so layer effects / ShaderEffectSource / MultiEffect masks render exactly
as on the user's GPU — offscreen cannot produce those textures (a headless grab
shows blank where they live, a false alarm).

Modes
-----
default        screenshots only:
                   .tmp/real_wide.png     1280x720, left column at top
                   .tmp/real_bottom.png   left column scrolled to bottom
--probe        the CONTRAST HARD GATE for the vibrancy glass (rvc.md §3.9):
               for every GlassPanel (objectName-tagged in Main.qml), park each
               background blob at its nearest REACHABLE drift extreme toward
               that card (worst composited brightness the animation can ever
               produce), grab, and measure WCAG contrast of textPrimary /
               textSecond against the worst background-like pixel INSIDE the
               card. Collapsible panels (高级/融合) are expanded first; both
               scroll extremes are probed; a free-drift frame series is added
               as a sanity check. Gate: >=4.5:1 for both roles on every card.

Run in .venv-applio (needs a real display + GPU):
    . .\\setup_env_applio.ps1
    python tools\\real_shoot.py [--probe]

Exit codes: 0 OK / 1 load-or-tooling failure / 2 contrast gate FAILED.
"""

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "windows"     # REAL platform plugin, not offscreen

RVC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RVC)
os.chdir(RVC)

from PySide6.QtCore import QObject, QPointF, QTimer, QUrl              # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonType  # noqa: E402
from PySide6.QtQuickControls2 import QQuickStyle              # noqa: E402
from PySide6.QtWidgets import QApplication                    # noqa: E402

from meloie.ui.backend import Backend                         # noqa: E402
# shared color math + Theme-parsed text tokens (check_qml's setdefault can no
# longer override QT_QPA_PLATFORM — it was hard-set above before this import)
from tools.check_qml import (TEXT_MUTED, TEXT_PRIMARY, TEXT_SECOND,  # noqa: E402
                             _contrast, _lum, _near)

QML_DIR = os.path.join(RVC, "meloie", "ui", "qml")
OUT = os.path.join(RVC, ".tmp")

PANELS = ["panelHeader", "panelDevices", "panelCreative",
          "panelAdvanced", "panelMonitor", "panelPrecise"]
GATE = 4.5
FREE_FRAMES = 8
FREE_SPACING_MS = 600
PIXEL_STEP = 2


CELL = 4          # text-adjacency grid cell (px); radius 1 cell => bg within ~8px of text


def _worst_bg_in_rect(img, x0, y0, x1, y1):
    """Worst background pixel ADJACENT TO TEXT inside a card rect.

    WCAG contrast is text-vs-its-own-background, so we only score background
    pixels within ~8px of an actual textPrimary/textSecond glyph — that
    excludes UI chrome (slider tracks, switch fills, accent pills) that a
    whole-card sweep would misreport as "background". Foreground exclusions:
    proximity to any text token (incl. textMuted — it IS foreground), bright
    accents (lum), saturated fills (chroma), AA edges (local uniformity)."""
    x0, y0 = max(2, int(x0)), max(2, int(y0))
    x1, y1 = min(img.width() - 2, int(x1)), min(img.height() - 2, int(y1))

    # pass 1: grid cells containing gated-role text pixels (delta 60: small
    # 11-12px CJK glyphs are AA-heavy, few pixels hit the token exactly; a
    # too-loose match only ADDS candidate area, and pass 2's exclusions still
    # decide what counts as background)
    text_cells = set()
    for y in range(y0, y1, PIXEL_STEP):
        for x in range(x0, x1, PIXEL_STEP):
            c = img.pixelColor(x, y)
            if _near(c, TEXT_PRIMARY, 60) or _near(c, TEXT_SECOND, 60):
                text_cells.add((x // CELL, y // CELL))
    if not text_cells:
        return None

    # pass 2: worst background candidate adjacent to a text cell
    worst, worst_lum = None, -1.0
    for y in range(y0, y1, PIXEL_STEP):
        for x in range(x0, x1, PIXEL_STEP):
            cx, cy = x // CELL, y // CELL
            if not any((cx + dx, cy + dy) in text_cells
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                continue
            c = img.pixelColor(x, y)
            lum = _lum(c)
            if lum > 0.30 or lum <= worst_lum:
                continue
            if (_near(c, TEXT_PRIMARY) or _near(c, TEXT_SECOND)
                    or _near(c, TEXT_MUTED)):
                continue
            # NOTE: no chroma cap — the vibrancy glass background is itself
            # saturated when a blob parks behind a card (that IS the pixel
            # under test), while bright accent FILLS are already excluded by
            # the luminance ceiling above.
            uniform = True
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                n = img.pixelColor(x + dx, y + dy)
                if (abs(n.red() - c.red()) + abs(n.green() - c.green())
                        + abs(n.blue() - c.blue())) > 18:
                    uniform = False
                    break
            if uniform:
                worst_lum, worst = lum, c
    return worst


def _visual_descendants(item, name, out):
    """Repeater-created delegates have NO QObject parent chain (QML/JS
    ownership), so findChildren can't see them — walk the VISUAL tree."""
    for ch in item.childItems():
        if ch.objectName() == name:
            out.append(ch)
        _visual_descendants(ch, name, out)


class Probe:
    def __init__(self, win, backend, app):
        self.win, self.backend, self.app = win, backend, app
        self.dpr = float(win.devicePixelRatio() or 1.0)
        self.bg = win.findChild(QObject, "appBackground")
        self.blobs = []
        if self.bg is not None:
            _visual_descendants(self.bg, "bgBlob", self.blobs)
        self.flick = None
        scroll = win.findChild(QObject, "leftScroll")
        if scroll is not None:
            self.flick = scroll.property("contentItem")
        self.worst = {}          # panel objectName -> (lum, QColor, state-tag)
        self.rc = 0

    # ---- scene helpers ----
    def panel_rect(self, panel, inset=8.0):
        """Panel interior rect in IMAGE (device-pixel) coords, or None."""
        tl = panel.mapToScene(QPointF(0, 0))
        w, h = float(panel.property("width")), float(panel.property("height"))
        x0, y0 = tl.x() + inset, tl.y() + inset
        x1, y1 = tl.x() + w - inset, tl.y() + h - inset
        ww, wh = float(self.win.width()), float(self.win.height())
        vx0, vy0, vx1, vy1 = max(x0, 0), max(y0, 0), min(x1, ww), min(y1, wh)
        if vx1 - vx0 < 40 or vy1 - vy0 < 24:
            return None                          # (mostly) off-viewport in this state
        if (vx1 - vx0) * (vy1 - vy0) < 0.5 * (x1 - x0) * (y1 - y0):
            return None                          # less than half visible -> other scroll state covers it
        return (vx0 * self.dpr, vy0 * self.dpr, vx1 * self.dpr, vy1 * self.dpr)

    def set_freeze(self, on):
        if self.bg is not None:
            self.bg.setProperty("probeFreeze", bool(on))

    def park_blobs_at(self, scene_cx, scene_cy):
        """Move every blob to its nearest REACHABLE drift position toward a
        point — the worst composited state the animation could ever produce."""
        for b in self.blobs:
            bx, by = float(b.property("bx")), float(b.property("by"))
            ax, ay = float(b.property("ampx")), float(b.property("ampy"))
            w = float(b.property("width"))
            tx = max(bx - ax, min(bx + ax, scene_cx - w / 2.0))
            ty = max(by - ay, min(by + ay, scene_cy - w / 2.0))
            b.setProperty("x", tx)
            b.setProperty("y", ty)

    def record(self, img, tag, only_panel=None):
        for name in PANELS:
            if only_panel is not None and name != only_panel:
                continue
            panel = self.win.findChild(QObject, name)
            if panel is None:
                continue
            r = self.panel_rect(panel)
            if r is None:
                continue
            c = _worst_bg_in_rect(img, *r)
            if c is not None and (name not in self.worst or _lum(c) > self.worst[name][0]):
                self.worst[name] = (_lum(c), c, tag)

    def finish(self):
        print(f"\n==== glass contrast gate (>= {GATE}:1, worst reachable frame) ====")
        missing = [n for n in PANELS if n not in self.worst]
        for name in PANELS:
            if name not in self.worst:
                continue
            lum, c, tag = self.worst[name]
            cp, cs = _contrast(TEXT_PRIMARY, c), _contrast(TEXT_SECOND, c)
            ok = cp >= GATE and cs >= GATE
            if not ok:
                self.rc = 2
            print(f"  {name:14s} worst={c.name().upper()} [{tag}]  "
                  f"primary={cp:.2f}:1  second={cs:.2f}:1  {'OK' if ok else 'FAIL'}")
        if missing:
            print(f"  NOT SAMPLED (objectName missing / never visible): {missing}")
            self.rc = self.rc or 1
        print("PROBE OK" if self.rc == 0 else "PROBE FAILED")


def main() -> int:
    probe_mode = "--probe" in sys.argv
    app = QApplication(sys.argv)
    QQuickStyle.setStyle("Basic")
    qmlRegisterSingletonType(
        QUrl.fromLocalFile(os.path.join(QML_DIR, "Theme.qml")), "App", 1, 0, "Theme"
    )
    engine = QQmlApplicationEngine()
    backend = Backend()
    engine.rootContext().setContextProperty("backend", backend)
    engine.addImportPath(QML_DIR)
    engine.load(QUrl.fromLocalFile(os.path.join(QML_DIR, "Main.qml")))
    if not engine.rootObjects():
        print("LOAD FAILED: no root object")
        return 1
    win = engine.rootObjects()[0]
    win.show()

    def grab(name=None):
        img = win.grabWindow()
        if name:
            img.save(os.path.join(OUT, name))
            print(f"saved {os.path.join(OUT, name)}  ({img.width()}x{img.height()})")
        return img

    def set_scroll(to_bottom):
        flick = win.findChild(QObject, "leftScroll")
        content = flick.property("contentItem") if flick is not None else None
        if content is None:
            print("left ScrollView not found")
            return
        ch = float(content.property("contentHeight"))
        vh = float(content.property("height"))
        content.setProperty("contentY", max(0.0, ch - vh) if to_bottom else 0.0)

    def expand_collapsibles():
        for tname in ("advToggle", "mergeToggle"):
            t = win.findChild(QObject, tname)
            if t is not None:
                t.setProperty("checked", True)

    state = {"rc": 0}

    # ---------------- screenshot mode (default) ----------------
    if not probe_mode:
        step = {"n": 0}

        def tick():
            step["n"] += 1
            if step["n"] == 1:
                grab("real_wide.png")
                set_scroll(True)
                QTimer.singleShot(800, tick)
            else:
                grab("real_bottom.png")
                backend.shutdown()
                app.quit()

        QTimer.singleShot(1500, tick)
        app.exec()
        return 0

    # ---------------- probe mode ----------------
    pr = Probe(win, backend, app)
    if pr.bg is None or not pr.blobs:
        print("PROBE TOOLING FAILURE: appBackground / bgBlob objectNames not found")
        return 1
    plan = []        # (description, thunk) executed with inter-step settle delays

    def parked_pass(tag):
        """For each panel: park all blobs toward it, settle, grab, record."""
        for name in PANELS:
            def mk(nm):
                def run():
                    panel = win.findChild(QObject, nm)
                    if panel is None:
                        return
                    tl = panel.mapToScene(QPointF(0, 0))
                    cx = tl.x() + float(panel.property("width")) / 2.0
                    cy = tl.y() + float(panel.property("height")) / 2.0
                    pr.park_blobs_at(cx, cy)
                return run
            plan.append((f"park->{name} [{tag}]", mk(name)))
            def mk_grab(nm, tg):
                def run():
                    img = grab(f"real_probe_{nm}_{tg}.png" if nm in ("panelCreative", "panelMonitor") else None)
                    # record ALL visible panels, not just the parked target — a
                    # frame parked for card X is also a valid (sub-worst) frame
                    # for every other card, and it makes coverage robust against
                    # a single flaky text-detection pass.
                    pr.record(img, f"parked/{tg}")
                return run
            plan.append((f"grab {name} [{tag}]", mk_grab(name, tag)))

    # plan: expand collapsibles -> freeze -> parked pass at top -> bottom ->
    # unfreeze -> free-drift frames -> finish
    plan.append(("expand collapsibles", expand_collapsibles))
    plan.append(("freeze blobs", lambda: pr.set_freeze(True)))
    plan.append(("scroll top", lambda: set_scroll(False)))
    parked_pass("top")
    plan.append(("scroll bottom", lambda: set_scroll(True)))
    parked_pass("bottom")
    plan.append(("unfreeze", lambda: pr.set_freeze(False)))
    plan.append(("scroll top", lambda: set_scroll(False)))
    for i in range(FREE_FRAMES):
        plan.append((f"free frame {i}", lambda: pr.record(grab(), "free")))

    idx = {"i": 0}

    def step():
        if idx["i"] >= len(plan):
            pr.finish()
            grab("real_probe_final.png")
            state["rc"] = pr.rc
            backend.shutdown()
            app.quit()
            return
        desc, thunk = plan[idx["i"]]
        idx["i"] += 1
        try:
            thunk()
        except Exception as exc:
            print(f"PROBE step failed ({desc}): {exc}")
            state["rc"] = 1
        # settle: parking/scroll/expansion needs layout + blur-layer frames;
        # free-drift frames need real time between them.
        delay = FREE_SPACING_MS if desc.startswith("free frame") else 250
        QTimer.singleShot(delay, step)

    QTimer.singleShot(1500, step)
    app.exec()
    return state["rc"]


if __name__ == "__main__":
    sys.exit(main())
