"""Real-GPU window capture of Main.qml (dev verification tool).

Unlike tools/check_qml.py (offscreen), this forces the REAL windows platform
plugin so layer effects / MultiEffect masks render exactly as on the user's
GPU — the offscreen platform cannot produce the left column's edge-dissolve
mask textures (a headless grab shows a blank column, a false alarm). Saves:

    .tmp/real_wide.png      1280x720 two-column, scrolled to top
    .tmp/real_bottom.png    same window, left column scrolled to the bottom

(Rebuilt 2026-06-10 — the original .tmp incarnation was lost; behavior follows
rvc.md §3.9. Lives in tools/ so it can't get lost again.)

Run in .venv-applio (needs a real display + GPU):
    . .\\setup_env_applio.ps1
    python tools\\real_shoot.py
"""

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "windows"     # REAL platform plugin, not offscreen

RVC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RVC)
os.chdir(RVC)

from PySide6.QtCore import QUrl, QTimer                      # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonType  # noqa: E402
from PySide6.QtQuickControls2 import QQuickStyle             # noqa: E402
from PySide6.QtWidgets import QApplication                   # noqa: E402

from meloie.ui.backend import Backend                        # noqa: E402

QML_DIR = os.path.join(RVC, "meloie", "ui", "qml")
OUT = os.path.join(RVC, ".tmp")


def main() -> int:
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

    def grab(name):
        img = win.grabWindow()
        p = os.path.join(OUT, name)
        img.save(p)
        print(f"saved {p}  ({img.width()}x{img.height()})")

    def scroll_left_column_to_bottom():
        # leftScroll is the left column's ScrollView; drive its Flickable.
        flick = win.findChild(object, "leftScroll")
        if flick is None:
            for obj in win.findChildren(object):
                if obj.metaObject().className().startswith("QQuickScrollView"):
                    flick = obj
                    break
        if flick is not None:
            content = flick.property("contentItem")
            try:
                ch = float(content.property("contentHeight"))
                vh = float(content.property("height"))
                content.setProperty("contentY", max(0.0, ch - vh))
                return True
            except Exception as exc:
                print(f"scroll failed: {exc}")
        else:
            print("left ScrollView not found; skipping bottom shot")
        return False

    step = {"n": 0}

    def tick():
        step["n"] += 1
        if step["n"] == 1:
            grab("real_wide.png")
            scrolled = scroll_left_column_to_bottom()
            QTimer.singleShot(800 if scrolled else 50, tick)
        elif step["n"] == 2:
            grab("real_bottom.png")
            backend.shutdown()
            app.quit()

    QTimer.singleShot(1500, tick)              # let layout + flowing light settle
    app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
