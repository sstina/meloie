"""QApplication + QQmlApplicationEngine bootstrap for the RVC GUI."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonType
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication

from .backend import Backend

_UI_DIR = os.path.dirname(os.path.abspath(__file__))
_QML_DIR = os.path.join(_UI_DIR, "qml")
RVC_ROOT = os.path.dirname(os.path.dirname(_UI_DIR))   # .../RVC


def main() -> int:
    # Resolve model/profile/index relative paths against the project root.
    if os.path.abspath(os.getcwd()) != os.path.abspath(RVC_ROOT):
        os.chdir(RVC_ROOT)

    # Basic style so our hand-styled controls (accent slider/switch/button/combo)
    # fully apply, instead of the Windows-native style clashing with the dark glass.
    QQuickStyle.setStyle("Basic")

    app = QApplication(sys.argv)
    app.setApplicationName("RVC Voice Changer")

    # Theme tokens + glass switches live in a QML singleton under the "App" URI.
    # qmlRegisterSingletonType(url, ...) avoids the local-dir module-name matching
    # friction of a qmldir module; same-dir components still resolve by filename.
    qmlRegisterSingletonType(
        QUrl.fromLocalFile(os.path.join(_QML_DIR, "Theme.qml")), "App", 1, 0, "Theme"
    )

    engine = QQmlApplicationEngine()
    backend = Backend()
    engine.rootContext().setContextProperty("backend", backend)
    engine.addImportPath(_QML_DIR)
    engine.load(QUrl.fromLocalFile(os.path.join(_QML_DIR, "Main.qml")))
    if not engine.rootObjects():
        print("ERROR: failed to load Main.qml", file=sys.stderr)
        return 1

    app.aboutToQuit.connect(backend.shutdown)
    return app.exec()
