"""System tray icon + minimize-to-tray for the GUI (QtWidgets layer).

The voice changer is a background utility (you Start it, then use Discord/OBS/
games while it keeps routing the converted voice to CABLE Input), so it lives in
the system tray: closing the window hides it to the tray (the stream keeps
running) and you quit explicitly from the tray menu. Pure UI/lifecycle — touches
neither the engine nor the faithful-carrier contract.

The tray ``退出`` routes through ``app.quit()`` → ``aboutToQuit`` →
``Backend.shutdown`` (the hardened unbounded-wait thread join), so there is one
clean-exit path.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

APP_NAME = "RVC Voice Changer"

# tray + taskbar + alt-tab want different raster sizes; pre-render all of them.
_ICON_SIZES = (16, 20, 24, 32, 48, 64, 128, 256)

# Dark context menu matching the app's glass theme (Theme.qml tokens): bgSurface
# panel, soft hairline border + rounded corners, accent-tinted hover, textMuted
# disabled, hairline separator. Scoped to the menu (not app-wide).
_MENU_QSS = """
QMenu {
    background-color: #1A2129;
    color: #F1F5F9;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 10px;
    padding: 6px;
    font-family: "Segoe UI Variable Text";
    font-size: 13px;
}
QMenu::item {
    background: transparent;
    padding: 7px 18px 7px 10px;
    border-radius: 6px;
    margin: 1px 2px;
}
QMenu::item:selected {
    background-color: rgba(45, 212, 191, 0.18);
    color: #5EEAD4;
}
QMenu::item:disabled {
    color: #64748B;
}
QMenu::separator {
    height: 1px;
    background: rgba(255, 255, 255, 0.06);
    margin: 5px 10px;
}
QMenu::icon {
    padding-left: 8px;
}
"""

_GLYPH_COLOR = "#94A3B8"        # Theme.textSecond — present but calm on the dark menu


def _new_canvas(size: int, scale: int = 2):
    pm = QPixmap(size * scale, size * scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.scale(scale, scale)
    return pm, p


def _finish(pm: QPixmap, p: QPainter, scale: int = 2) -> QIcon:
    p.end()
    pm.setDevicePixelRatio(scale)        # crisp at the menu's logical size
    return QIcon(pm)


def _icon_show(size: int = 18) -> QIcon:
    """A little window: rounded outline + a top title-bar divider."""
    pm, p = _new_canvas(size)
    pen = QPen(QColor(_GLYPH_COLOR), 1.6)
    pen.setJoinStyle(Qt.RoundJoin)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    m = size * 0.2
    p.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), 2.0, 2.0)
    y = m + (size - 2 * m) * 0.3
    p.drawLine(QPointF(m, y), QPointF(size - m, y))
    return _finish(pm, p)


def _icon_stop(size: int = 18) -> QIcon:
    """A filled rounded square (the universal stop glyph)."""
    pm, p = _new_canvas(size)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(_GLYPH_COLOR))
    m = size * 0.28
    p.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), 2.4, 2.4)
    return _finish(pm, p)


def _icon_quit(size: int = 18) -> QIcon:
    """A power symbol: a ring with a gap at the top + a stem through it."""
    pm, p = _new_canvas(size)
    pen = QPen(QColor(_GLYPH_COLOR), 1.7)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    cx, cy, r = size / 2.0, size * 0.54, size * 0.27
    # Qt angles: 0 = 3 o'clock, CCW, 1/16°. start 140°, span 260° -> 100° gap at top.
    p.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r), int(140 * 16), int(260 * 16))
    p.drawLine(QPointF(cx, cy - r * 1.25), QPointF(cx, cy - r * 0.15))
    return _finish(pm, p)


def load_app_icon(svg_path: str) -> QIcon:
    """Render ``svg_path`` once to a multi-size :class:`QIcon` (crisp at every
    tray/taskbar size). Best-effort: falls back to the SVG icon engine, then an
    empty icon, and never raises."""
    icon = QIcon()
    try:
        from PySide6.QtSvg import QSvgRenderer
        renderer = QSvgRenderer(svg_path)
        if renderer.isValid():
            for size in _ICON_SIZES:
                pm = QPixmap(size, size)
                pm.fill(Qt.transparent)
                painter = QPainter(pm)
                renderer.render(painter)
                painter.end()
                icon.addPixmap(pm)
            if not icon.isNull():
                return icon
    except Exception:
        pass
    fallback = QIcon(svg_path)        # qsvgicon engine, if the plugin is present
    return fallback if not fallback.isNull() else icon


class TrayController(QObject):
    """Owns the ``QSystemTrayIcon`` + menu and the minimize-to-tray behaviour.

    Holds references to the app / backend / window so none are garbage-collected;
    the controller itself is kept alive by the caller (``app.main``)."""

    def __init__(self, app, backend, icon: QIcon, *, parent=None) -> None:
        super().__init__(parent)
        self._app = app
        self._backend = backend
        self._window = None
        self._notified = False        # one-time "still running" balloon

        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip(APP_NAME)

        menu = QMenu()
        menu.setStyleSheet(_MENU_QSS)             # dark, theme-matched (vs native light)
        self._act_show = menu.addAction(_icon_show(), "显示窗口")
        self._act_show.triggered.connect(self.show_window)
        self._act_stop = menu.addAction(_icon_stop(), "停止变声")
        self._act_stop.triggered.connect(self._backend.stop)
        self._act_stop.setEnabled(False)         # only meaningful while RUNNING
        menu.addSeparator()
        act_quit = menu.addAction(_icon_quit(), "退出")
        act_quit.triggered.connect(self._app.quit)
        self._menu = menu                         # keep alive

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_activated)
        self._backend.stateChanged.connect(self._on_state)

    def show(self) -> None:
        self._tray.show()

    def attach_window(self, window) -> None:
        """Bind the QML root window so the tray can restore/hide it and notice
        when it is hidden to the tray."""
        self._window = window
        if window is not None:
            window.visibleChanged.connect(self._on_window_visible)

    # ------------------------------------------------------------------ slots
    def _on_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.show_window()

    def show_window(self) -> None:
        w = self._window
        if w is None:
            return
        w.show()
        w.raise_()
        w.requestActivate()

    def _on_window_visible(self, visible: bool) -> None:
        if not visible and not self._notified and self._tray.isVisible():
            self._notified = True
            self._tray.showMessage(
                APP_NAME, "仍在后台运行 — 右键托盘图标可退出", self._tray.icon(), 4000
            )

    def _on_state(self, state: str) -> None:
        self._tray.setToolTip(f"{APP_NAME} — {state}")
        self._act_stop.setEnabled(state == "running")
