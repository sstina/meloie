import QtQuick
import QtQuick.Controls
import App

// Single-line text field. At rest it matches AppComboBox exactly (same bgElevated
// fill + hairline border + radiusMd) so the whole form reads as one family. The
// accent appears ONLY on focus -- the field's one unmissable high moment: the
// border lights up, a soft accent glow ring floats just outside, the caret turns
// accent-colored, and the selection tints to match. `accent` defaults to mint
// (Theme's "active/interactive"); pass a domain hue (e.g. Theme.resonance for a
// merge field, Theme.pitch for a pitch field) to focus in that color.
//
// Brightness, not color, carries the hierarchy: label (textSecond) -> placeholder
// (textMuted, dimmest, so "empty" reads at a glance) -> entered text (textPrimary,
// brightest).
TextField {
    id: control
    property color accent: Theme.accent

    implicitHeight: 34
    leftPadding: Theme.s3
    rightPadding: Theme.s3
    color: Theme.textPrimary                       // entered text = brightest
    placeholderTextColor: Theme.textMuted          // placeholder = dimmest
    font.family: Theme.fontFamily
    font.pixelSize: Theme.fsBody
    selectByMouse: true
    selectionColor: Qt.rgba(control.accent.r, control.accent.g, control.accent.b, 0.32)
    selectedTextColor: Theme.textPrimary

    // accent caret — the default caret matches the text color, so the mint/lilac
    // caret only appears if we draw it ourselves.
    cursorDelegate: Rectangle {
        width: 2
        radius: 1
        color: control.accent
        visible: control.cursorVisible
        SequentialAnimation on opacity {
            loops: Animation.Infinite
            running: control.cursorVisible
            NumberAnimation { to: 0; duration: 1 }
            PauseAnimation  { duration: 540 }
            NumberAnimation { to: 1; duration: 1 }
            PauseAnimation  { duration: 540 }
        }
    }

    background: Item {
        // focus glow ring — a translucent accent stroke just outside the field
        Rectangle {
            anchors.fill: parent
            anchors.margins: -3
            radius: Theme.radiusMd + 3
            color: "transparent"
            border.width: 3
            border.color: Qt.rgba(control.accent.r, control.accent.g, control.accent.b, 0.18)
            visible: control.activeFocus
            opacity: visible ? 1.0 : 0.0
            Behavior on opacity { NumberAnimation { duration: Theme.durFast } }
        }
        // the field itself — same fill + radius as AppComboBox (glass-gated)
        Rectangle {
            anchors.fill: parent
            radius: Theme.radiusMd
            color: Theme.glassEnabled ? Theme.glassField : Theme.bgElevated
            border.width: 1
            border.color: control.activeFocus ? control.accent
                         : (control.hovered ? Qt.rgba(1, 1, 1, 0.18) : Theme.hairline)
            Behavior on border.color { ColorAnimation { duration: Theme.durFast } }
        }
    }
}
