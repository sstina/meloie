import QtQuick
import QtQuick.Controls
import App

// Accent-filled primary button (Basic style). `flat: true` -> ghost/tool button.
Button {
    id: control
    property color accentColor: Theme.accent     // filled-button hue (Start=mint, Stop=coral, ...)

    font.family: Theme.fontFamily
    font.pixelSize: Theme.fsBody
    font.weight: Theme.fwMedium
    topPadding: Theme.s2
    bottomPadding: Theme.s2
    leftPadding: Theme.s4
    rightPadding: Theme.s4

    // press <100ms, ease back ~150ms with a tiny bounce — "pushed in" feel.
    scale: control.down ? Theme.pressScale : 1.0
    Behavior on scale {
        NumberAnimation {
            duration: control.down ? Theme.durPress : Theme.durRelease
            easing.type: control.down ? Easing.OutCubic : Easing.OutBack
            easing.overshoot: 1.4
        }
    }

    contentItem: Text {
        text: control.text
        font: control.font
        color: control.flat ? Theme.textSecond : Theme.bgBase
        opacity: control.enabled ? 1.0 : 0.4
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }

    background: Rectangle {
        implicitHeight: 34
        radius: Theme.radiusMd
        border.width: control.flat ? 1 : 0
        border.color: Theme.hairline
        color: {
            if (control.flat)
                return control.hovered ? Theme.bgElevated : "transparent";
            if (!control.enabled) return Theme.bgElevated;
            if (control.pressed)  return Qt.darker(control.accentColor, 1.2);
            return control.hovered ? Qt.lighter(control.accentColor, 1.15) : control.accentColor;
        }
        // snap the colour on press, ease it on hover/release
        Behavior on color { ColorAnimation { duration: control.down ? Theme.durPress : Theme.durFast } }
    }
}
