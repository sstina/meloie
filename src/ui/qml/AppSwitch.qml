import QtQuick
import QtQuick.Controls
import App

// Accent toggle switch with a sliding knob.
Switch {
    id: control
    font.family: Theme.fontFamily
    font.pixelSize: Theme.fsBody
    spacing: Theme.s2

    indicator: Rectangle {
        implicitWidth: 40
        implicitHeight: 22
        x: control.leftPadding
        y: control.height / 2 - height / 2
        radius: height / 2
        color: control.checked ? Theme.accent : Theme.bgElevated
        border.width: 1
        border.color: control.checked ? Theme.accent : Theme.hairline
        Behavior on color { ColorAnimation { duration: Theme.durFast } }

        Rectangle {
            x: control.checked ? parent.width - width - 2 : 2
            y: 2
            width: 18; height: 18; radius: 9
            color: control.checked ? Theme.bgBase : Theme.textSecond
            scale: control.down ? 0.86 : 1.0      // squish on press
            Behavior on x { NumberAnimation { duration: Theme.durFast; easing.type: Easing.OutCubic } }
            Behavior on scale { NumberAnimation {
                duration: control.down ? Theme.durPress : Theme.durRelease; easing.type: Easing.OutCubic } }
        }
    }

    contentItem: Text {
        text: control.text
        font: control.font
        color: control.enabled ? Theme.textSecond : Theme.textMuted
        opacity: control.enabled ? 1.0 : 0.5
        verticalAlignment: Text.AlignVCenter
        leftPadding: control.indicator.width + control.spacing
    }
}
